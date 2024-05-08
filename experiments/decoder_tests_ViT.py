import argparse
import torch
from models.vit import PatchDecoder
from experiments_utils import (
    load_raw_images,
    deactivate_dropout_layers,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    images, _ = load_raw_images(args.img_dir)
    images = images.to(device)

    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
    )
    deactivate_dropout_layers(model)
    model = model.to(device)

    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=model.embedding.patch_size,
        model_embedding_layer=model.embedding,
    ).to(device)

    errors = 0
    for idx, img in enumerate(images):
        input_embedding = model.embedding(model.patcher(img.unsqueeze(0)))
        original_pred = torch.argmax(model(img.unsqueeze(0))[0])
        decoded_image = decoder(input_embedding)
        decoded_pred = torch.argmax(model(decoded_image)[0])
        if original_pred != decoded_pred:
            print(
                f"Image: {idx}\tPrediction: {original_pred}\tDecoder prediction: {decoded_pred}"
            )
            errors += 1
    print(f"Total errors: {errors}/{len(images)} images")


if __name__ == "__main__":
    main()
