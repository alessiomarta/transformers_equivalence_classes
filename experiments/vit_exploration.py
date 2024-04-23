import argparse
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import torch
from simec.logics import explore
from models.vit import PatchDecoder
from utils import load_raw_images, deactivate_dropout_layers, load_model, load_object


def interpret(
    model,
    decoder,
    input_embedding,
    output_embedding,
    iteration,
    eq_class_patch_ids,
    img_out_dir=".",
):
    pred = torch.argmax(model.classifier(output_embedding[:, 0]))
    norm = Normalize(vmin=-1, vmax=1)
    fname = os.path.join(img_out_dir, f"{iteration}-{pred}.png")
    image = decoder(input_embedding)
    _, ax = plt.subplots()
    ax.imshow(image.squeeze().cpu().numpy(), cmap="gray", norm=norm)
    for p in eq_class_patch_ids:
        image_index = (
            np.array(
                np.unravel_index(
                    p - 1,
                    (
                        28 // model.embedding.patch_size,
                        28 // model.embedding.patch_size,
                    ),
                )
            )
            * model.embedding.patch_size
        )[::-1]
        ax.add_patch(
            Rectangle(
                tuple(image_index + np.array([-0.5, -0.5])),
                model.embedding.patch_size,
                model.embedding.patch_size,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    plt.savefig(fname)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-type", type=str, choices=["same", "diff"], required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--keep-constant", default=0)
    parser.add_argument("--delta", type=float, default=9e-1)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    # MNIST data
    images, names = load_raw_images(args.img_dir)
    images = images.to(device)

    # Select patches to explore
    eq_class_patch = json.load(open(os.path.join(args.img_dir, "config.json"), "r"))

    # load modified ViT model and deactivate it dropout layers
    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
    )
    deactivate_dropout_layers(model)

    # for naming results directories
    str_time = time.strftime("%Y%m%d-%H%M%S")
    str_time = "20240423-154821"
    res_path = os.path.join(
        args.out_dir, "input-space-exploration", args.exp_name + "-" + str_time
    )

    for idx, img in enumerate(images):

        # Clone and require gradient of the embedded input and prepare for the
        # first iteration
        input_patches = model.patcher(img.unsqueeze(0))
        input_embedding = model.embedding(input_patches)

        # input exploration
        explore(
            same_equivalence_class=args.exp_type == "same",
            input_embedding=input_embedding,
            model=model.encoder,
            delta=args.delta,
            threshold=args.threshold,
            n_iterations=args.iter,
            pred_id=args.keep_constant,
            eq_class_emb_ids=(
                None if eq_class_patch[names[idx]] == [] else eq_class_patch[names[idx]]
            ),
            device=device,
            out_dir=os.path.join(res_path, names[idx]),
        )

    # define decoder to plot images from embeddings
    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=model.embedding.patch_size,
        model_embedding_layer=model.embedding,
    )

    with torch.no_grad():
        for img_dir in os.listdir(res_path):
            if os.path.isdir(os.path.join(res_path, img_dir)):
                for filename in os.listdir(os.path.join(res_path, img_dir)):
                    if os.path.isfile(
                        os.path.join(res_path, img_dir, filename)
                    ) and filename.lower().endswith(".pkl"):
                        res = load_object(os.path.join(res_path, img_dir, filename))
                        interpret(
                            model=model,
                            decoder=decoder,
                            input_embedding=res["input_embedding"],
                            output_embedding=res["output_embedding"],
                            iteration=res["iteration"],
                            eq_class_patch_ids=eq_class_patch[img_dir],
                            img_out_dir=os.path.join(res_path, img_dir, "images"),
                        )


if __name__ == "__main__":
    main()
