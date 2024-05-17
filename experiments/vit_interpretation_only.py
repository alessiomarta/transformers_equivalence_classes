"""
A module designed for exploration of feature importance in Vision Transformer (ViT) models.
This includes functionality for interpreting and visualizing how specific patches of an image
affect the model's predictions, and facilitates the exploration of input space to understand
model behavior better.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import torch
import seaborn as sns
from tqdm import tqdm
from models.vit import PatchDecoder
from experiments_utils import (
    load_raw_images,
    load_raw_image,
    deactivate_dropout_layers,
    load_model,
    load_object,
    save_object,
)


def interpret(
    img_filename: tuple,
    model: torch.nn.Module,
    decoder: PatchDecoder,
    input_embedding: torch.Tensor,
    output_embedding: torch.Tensor,
    iteration: int,
    eq_class_patch_ids: list,
    device: torch.device,
    min_cap: torch.Tensor,
    max_cap: torch.Tensor,
    img_out_dir: str = ".",
) -> None:
    """
    Interpret and visualize the effects of specific patches on the model's predictions.

    Args:
        model: The ViT model used for predictions.
        decoder: A decoder to reconstruct images from their embeddings.
        input_embedding: The embedding of the input image.
        output_embedding: The output embedding from the model after processing the input.
        iteration: The iteration number of the exploration process.
        eq_class_patch_ids: List of indices for patches to explore.
        img_out_dir: The directory to save the output images.

    Returns:
        None. Saves the interpreted image with marked patches to the specified directory.
    """
    original_image, _ = load_raw_image(*img_filename)
    model.eval()

    pred = torch.argmax(model.classifier(output_embedding[:, 0])).to(device)

    input_embedding = input_embedding.detach()
    # cap input embeddings to bring them back to what the decoder knows
    input_embedding[input_embedding < min_cap] = min_cap[input_embedding < min_cap]
    input_embedding[input_embedding > max_cap] = max_cap[input_embedding > max_cap]

    pred_capped = torch.argmax(
        model.classifier(model.encoder(input_embedding)[0][:, 0])
    ).to(device)

    # select those patches to replace with modified ones
    patch_idx = []
    if eq_class_patch_ids:
        for p in eq_class_patch_ids:
            patch_idx.append(
                (
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
            )
        mod_pixels = [[], []]
        for p in patch_idx:
            for i in range(model.embedding.patch_size):
                for j in range(model.embedding.patch_size):
                    mod_pixels[0].append(p[0] + i)
                    mod_pixels[1].append(p[1] + j)
        # embeddings->image
        decoded_image = decoder(input_embedding.to(device)).squeeze()

        # replace patches in original image with those in modified image
        modified_image = original_image.clone().to(device)
        modified_image[:, mod_pixels[1], mod_pixels[0]] = decoded_image[
            mod_pixels[1], mod_pixels[0]
        ]
    else:
        modified_image = decoder(input_embedding.to(device)).squeeze().to(device)
        if len(modified_image.size()) == 2:
            modified_image = modified_image.unsqueeze(0)
    modified_image_pred = torch.argmax(model(modified_image.unsqueeze(0))[0])
    fname = os.path.join(
        img_out_dir,
        f"{iteration}-{pred}-{pred_capped}-{modified_image_pred}.png",
    )
    _, ax = plt.subplots()
    ax.imshow(
        modified_image.squeeze().detach().cpu().numpy(),
        cmap="gray",
        norm=Normalize(vmin=-1, vmax=1),
    )
    for p in patch_idx:
        ax.add_patch(
            Rectangle(
                tuple(p + np.array([-0.5, -0.5])),
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
    return pred_capped, modified_image_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=str, required=True)
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

    images, names = load_raw_images(args.img_dir)
    images = images.to(device)

    eq_class_patch = json.load(open(os.path.join(args.img_dir, "config.json"), "r"))

    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
    )
    deactivate_dropout_layers(model)
    model = model.to(device)

    res_path = args.pkl_dir

    print("\tMeasuring input distribution...")
    patches_embeddings = []
    for idx, img in enumerate(images):
        input_patches = model.patcher(img.unsqueeze(0))
        patches_embeddings.append((input_patches, model.embedding(input_patches)))

    embeddings = torch.stack([el[1] for el in patches_embeddings], dim=-1)
    min_embeddings = torch.min(embeddings, dim=-1).values
    max_embeddings = torch.max(embeddings, dim=-1).values

    save_object(
        obj=min_embeddings.cpu(),
        filename=os.path.join(res_path, "min_distribution.pkl"),
    )
    save_object(
        obj=max_embeddings.cpu(),
        filename=os.path.join(res_path, "max_distribution.pkl"),
    )

    print("Interpretation phase")

    predictions = {"pred": [], "it": [], "name": []}

    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=model.embedding.patch_size,
        model_embedding_layer=model.embedding,
    ).to(device)

    for img_dir in os.listdir(res_path):
        if os.path.isdir(os.path.join(res_path, img_dir)):
            for filename in tqdm(
                os.listdir(os.path.join(res_path, img_dir)), desc=img_dir
            ):
                if os.path.isfile(
                    os.path.join(res_path, img_dir, filename)
                ) and filename.lower().endswith(".pkl"):
                    res = load_object(os.path.join(res_path, img_dir, filename))
                    min_embeddings = load_object(
                        os.path.join(res_path, "min_distribution.pkl")
                    )
                    max_embeddings = load_object(
                        os.path.join(res_path, "max_distribution.pkl")
                    )
                    pred, decoded_pred = interpret(
                        img_filename=(args.img_dir, img_dir),
                        model=model.to(device),
                        decoder=decoder,
                        input_embedding=res["input_embedding"].to(device),
                        output_embedding=res["output_embedding"].to(device),
                        iteration=res["iteration"],
                        eq_class_patch_ids=eq_class_patch[img_dir],
                        img_out_dir=os.path.join(res_path, img_dir, "images"),
                        min_cap=min_embeddings.to(device),
                        max_cap=max_embeddings.to(device),
                        device=device,
                    )
                predictions["pred"].append((pred == decoded_pred).item())
                predictions["it"].append(res["iteration"])
                predictions["name"].append(img_dir)
    pl = sns.scatterplot(data=predictions, x="it", y="pred")
    figure = pl.get_figure()
    # Customize the y-axis to show only 0 and 1
    plt.yticks([0, 1])
    plt.ylim(-0.3, 1.3)
    plt.grid(axis="x", which="major")
    figure.savefig(os.path.join(res_path, "pred.png"), dpi=400)
    plt.close()


if __name__ == "__main__":
    main()
