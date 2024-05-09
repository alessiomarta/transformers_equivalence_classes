"""
A module designed for exploration of feature importance in Vision Transformer (ViT) models.
This includes functionality for interpreting and visualizing how specific patches of an image
affect the model's predictions, and facilitates the exploration of input space to understand
model behavior better.
"""

import argparse
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import torch
from torchvision import transforms
from simec.logics import explore
from models.vit import PatchDecoder
from experiments_utils import (
    load_raw_images,
    load_raw_image,
    deactivate_dropout_layers,
    load_model,
    load_object,
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
    with torch.no_grad():
        pred = torch.argmax(model.classifier(output_embedding[:, 0].to(device)))
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
            decoded_image = decoder(input_embedding.to(device)).squeeze()
            decoded_image = -1 + 2 * (decoded_image - torch.min(decoded_image)) / (
                torch.max(decoded_image) - torch.min(decoded_image)
            )
            decoded_image[decoded_image < -1] = -1
            decoded_image[decoded_image > 1] = 1
            modified_image = original_image.clone().to(device)
            modified_image[:, mod_pixels[1], mod_pixels[0]] = decoded_image[
                mod_pixels[1], mod_pixels[0]
            ]
        else:
            modified_image = decoder(input_embedding.to(device)).squeeze().to(device)
            modified_image = -1 + 2 * (modified_image - torch.min(modified_image)) / (
                torch.max(modified_image) - torch.min(modified_image)
            )
            if len(modified_image.size()) == 2:
                modified_image = modified_image.unsqueeze(0)
        modified_image_pred = torch.argmax(model(modified_image.unsqueeze(0))[0])
        fname = os.path.join(
            img_out_dir, f"{iteration}-{pred}-{modified_image_pred}.png"
        )
        _, ax = plt.subplots()
        ax.imshow(
            modified_image.squeeze().cpu().numpy(),
            cmap="gray",
            # norm=Normalize(vmin=-1, vmax=1),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-type", type=str, choices=["same", "diff"], required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--keep-constant", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--save-each", type=int, default=100)
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

    str_time = time.strftime("%Y%m%d-%H%M%S")
    str_time = "20240509-122102"
    res_path = os.path.join(
        args.out_dir, "input-space-exploration", args.exp_name + "-" + str_time
    )

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    with open(os.path.join(res_path, "params.json"), "w") as file:
        json.dump(vars(args), file)

    for idx, img in enumerate(images):

        print("Image:", idx)

        input_patches = model.patcher(img.unsqueeze(0))
        input_embedding = model.embedding(input_patches)

        if False:

            print("\tExploration phase")

            explore(
                same_equivalence_class=args.exp_type == "same",
                input_embedding=input_embedding,
                model=model.encoder,
                threshold=args.threshold,
                n_iterations=args.iter,
                pred_id=args.keep_constant,
                eq_class_emb_ids=(
                    None
                    if eq_class_patch[names[idx]] == []
                    else eq_class_patch[names[idx]]
                ),
                device=device,
                out_dir=os.path.join(res_path, names[idx]),
                keep_timing=True,
                save_each=args.save_each,
            )

    print("\tInterpretation phase")

    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=model.embedding.patch_size,
        model_embedding_layer=model.embedding,
    ).to(device)

    with torch.no_grad():
        for img_dir in os.listdir(res_path):
            if os.path.isdir(os.path.join(res_path, img_dir)):
                for filename in os.listdir(os.path.join(res_path, img_dir)):
                    if os.path.isfile(
                        os.path.join(res_path, img_dir, filename)
                    ) and filename.lower().endswith(".pkl"):
                        res = load_object(os.path.join(res_path, img_dir, filename))
                        interpret(
                            img_filename=(args.img_dir, img_dir),
                            model=model,
                            decoder=decoder,
                            input_embedding=res["input_embedding"],
                            output_embedding=res["output_embedding"],
                            iteration=res["iteration"],
                            eq_class_patch_ids=eq_class_patch[img_dir],
                            img_out_dir=os.path.join(res_path, img_dir, "images"),
                            device=device,
                        )


if __name__ == "__main__":
    main()
