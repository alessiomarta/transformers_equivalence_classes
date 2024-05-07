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
from simec.logics import explore
from models.vit import PatchDecoder
from experiments_utils import (
    load_raw_images,
    deactivate_dropout_layers,
    load_model,
    load_object,
)


def interpret(
    model: torch.nn.Module,
    decoder: PatchDecoder,
    input_embedding: torch.Tensor,
    output_embedding: torch.Tensor,
    iteration: int,
    eq_class_patch_ids: list,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-type", type=str, choices=["same", "diff"], required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--keep-constant", type=int, default=0)
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

        print("\tExploration phase")

        explore(
            same_equivalence_class=args.exp_type == "same",
            input_embedding=input_embedding,
            model=model.encoder,
            threshold=args.threshold,
            n_iterations=args.iter,
            pred_id=args.keep_constant,
            eq_class_emb_ids=(
                None if eq_class_patch[names[idx]] == [] else eq_class_patch[names[idx]]
            ),
            device=device,
            out_dir=os.path.join(res_path, names[idx]),
            keep_timing=True,
            save_each=10,
        )

    print("\tInterpretation phase")

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
