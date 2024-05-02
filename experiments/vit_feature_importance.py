"""
A module for analyzing feature importance in Vision Transformer (ViT) models. This includes 
functionality for visual interpretation of feature contributions and saving these visualizations.
"""

import argparse
import os
import time
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from simec.logics import pullback_eigenvalues
from experiments_utils import (
    load_raw_images,
    deactivate_dropout_layers,
    load_model,
    load_object,
)


def interpret(
    model: torch.nn.Module,
    output_embedding: torch.Tensor,
    starting_img: torch.Tensor,
    eigenvalues: torch.Tensor,
    img_out_dir: str = ".",
) -> None:
    """
    Visualize and save the interpretation of an image's features based on the eigenvalues.

    Args:
        model: The ViT model used for classification.
        output_embedding: The embedding produced by the model.
        starting_img: The original image to be analyzed.
        eigenvalues: The eigenvalues indicating feature importance.
        img_out_dir: The directory where the output image will be saved.
    """
    pred = torch.argmax(model.classifier(output_embedding[:, 0]))
    max_eigenvalues = [
        torch.tensor(v) for v in torch.max(eigenvalues, dim=1).values.tolist()
    ]

    max_eigenvalues = (
        torch.stack(max_eigenvalues[1:])
        .reshape(14, 14)
        .repeat_interleave(2, dim=0)
        .repeat_interleave(2, dim=1)
    )
    fname = os.path.join(img_out_dir, f"{pred}.png")
    fig = plt.figure(figsize=(8, 4))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    ax1, ax2, cax = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    )

    ax1.imshow(starting_img.squeeze().cpu().detach().numpy(), cmap="gray")
    ax1.axis("off")
    feature_importance = ax2.imshow(max_eigenvalues.squeeze().cpu().detach().numpy())
    ax2.axis("off")
    cbar = plt.colorbar(feature_importance, cax=cax)
    cbar.set_label("Eigenvalues")
    plt.subplots_adjust(wspace=0, hspace=0)
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    plt.savefig(fname)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
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

    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
    )
    deactivate_dropout_layers(model)

    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, "feature-importance", args.exp_name + "-" + str_time
    )

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    with open(os.path.join(res_path, "params.json"), "w") as file:
        json.dump(args, file)

    for idx, img in enumerate(images):
        input_patches = model.patcher(img.unsqueeze(0))
        input_embedding = model.embedding(input_patches)
        pullback_eigenvalues(
            model=model.encoder,
            input_embedding=input_embedding,
            pred_id=0,
            device=device,
            out_dir=os.path.join(res_path, names[idx]),
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
                            output_embedding=res["output_embedding"],
                            starting_img=images[
                                [idx for idx, n in enumerate(names) if n == img_dir][0]
                            ],
                            eigenvalues=res["eigenvalues"],
                            img_out_dir=os.path.join(res_path, img_dir, "images"),
                        )


if __name__ == "__main__":
    main()
