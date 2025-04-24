"""
A module designed for exploration of feature importance in Vision Transformer (ViT) models.
This includes functionality for interpreting and visualizing how specific patches of an image
affect the model's predictions, and facilitates the exploration of input space to understand
model behavior better.

Example usage:
    To run this script, use the following command:
    
    python vit_exploration_interpretation.py --experiment-path /path/to/experiment --results-dir /path/to/results --device cuda
"""

import argparse
import os
import logging as log
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import sys
sys.path.append("./")
from models.vit import PatchDecoder
from experiments_utils import (
    load_and_transform_raw_images,
    deactivate_dropout_layers,
    load_model,
    save_object,
    load_json,
    load_metadata_tensors,
)
from models.const import *

matplotlib.use("Agg")


class InterpretationException(Exception):
    pass


# Configure the logger
log.basicConfig(
    filename="error_log.txt",
    filemode="a",
    level=log.ERROR,
    format="\n\n%(asctime)s - %(levelname)s - %(message)s",
)


def denormalize(img, mean, std):
    """Undo normalization to convert back to [0,1](?) range"""
    mean = mean.view(-1, 1, 1)  # Reshape for broadcasting
    std = std.view(-1, 1, 1)
    img = img * std + mean  # Reverse normalization
    return img


def interpret(
    original_image: torch.Tensor,
    model: torch.nn.Module,
    decoder: PatchDecoder,
    input_embedding: torch.Tensor,
    iteration: int,
    eq_class_patch_ids: list,
    device: torch.device,
    means: list,
    sds: list,
    img_out_dir: str = ".",
    capping: str = "",
) -> None:
    """
    Interpret and visualize the effect of selected patch embeddings on the model's prediction.

    Args:
        original_image: The original input image tensor (C, H, W).
        model: The Vision Transformer (ViT) model.
        decoder: A decoder that maps embeddings back to image space.
        input_embedding: The latent embedding representation of the input.
        iteration: Current iteration count (for naming output).
        eq_class_patch_ids: List of patch indices to modify (excluding [CLS]).
        device: Device to perform computations on.
        means, sds: For denormalization.
        img_out_dir: Directory where results are saved.
        capping: Not used directly here (placeholder).
    """
    input_embedding = input_embedding.to(device, original_image.dtype)
    original_image = original_image.to(device)
    input_embedding = input_embedding.detach()

    width, height = original_image.shape[1:]
    patch_size = model.embedding.patch_size

    model.eval()
    json_stats = {}

    # -- Original prediction --
    with torch.no_grad():
        orig_pred_logits = model(original_image.unsqueeze(0))[0]
        json_stats["original_image_pred_proba"] = orig_pred_logits.cpu()
        json_stats["original_image_pred"] = torch.argmax(orig_pred_logits).item()

        # -- Prediction from embedding --
        output_embedding, _ = model.encoder(input_embedding.unsqueeze(0))
        emb_pred_logits = model.classifier(output_embedding[:, 0])
        json_stats["embedding_pred_proba"] = emb_pred_logits.cpu()
        json_stats["embedding_pred"] = torch.argmax(emb_pred_logits).item()

    # -- Remove CLS token index (assumed to be at index 0) --
    num_total_patches = (width // patch_size) * (height // patch_size)
    eq_class_patch_ids = [p for p in eq_class_patch_ids if 1 <= p <= num_total_patches]

    # -- Decode modified image and apply selected patches --
    decoded_image = decoder(input_embedding)
    modified_image = original_image.clone()

    if eq_class_patch_ids:
        mod_pixels = [[], []]  # [x coords], [y coords]

        for p in eq_class_patch_ids:
            row, col = np.unravel_index(p - 1, (height // patch_size, width // patch_size))
            x = col * patch_size
            y = row * patch_size
            for i in range(patch_size):
                for j in range(patch_size):
                    mod_pixels[0].append(x + i)
                    mod_pixels[1].append(y + j)

        modified_image[:, mod_pixels[1], mod_pixels[0]] = decoded_image[
            :, :, mod_pixels[1], mod_pixels[0]
        ]

        json_stats["modified_patches"] = modified_image[:, mod_pixels[1], mod_pixels[0]].cpu()
    else:
        modified_image = decoded_image.squeeze()
        if modified_image.dim() == 2:
            modified_image = modified_image.unsqueeze(0)
        json_stats["modified_patches"] = modified_image.cpu()

    # -- Predict again using reconstructed image --
    with torch.no_grad():
        mod_logits = model(modified_image.unsqueeze(0))[0].squeeze()
        json_stats["modified_image_pred"] = torch.argmax(mod_logits).item()
        json_stats["modified_image_pred_proba"] = mod_logits.cpu()
        json_stats["modified_original_pred_proba"] = mod_logits[
            json_stats["original_image_pred"]
        ].item()

    # -- Save visualization --
    if normalize:
        modified_image = denormalize(
            modified_image,
            std=torch.tensor(sds).to(device),
            mean=torch.tensor(means).to(device),
        )

    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    img_fname = os.path.join(
        img_out_dir,
        f"{iteration}-{json_stats['original_image_pred']}-{json_stats['embedding_pred']}-{json_stats['modified_image_pred']}.png",
    )
    patches_fname = os.path.join(
        img_out_dir,
        f"patches-{iteration}-{json_stats['original_image_pred']}-{json_stats['embedding_pred']}-{json_stats['modified_image_pred']}.png",
    )

    save_image(modified_image, img_fname, format="png")

    # -- Plot patch outlines --
    _, ax = plt.subplots()
    img_np = modified_image.permute(1, 2, 0).detach().cpu().numpy()
    if modified_image.size(0) == 1:
        ax.imshow(img_np.squeeze(), cmap="gray")
    else:
        ax.imshow(img_np)

    for p in eq_class_patch_ids:
        row, col = np.unravel_index(p - 1, (height // patch_size, width // patch_size))
        x, y = col * patch_size, row * patch_size
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                patch_size,
                patch_size,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )

    plt.savefig(patches_fname)
    plt.close()

    # -- Save stats --
    json_fname = os.path.join(img_out_dir, f"{iteration}-stats.pkl")
    save_object(json_stats, json_fname)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Directory containing input data, config.json, and parameters.json. Automatically created with prepare_experiment.py",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory where the exploration output from this experiment is stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Where to run this experiment",
    )
    parser.add_argument("--cap-ex", default=True)

    arguments = parser.parse_args()
    if arguments.device is None:
        arguments.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ).type
    else:
        arguments.device = torch.device(arguments.device)
    return arguments
#--experiment-path experiments_data_test/mnist-test-1p0-16-all --results-dir ../res/mnist-test-1p0-16-all-20250423-070701 --device cpu


def main():
    args = parse_args()
    if args.experiment_path.endswith("/"):
        args.experiment_path = args.experiment_path[:-1]
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    device = torch.device(args.device)

    images, names = load_and_transform_raw_images(args.experiment_path)
    if normalize:
        means = CIFAR_MEAN if "cifar" in params["orig_data_dir"] else MNIST_MEAN
        sds = CIFAR_STD if "cifar" in params["orig_data_dir"] else MNIST_STD
    else:
        means = [0.0]
        sds = [1.0] 
        
    images = images.to(device)
    models_filename = [f for f in os.listdir(params["model_path"]) if f.endswith(".pt")]
    model_filename = models_filename[0]
    for f in models_filename:
        if "final.pt" in f:
            model_filename = f
            break
    
    model, _ = load_model(
        model_path=os.path.join(params["model_path"], model_filename),
        config_path=os.path.join(params["model_path"], "config.json"),
        device=torch.device("cpu"),
    )
    deactivate_dropout_layers(model)
    model = model.to(device)

    print("\tInterpretation phase")
    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=model.embedding.patch_size,
        model_embedding_layer=model.embedding,
    ).to(device)
    
    img_extension = "." + names[0].split(".")[-1]  # to compose the image filename
    for folder in os.listdir(args.results_dir):
        experiment_dir = os.path.join(args.results_dir, folder)
        # load dei metadati e dei tensori in npz
        experiment_data = load_metadata_tensors(experiment_dir)
        for iteration in tqdm(range(len(experiment_data["iteration"])), desc=experiment_dir):
            img_name = folder.split("-")[-2] + img_extension
            try:
                interpret(
                    original_image=images[names.index(img_name)],
                    model=model.to(device),
                    decoder=decoder,
                    input_embedding=experiment_data["input_embedding"][iteration].to(device),
                    iteration=experiment_data["iteration"][iteration],
                    eq_class_patch_ids=config[img_name]["explore"],
                    img_out_dir=os.path.join(experiment_dir, "interpretation"),
                    capping=args.results_dir if not args.cap_ex else "",
                    device=device,
                    means=means,
                    sds=sds,
                )
            except Exception as e:
                log.error(
                    "Unhandled exception during intepretation:\nContext:\n"
                    "experiment: %s\nmodel: %s\nimage: %s\nparams: %d\nError: %s",
                    args.experiment_path,
                    params["model_path"],
                    img_name,
                    params,
                    e,
                    exc_info=True,
                )
                raise InterpretationException from e


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if not isinstance(e, InterpretationException):
            log.error("An error occurred: %s", e, exc_info=True)
        raise
