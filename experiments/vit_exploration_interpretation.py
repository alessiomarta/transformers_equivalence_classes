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
from experiments.models.vit import PatchDecoder
from experiments.experiments_utils import (
    load_and_transform_raw_images,
    deactivate_dropout_layers,
    load_model,
    load_object,
    save_object,
    load_json,
    collect_pkl_res_files,
)
from experiments.models.const import CIFAR_MEAN, CIFAR_STD, MNIST_MEAN, MNIST_STD

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
    output_embedding: torch.Tensor,
    iteration: int,
    eq_class_patch_ids: list,
    device: torch.device,
    means: list,
    sds: list,
    img_out_dir: str = ".",
    capping: str = "",
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
    width, height = original_image.shape[1:]
    model.eval()
    json_stats = {}
    original_image_pred = model(original_image.to(device).unsqueeze(0))[0]
    json_stats["original_image_pred_proba"] = (
        original_image_pred  # prediction probabilities from original image
    ).cpu()
    json_stats["original_image_pred"] = torch.argmax(
        original_image_pred
    ).item()  # prediction from original image

    pred_proba = model.classifier(
        output_embedding[:, 0]
    )  # prediction prababilities from modified embedding at a certain iteration
    json_stats["embedding_pred_proba"] = pred_proba.cpu()
    json_stats["embedding_pred"] = torch.argmax(
        pred_proba
    ).item()  # prediction from modified embedding at a certain iteration

    input_embedding = input_embedding.detach()
    if capping:
        min_cap = load_object(os.path.join(capping, "min_distribution.pkl")).to(device)
        max_cap = load_object(os.path.join(capping, "max_distribution.pkl")).to(device)
        # cap input embeddings to bring them back to what the decoder knows
        input_embedding[input_embedding < min_cap] = min_cap[input_embedding < min_cap]
        input_embedding[input_embedding > max_cap] = max_cap[input_embedding > max_cap]

        pred_capped_proba = model.classifier(
            model.encoder(input_embedding)[0][:, 0]
        )  # prediction from modified embedding at a certain iteration, when exploring phase does not perform capping at each iteration
        json_stats["capped_embedding_pred_proba"] = pred_capped_proba.cpu()
        json_stats["capped_embedding_pred"] = torch.argmax(pred_capped_proba).item()
    else:
        json_stats["capped_embedding_pred_proba"] = (
            None  # this means that the capping has been performed at exploration time
        )
        json_stats["capped_embedding_pred"] = "exploration-capping"

    # select those patches to replace with modified ones
    patch_idx = []
    if eq_class_patch_ids:
        for p in eq_class_patch_ids:
            patch_idx.append(
                (
                    np.array(
                        np.unravel_index(  # TODO sistemare value error index -1 is out of bounds for array with size 256 per immagini dove non tutte le patch cambiano (non "all")
                            p,  # -1 ?
                            (
                                width // model.embedding.patch_size,
                                height // model.embedding.patch_size,
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
        decoded_image = decoder(input_embedding.to(device))

        # TODO da vedere effettivamente la distribuzione dei valori decodificati
        # TODO verificare comunque i bound in uscita dal decoder, ho visto dei 271...
        # TODO speriamo di risolvere le immagini in bianco

        # replace patches in original image with those in modified image
        modified_image = original_image.clone().to(device)
        modified_image[:, mod_pixels[1], mod_pixels[0]] = decoded_image[
            :, :, mod_pixels[1], mod_pixels[0]
        ]
        json_stats["modified_patches"] = modified_image[
            :, mod_pixels[1], mod_pixels[0]
        ].cpu()
    else:
        modified_image = decoder(input_embedding.to(device)).squeeze().to(device)
        if len(modified_image.size()) == 2:
            modified_image = modified_image.unsqueeze(0)
        json_stats["modified_patches"] = modified_image.cpu()
    modified_image_pred_proba = model(modified_image.unsqueeze(0))[
        0
    ].squeeze()  # prediction from translating embeddings back to image, and processing that image

    modified_image = denormalize(
        modified_image.to(device),
        std=torch.tensor(sds).to(device),
        mean=torch.tensor(means).to(device),
    )
    json_stats["modified_image_pred"] = torch.argmax(modified_image_pred_proba).item()
    json_stats["modified_image_pred_proba"] = modified_image_pred_proba.cpu()
    json_stats["modified_original_pred_proba"] = modified_image_pred_proba[
        json_stats["original_image_pred"]
    ].item()  # this is to compare probabilities in case the prediction has changed

    fname = os.path.join(
        img_out_dir,
        f"{iteration}-{json_stats['embedding_pred']}-{json_stats['capped_embedding_pred']}-{json_stats['modified_image_pred']}.png",
    )
    patches_fname = os.path.join(
        img_out_dir,
        f"patches-{iteration}-{json_stats['embedding_pred']}-{json_stats['capped_embedding_pred']}-{json_stats['modified_image_pred']}.png",
    )
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    save_image(
        modified_image,
        fname,
        format="png",
    )
    _, ax = plt.subplots()
    if modified_image.size(0) == 1:
        ax.imshow(
            modified_image.permute(1, 2, 0).squeeze().detach().cpu().numpy(),
            cmap="gray",
            # norm=Normalize(vmin=0, vmax=1),
        )
    else:
        ax.imshow(modified_image.permute(1, 2, 0).squeeze().detach().cpu().numpy())
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

    plt.savefig(patches_fname)
    plt.close()

    json_fname = os.path.join(
        img_out_dir,
        f"{iteration}-stats.pkl",
    )
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
        arguments.device = torch.device(arguments.device).type
    return arguments


def main():
    args = parse_args()
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    device = torch.device(args.device)
    images, names = load_and_transform_raw_images(args.experiment_path)
    means = CIFAR_MEAN if "cifar" in params["orig_data_dir"] else MNIST_MEAN
    sds = CIFAR_STD if "cifar" in params["orig_data_dir"] else MNIST_STD
    images = images.to(device)
    model_filename = [f for f in os.listdir(params["model_path"]) if f.endswith(".pt")]
    model, _ = load_model(
        model_path=os.path.join(params["model_path"], model_filename[0]),
        config_path=os.path.join(params["model_path"], "config.json"),
        device=device,
    )
    deactivate_dropout_layers(model)
    model = model.to(device)

    print("\tInterpretation phase")
    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=model.embedding.patch_size,
        model_embedding_layer=model.embedding,
    ).to(device)

    pkl_results_paths = collect_pkl_res_files(args.results_dir)
    img_extension = "." + names[0].split(".")[-1]  # to compose the image filename
    for pkl_path in tqdm(pkl_results_paths, desc=args.results_dir):
        res = load_object(pkl_path)
        img_name = pkl_path.split("-")[-2] + img_extension
        try:
            interpret(
                original_image=images[names.index(img_name)],
                model=model.to(device),
                decoder=decoder,
                input_embedding=res["input_embedding"].to(device),
                output_embedding=res["output_embedding"].to(device),
                iteration=res["iteration"],
                eq_class_patch_ids=config[img_name]["explore"],
                img_out_dir=os.path.join(os.path.dirname(pkl_path), "interpretation"),
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
