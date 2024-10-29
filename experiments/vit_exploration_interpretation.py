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
    original_image, _ = load_raw_image(*img_filename)
    model.eval()
    json_stats = {}
    json_stats["original_image_pred"] = torch.argmax(
        model(original_image.to(device).unsqueeze(0))[0]
    ).item()  # prediction from original image

    pred = torch.argmax(model.classifier(output_embedding[:, 0])).to(
        device
    )  # prediction from modified embedding at a certain iteration
    pred_capped = "exploration-capping"  # just for return values purposes

    input_embedding = input_embedding.detach()

    if capping:
        min_cap = load_object(os.path.join(capping, "min_distribution.pkl")).to(device)
        max_cap = load_object(os.path.join(capping, "max_distribution.pkl")).to(device)
        # cap input embeddings to bring them back to what the decoder knows
        input_embedding[input_embedding < min_cap] = min_cap[input_embedding < min_cap]
        input_embedding[input_embedding > max_cap] = max_cap[input_embedding > max_cap]

        pred_capped = torch.argmax(
            model.classifier(model.encoder(input_embedding)[0][:, 0])
        ).to(
            device
        )  # prediction from modified embedding at a certain iteration, when exploring phase does not perform capping at each iteration

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
    modified_image_pred = model(modified_image.unsqueeze(0))[
        0
    ].squeeze()  # prediction from translating embeddings back to image, and processing that image
    json_stats["modified_image_pred"] = torch.argmax(modified_image_pred).item()
    json_stats["modified_image_pred_proba"] = torch.max(modified_image_pred).item()
    json_stats["modified_original_pred_proba"] = modified_image_pred[
        json_stats["original_image_pred"]
    ].item()  # this is to compare probabilities in case the prediction has changed
    modified_image_pred = torch.argmax(modified_image_pred)
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

    json_fname = os.path.join(
        img_out_dir,
        f"{iteration}-stats.json",
    )
    with open(json_fname, "w") as file:
        json.dump(json_stats, file)
    if capping:
        return pred_capped, modified_image_pred
    return pred, modified_image_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--cap-ex", action="store_true")

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    sub_dirs_exists = any(
        [os.path.isdir(os.path.join(args.img_dir, d)) for d in os.listdir(args.img_dir)]
    )
    if sub_dirs_exists:
        all_images, all_names, all_eq_class_patch = {}, {}, {}
        for subdir in os.listdir(args.img_dir):
            if os.path.isdir(os.path.join(args.img_dir, subdir)):
                im, n = load_raw_images(os.path.join(args.img_dir, subdir))
                all_images[subdir] = im.to(device)
                all_names[subdir] = n
                all_eq_class_patch[subdir] = json.load(
                    open(os.path.join(args.img_dir, subdir, "config.json"), "r")
                )
    else:
        images, _ = load_raw_images(args.img_dir)
        images = images.to(device)
        eq_class_patch = json.load(open(os.path.join(args.img_dir, "config.json"), "r"))

    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
    )
    deactivate_dropout_layers(model)
    model = model.to(device)
    for res_dir in os.listdir(args.pkl_dir):
        if res_dir.startswith("sime") and "vit" in res_dir:
            res_path = os.path.join(args.pkl_dir, res_dir)
            print(res_dir)
            if sub_dirs_exists:
                res_img_names = [
                    d
                    for d in os.listdir(res_path)
                    if os.path.isdir(os.path.join(res_path, d))
                ]
                img_name = [
                    k for k, v in all_names.items() if set(v) == set(res_img_names)
                ][0]
                images = all_images[img_name]
                eq_class_patch = all_eq_class_patch[img_name]

            patches_embeddings = []
            for img in images:
                input_patches = model.patcher(img.unsqueeze(0))
                patches_embeddings.append(
                    (input_patches, model.embedding(input_patches))
                )

            print("Interpretation phase")

            decoder = PatchDecoder(
                image_size=model.image_size,
                patch_size=model.embedding.patch_size,
                model_embedding_layer=model.embedding,
            ).to(device)

            for img_dir in os.listdir(res_path):
                if os.path.isdir(os.path.join(res_path, img_dir)):
                    predictions = {}
                    for filename in tqdm(
                        os.listdir(os.path.join(res_path, img_dir)), desc=img_dir
                    ):
                        if os.path.isfile(
                            os.path.join(res_path, img_dir, filename)
                        ) and filename.lower().endswith(".pkl"):
                            res = load_object(os.path.join(res_path, img_dir, filename))
                            pred, decoded_pred = interpret(
                                img_filename=(
                                    (args.img_dir, img_dir)
                                    if not sub_dirs_exists
                                    else (os.path.join(args.img_dir, img_name), img_dir)
                                ),
                                model=model.to(device),
                                decoder=decoder,
                                input_embedding=res["input_embedding"].to(device),
                                output_embedding=res["output_embedding"].to(device),
                                iteration=res["iteration"],
                                eq_class_patch_ids=eq_class_patch[img_dir],
                                img_out_dir=os.path.join(
                                    res_path, img_dir, "interpretation"
                                ),
                                capping=res_path if args.cap_ex else "",
                                device=device,
                            )
                            predictions[res["iteration"]] = (
                                pred == decoded_pred
                            ).item()
                    with open(
                        os.path.join(res_path, img_dir, "pred-stats.json"), "w"
                    ) as file:
                        json.dump(predictions, file)


if __name__ == "__main__":
    main()
