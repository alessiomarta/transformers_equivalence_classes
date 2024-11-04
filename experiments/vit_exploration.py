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
import torch
from simec.logics import explore
from experiments_utils import (
    load_raw_images,
    deactivate_dropout_layers,
    load_model,
    save_object,
)


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
    parser.add_argument("--cap-ex", action="store_true")

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

    patches_embeddings = []
    for idx, img in enumerate(images):
        input_patches = model.patcher(img.unsqueeze(0))
        patches_embeddings.append((input_patches, model.embedding(input_patches)))

    print("\tMeasuring and saving input distribution for capping...")
    embeddings = torch.stack([el[1] for el in patches_embeddings], dim=-1)
    min_embeddings = torch.min(
        embeddings, dim=-1
    ).values  # TODO - norma infinito della matrice di embedding
    max_embeddings = torch.max(
        embeddings, dim=-1
    ).values  # TODO + norma infinito della matrice di embedding

    save_object(
        obj=min_embeddings.cpu(),
        filename=os.path.join(res_path, "min_distribution.pkl"),
    )
    save_object(
        obj=max_embeddings.cpu(),
        filename=os.path.join(res_path, "max_distribution.pkl"),
    )

    print("\tExploration phase")
    for idx, img in enumerate(images):
        print("Image:", idx)
        input_patches, input_embedding = patches_embeddings[idx]
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
            save_each=args.save_each,
            capping=res_path if args.cap_ex else "",
        )


if __name__ == "__main__":
    main()
