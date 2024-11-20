"""
A module designed for exploration of feature importance in Vision Transformer (ViT) models.
This includes functionality for interpreting and visualizing how specific patches of an image
affect the model's predictions, and facilitates the exploration of input space to understand
model behavior better.
"""

import argparse
import os
import time
import torch
from experiments_utils import (
    load_raw_images,
    deactivate_dropout_layers,
    load_model,
    save_object,
    load_json,
    save_json,
)
from simec.logics import explore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Directory containing data, config.json, and parameters.json. Automatically created with prepare_experiment.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory where to store the exploration output from this experiment.",
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


if __name__ == "__main__":
    args = parse_args()
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    device = torch.device(args.device)
    images, names = load_raw_images(args.experiment_path)
    images = images.to(device)
    model_filename = [f for f in os.listdir(params["model_path"]) if f.endswith(".pt")]
    model, _ = load_model(
        model_path=os.path.join(params["model_path"], model_filename[0]),
        config_path=os.path.join(params["model_path"], "config.json"),
        device=device,
    )
    deactivate_dropout_layers(model)
    model = model.to(device)
    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, os.path.basename(args.experiment_path) + "-" + str_time
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    print("\tMeasuring and saving input distribution for capping...")
    patches_embeddings = []
    for idx, img in enumerate(images):
        input_patches = model.patcher(img.unsqueeze(0))
        patches_embeddings.append((input_patches, model.embedding(input_patches)))
    embeddings = torch.stack([el[1] for el in patches_embeddings], dim=-1)
    min_embeddings = torch.min(torch.abs(embeddings), dim=-1).values
    max_embeddings = torch.max(torch.abs(embeddings), dim=-1).values
    save_object(
        obj=min_embeddings.cpu(),
        filename=os.path.join(res_path, "min_distribution.pkl"),
    )
    save_object(
        obj=max_embeddings.cpu(),
        filename=os.path.join(res_path, "max_distribution.pkl"),
    )

    algorithms = ["simec"]
    if params["algo"] == "both":
        algorithms.append("simexp")
    elif params["algo"] == "simexp":
        algorithms[0] = "simexp"
    for algorithm in algorithms:
        print(f"\t{algorithm.upper()} exploration phase")
        for idx in range(len(images)):
            input_patches, input_embedding = patches_embeddings[idx]
            for r in range(params["repeat"]):
                print(
                    f"Image: {names[idx]}\t{idx+1}/{len(images)}\tRepetition: {r+1}/{params['repeat']}"
                )
                explore(
                    same_equivalence_class=algorithm == "simec",
                    input_embedding=input_embedding,
                    model=model.encoder,
                    threshold=params["threshold"],
                    delta_multiplier=params["delta_mult"],
                    n_iterations=params["iterations"],
                    pred_id=config[names[idx]]["objective"],
                    eq_class_emb_ids=(
                        None
                        if config[names[idx]]["explore"] == []
                        else config[names[idx]]["explore"]
                    ),
                    device=device,
                    out_dir=os.path.join(
                        res_path, f"{algorithm}-{names[idx].split('.')[0]}-{str(r)}"
                    ),
                    keep_timing=True,
                    save_each=params["save_each"],
                    capping=res_path if args.cap_ex else "",
                )
