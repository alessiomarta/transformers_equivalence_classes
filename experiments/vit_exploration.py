"""
A module designed for exploration of feature importance in Vision Transformer (ViT) models.
This includes functionality for interpreting and visualizing how specific patches of an image
affect the model's predictions, and facilitates the exploration of input space to understand
model behavior better.
"""

import argparse
import os
import logging as log
import time
import torch
from experiments.experiments_utils import (
    load_and_transform_raw_images,
    deactivate_dropout_layers,
    load_model,
    load_json,
    load_object,
)
from simec.logics import explore, ExplorationException


# Configure the logger
log.basicConfig(
    filename="error_log.txt",
    filemode="a",
    level=log.ERROR,
    format="\n\n%(asctime)s - %(levelname)s - %(message)s",
)


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


def main():
    args = parse_args()
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    device = torch.device(args.device)
    images, names = load_and_transform_raw_images(args.experiment_path)
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

    input_patches = model.patcher(images)
    patches_embeddings = list(
        zip(input_patches, model.embedding(input_patches).unsqueeze(1))
    )

    algorithms = ["simec"]
    if params["algo"] == "both":
        algorithms.append("simexp")
    elif params["algo"] == "simexp":
        algorithms[0] = "simexp"

    min_embs = load_object(os.path.join(args.experiment_path, "min_distribution.pkl"))
    max_embs = load_object(os.path.join(args.experiment_path, "max_distribution.pkl"))

    for algorithm in algorithms:
        print(f"\t{algorithm.upper()} exploration phase")
        for idx, (input_patches, input_embedding) in enumerate(patches_embeddings):
            for r in range(params["repeat"]):
                print(
                    f"Image: {names[idx]}\t{idx+1}/{len(images)}\tRepetition: {r+1}/{params['repeat']}"
                )
                try:
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
                            res_path,
                            f"{algorithm}-{names[idx].split('.')[0]}-{str(r+1)}",
                        ),
                        save_each=params["save_each"],
                        capping=True,
                        min_embeddings=min_embs,
                        max_embeddings=max_embs,
                    )
                except Exception as e:
                    log.error(
                        "Unhandled exception during exploration:\nContext:\n"
                        "res_path: %s\nalgorithm: %s\nname: %s\nparams_repeat: %d\nError: %s",
                        res_path,
                        algorithm,
                        names[idx],
                        r + 1,
                        e,
                        exc_info=True,
                    )
                    raise ExplorationException from e


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if not isinstance(e, ExplorationException):
            log.error("An error occurred: %s", e, exc_info=True)
        raise
