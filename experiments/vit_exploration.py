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
import sys
from tqdm import tqdm
sys.path.append("./")

from experiments_utils import (
    load_and_transform_raw_images,
    deactivate_dropout_layers,
    load_model,
    load_json,
    save_json,
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
        "--continue-from",
        type=str,
        help="Directory containing previous iteration for the selected experiments. If this argument is used, this experiment will continue from these previous iterations. res_path will be ignored, as further iterations will be saved in the same experiment path.",
    )
    parser.add_argument(
        "--extra-iterations",
        type=int,
        help="How many extra iteration to continue the experiment.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Directory where to store the exploration output from this experiment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Where to run this experiment",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default = 16,
        help="Batch size.",
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
    if args.continue_from is not None:
        if not os.path.exists(args.continue_from):
            raise FileNotFoundError(
                "The directory selected for continuing the experiment does not exist."
            )
        if os.path.basename(args.experiment_path) not in args.continue_from:
            raise ValueError(
                "The directory for continuing experiment must match the metadata experiment directory."
            )
        if args.extra_iterations is None:
            raise ValueError(
                "Specify how many iterations to perform for continuing the experiment"
            )
        n_iterations = params["iterations"] + args.extra_iterations
        start_iteration = params["iterations"]
        params["iterations"] = n_iterations
        save_json(os.path.join(args.experiment_path, "parameters.json"), params)

    else:
        if args.out_dir is None:
            raise ValueError("No output path specified in out-dir argument.")
        str_time = time.strftime("%Y%m%d-%H%M%S")
        res_path = os.path.join(
            args.out_dir, os.path.basename(args.experiment_path) + "-" + str_time
        )
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        n_iterations = params["iterations"]
        start_iteration = 0
    device = torch.device(args.device)
    images, names = load_and_transform_raw_images(args.experiment_path)
    #images = images.to(device)
    batch_size = args.batch_size

    model_filename = [f for f in os.listdir(params["model_path"]) if f.endswith(".pt")]
    model, _ = load_model(
        model_path=os.path.join(params["model_path"], model_filename[0]),
        config_path=os.path.join(params["model_path"], "config.json"),
        device=torch.device("cpu"),
    )
    deactivate_dropout_layers(model)
    model = model.to(device)

    input_patches = model.patcher(images).to(device)
    # input_patches.shape = (len(images), N_patches, N_channels*kernel_size**2)
    patches_embeddings = model.embedding(input_patches).cpu()
    # patches_embedding.shape = (len(images), N_patches+1, embedding_size)

    # Organize patch embeddings in a tensor dataset to load them
    patches_embeddings = torch.utils.data.TensorDataset(patches_embeddings)
    image_loader = torch.utils.data.DataLoader(
        patches_embeddings,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    algorithms = ["simec"]
    if params["algo"] == ["both"]:
        algorithms.append("simexp")
    elif params["algo"] == "simexp":
        algorithms[0] = "simexp"

    min_embs = load_object(os.path.join(args.experiment_path, "min_distribution.pkl"))
    max_embs = load_object(os.path.join(args.experiment_path, "max_distribution.pkl"))
    
    for algorithm in algorithms:
        pbar = tqdm(total=len(params["repeat"]) * len(image_loader), desc=f"{args.experiment_path}, {algorithm.upper()}")
        for idx, input_embedding in enumerate(image_loader):
            # input_embedding[0].shape = (batch_size, N_patches+1, embedding_size)
            image_indices = (batch_size*idx, batch_size*(idx+1))
            file_names = names[image_indices[0]:image_indices[1]]
            
            # Parameters
            eq_class_emb_ids = [config[name]["explore"] for name in file_names]
            pred_id = [config[name]["objective"] for name in file_names]
                        
            for r in range(params["repeat"]):
                try:
                    # load last iteration if continuing another experiment
                    if args.continue_from is not None:
                        start_embeddings = []
                        distance = []
                        start_iteration = []
                        for name in file_names:
                            exp_dir = os.path.join(
                                args.continue_from,
                                f"{algorithm}-{name.split('.')[0]}-{str(r+1)}",
                            )
                            list_dir = os.listdir(exp_dir)
                            pkl_files = [
                                f
                                for f in list_dir
                                if os.path.splitext(os.path.basename(f))[0].isnumeric()
                            ]
                            last_pkl = load_object(
                                os.path.join(
                                    exp_dir,
                                    sorted(pkl_files, key=lambda x: int(x.split(".")[0]))[
                                        -1
                                    ],
                                )
                            )
                            start_embeddings.append(last_pkl["input_embedding"])
                            distance.append(last_pkl["distance"])
                            start_iteration.append(last_pkl['iteration'])

                        start_embeddings = torch.stack(start_embeddings)
                        distance = torch.stack(distance)
                        start_iteration = min(start_iteration)

                    else:
                        exp_dir = [
                            os.path.join(
                                res_path,
                                f"{algorithm}-{name.split('.')[0]}-{str(r+1)}",
                            )
                        for name in file_names]
                        distance = None
                        start_embeddings = input_embedding[0]
                        start_iteration = 0

                    # Exploration
                    explore(
                        same_equivalence_class=(algorithm == "simec"),
                        input_embedding=start_embeddings,
                        input_model=model,
                        threshold=params["threshold"],
                        delta_multiplier=params["delta_mult"],
                        n_iterations=n_iterations,
                        pred_id=pred_id,
                        eq_class_emb_ids=eq_class_emb_ids,
                        device=device,
                        out_dir=exp_dir,
                        save_each=params["save_each"],
                        capping=True,
                        min_embeddings=min_embs,
                        max_embeddings=max_embs,
                        start_iteration=start_iteration,
                        distance=distance,
                    )
                    pbar.update(1)
                except Exception as e:
                    log.error(
                        "Unhandled exception during exploration:\nContext:\n"
                        "res_path: %s\nalgorithm: %s\nname: %s\nparams_repeat: %d\nError: %s",
                        exp_dir,
                        algorithm,
                        file_names,
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
