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
sys.path.append("./experiments_data/")
from experiments_utils import (
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default = 64,
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
    #model = model.to(device)
    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, os.path.basename(args.experiment_path) + "-" + str_time
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    input_patches = model.patcher(images)
    # input_patches.shape = (len(images), N_patches, N_channels*kernel_size**2)
    patches_embeddings = model.embedding(input_patches)
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
    if params["algo"] == "both":
        algorithms.append("simexp")
    elif params["algo"] == "simexp":
        algorithms[0] = "simexp"

    min_embs = load_object(os.path.join(args.experiment_path, "min_distribution.pkl"))
    max_embs = load_object(os.path.join(args.experiment_path, "max_distribution.pkl"))

    for algorithm in algorithms:
        print(f"\t{algorithm.upper()} exploration phase")
        for idx, input_embedding in enumerate(image_loader):
            # input_embedding[0].shape = (batch_size, N_patches+1, embedding_size)
            image_indices = list(range(batch_size*idx, batch_size*(idx+1)))
            file_names = [names[i] for i in image_indices]
            
            # Parameters
            eq_class_emb_ids = [config[name]["explore"] for name in file_names]
            if len(sum(eq_class_emb_ids, [])) == 0:
                eq_class_emb_ids = None
            pred_id = [config[name]["objective"] for name in file_names]
                        
            for r in range(params["repeat"]):
                print(
                    f"Image: {file_names}\tRepetition: {r+1}/{params['repeat']}"
                )
                try:
                    explore(
                        same_equivalence_class=(algorithm == "simec"),
                        input_embedding=input_embedding[0],
                        model=model,
                        threshold=params["threshold"],
                        delta_multiplier=params["delta_mult"],
                        n_iterations=params["iterations"],
                        pred_id=pred_id,
                        eq_class_emb_ids= eq_class_emb_ids,
                        device=device,
                        out_dir=[os.path.join(
                            res_path,
                            f"{algorithm}-{name.split('.')[0]}-{str(r+1)}"
                        ) for name in file_names],
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
