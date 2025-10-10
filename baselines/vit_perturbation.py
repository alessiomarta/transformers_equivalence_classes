import argparse
import torch
import numpy as np
import time
import os
import json
import sys
sys.path.append("../experiments")
from simec.logics import OutputOnlyModel, jacobian
from experiments_utils import (
    load_and_transform_raw_images,
    deactivate_dropout_layers,
    load_model,
    load_json,
    save_json,
    load_object,
)
from tqdm import tqdm, trange
import gc

# Objective: 
# 1. travelling long distances by maintaining invariance (SIMEC)
# 2. travelling short distances to find other equivalence classes (SIMEXP)

EPS= 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Directory containing data, config.json, and parameters.json. Automatically created with prepare_experiment.py",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="How many iteration to run the experiment.",
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
        default = 4,
        help="Batch size.",
    )

    arguments = parser.parse_args()
    if arguments.device is None:
        arguments.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        arguments.device = torch.device(arguments.device)

    return arguments


def orthogonal_tensor(v):

    random_dim = np.random.randint(low = 0, high=v.shape[-1])
    all_dims = list(range(v.shape[-1]))
    all_dims.remove(random_dim)
    new_v  = v.clone()
    new_v[..., random_dim] = -1 * torch.sum(new_v[..., all_dims], dim = -1)

    return new_v


def gaussian_orthogonal_noise(v, std):

    noise = torch.normal(mean=torch.zeros_like(v), std=std.diagonal() + EPS)
    ort_direction = torch.nn.functional.normalize(orthogonal_tensor(v))
    
    if ort_direction.dim() == 2:
        magnitude = ort_direction @ noise.T
    elif ort_direction.dim() == 3:
        magnitude = torch.bmm(ort_direction, noise.transpose(-1, -2)[...,[0]])
    else:
        raise ValueError(f"Invalid dimension for ort_direction: {ort_direction.dim()}")

    return magnitude * ort_direction


def random_explore(model, data_loader, min_embs, max_embs, device, n_iterations, explore_patches, objectives, step_size = 0.1, direction ="same"):

    model = OutputOnlyModel(model)
    multiplier = 1 if direction == "same" else -1
    results = []
    objectives = [[ob] for ob in objectives]
    step_size = multiplier * step_size

    for idx, input_batch in enumerate(data_loader):

        sigma = torch.abs(
            torch.cov(input_batch[0].view(-1, input_batch[0].shape[-1]).T)
        ).sqrt().to(device)
        embeddings = input_batch[0].clone().to(device)

        with torch.no_grad():
            original_labels = model(embeddings)[1]
            original_labels = original_labels.flatten()

        embeddings.requires_grad_(True)
        batch_size = embeddings.shape[0]
        embedding_size = embeddings.shape[-1]

        for t in trange(n_iterations, desc=f"Batch {idx}"):
            
            results.append([])
            gradients, _ = jacobian(embeddings, model, pred_id = objectives)

            with torch.no_grad():
                probits = model(embeddings)[0].detach().cpu().reshape(batch_size, gradients.shape[1])
                original_and_top_class_probs = [
                    {
                        "batch_idx": idx,
                        "original": probits[j, original_labels[j]].item(), 
                        "top": probits[j].flatten().max().item(),
                        "iteration": t
                    }
                for j in range(batch_size)]
                results[idx].extend(original_and_top_class_probs)

                for exp_patch in explore_patches:
                    embeddings[:, exp_patch, :] = embeddings[:, exp_patch, :] + step_size * gaussian_orthogonal_noise(gradients[:, 0, exp_patch, :].reshape(batch_size, -1, embedding_size), std = sigma)

                # Project back into the feasible region
                embeddings = torch.clamp(embeddings, min_embs.to(device), max_embs.to(device))
                del probits, gradients

        del embeddings
        torch.cuda.empty_cache()
        gc.collect()

    return results



def main():

    args = parse_args()
    if args.experiment_path.endswith("/"):
        args.experiment_path = args.experiment_path[:-1]
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    
    if args.out_dir is None:
        raise ValueError("No output path specified in out-dir argument.")
    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, os.path.basename(args.experiment_path) + "-" + str_time
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    n_iterations = params["iterations"]

    device = torch.device(args.device)
    images, names = load_and_transform_raw_images(args.experiment_path)
    #images = images.to(device)
    batch_size = args.batch_size

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

    # Patching and embedding layers
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

    min_embs = load_object(os.path.join(args.experiment_path, "min_distribution.pkl"))
    max_embs = load_object(os.path.join(args.experiment_path, "max_distribution.pkl"))

    explore_patches = [config[name]["explore"] for name in names]
    objectives = [config[name]["objective"] for name in names]
    
    for alg in ["same", "opposite"]:
        stats = random_explore(model, image_loader, min_embs, max_embs, device, n_iterations, explore_patches, objectives, step_size = 0.1, direction = alg)

        with open(os.path.join(res_path, f"perturbation_results_{alg}.json"), "w") as f:
            json.dump(stats, f)


if __name__ == "__main__":
    main()
