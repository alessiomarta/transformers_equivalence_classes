import argparse
import os
import re
import json
import pickle
from tqdm import tqdm
import pandas as pd
from numpy import savez_compressed, load, array
from numpy import max as npmax
from numpy.linalg import norm
import torch


def load_experiment_result(filename):
    if os.path.exists(filename + "/embeddings.npz") and os.path.exists(
        filename + "/metadata.json"
    ):
        with open(
            filename + "/metadata.json", "r"
        ) as f:  # Overwrites any existing file.
            metadata = json.load(f)
        data = load(filename + "/embeddings.npz")
    return metadata, data


def load_pickle(filename: str):
    with open(filename, "rb") as outp:
        obj = pickle.load(outp)
    return obj


def load_json(filename: str) -> dict:
    return json.load(
        open(
            filename,
            "r",
            encoding="utf-8",
        )
    )


def collect_npz_res_files(exploration_result_dir: str) -> list:
    npz_paths = []
    for root, _, files in os.walk(exploration_result_dir):
        for f in files:
            if f.lower().endswith(".npz"):
                npz_paths.append(root)
                break
    return npz_paths


def collect_pickle_res_files(img_exploration_interpretation_dir: str) -> list:
    pickle_paths = []
    for root, _, files in os.walk(img_exploration_interpretation_dir):
        for f in files:
            if f.lower().endswith(".pkl"):
                pickle_paths.append(f)
    return pickle_paths


def get_latest_experiment(base_dir, experiment_prefix):
    # List all directories under base_dir
    all_dirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Filter directories that match the given experiment prefix
    matching_dirs = [d for d in all_dirs if d.startswith(experiment_prefix + "-")]

    if not matching_dirs:
        return None

    # Extract timestamps and sort by them
    def extract_timestamp(d):
        match = re.search(r"(\d{8}-\d{6})$", d)
        return match.group(1) if match else ""

    # Sort directories by timestamp
    matching_dirs.sort(key=extract_timestamp, reverse=True)

    # Return the most recent one
    return os.path.join(base_dir, matching_dirs[0])


def extract_result_info(file):
    metadata, tensors = load_experiment_result(file)
    return {
        "time": metadata.get("time", None),
        "iteration": metadata["iteration"],
        "distance": tensors["distance"],
        "delta": metadata["delta"],
        "input_embedding": tensors["input_embedding"],
    }


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
        "--out-name",
        type=str,
        required=True,
        help="Name to give to the final output file.",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    args = parse_args()
    res_dir = args.results_dir  # "../res"
    exp_metadata_dir = args.experiment_path  # "../experiments/experiments_data/"
    # --experiment-path ../experiments/experiments_data/ --results-dir ../res

    # collecting configurations and parameters
    pkl_paths = []
    for root, _, files in os.walk(exp_metadata_dir):
        pkl_paths.extend(root for f in files if "config.json" in f)
    results = []
    for config_file in tqdm(pkl_paths, desc="Processing experiment results"):
        c = load_json(os.path.join(config_file, "config.json"))
        p = load_json(os.path.join(config_file, "parameters.json"))
        if "cifar" in p["orig_data_dir"]:
            dataset = "cifar"
        elif "mnist" in p["orig_data_dir"]:
            dataset = "mnist"
        elif "hate-speech" in p["orig_data_dir"]:
            dataset = "hatespeech"
        elif "wino_bias" in p["orig_data_dir"]:
            dataset = "winobias"
        else:
            raise Exception("No dataset found")
        exp_res_dir = get_latest_experiment(
            res_dir, os.path.basename(os.path.normpath(config_file))
        )
        if dataset not in ["mnist"]:
            continue  # TODO remove when everything is ready
        
        result_files = collect_npz_res_files(exploration_result_dir=exp_res_dir)
        for img, img_c in tqdm(
            c.items(),
            desc=f"Processing results in {os.path.basename(config_file)}",
            leave=False,
        ):
            algorithms = ["simec", "simexp"] if p["algo"] == ["both"] else p["algo"]
            repetitions = list(range(1, p["repeat"] + 1))
            for algorithm in algorithms:
                for repetition in repetitions:
                    img_result_files = []
                    exp_file = []
                    for f in result_files:
                        if (
                            f"-{img.split('.')[0]}-" in f
                            and f"-{str(p['delta_mult']).replace('.', 'p')}-" in f
                            and f"{dataset}-" in f
                            and f"-{p['patches']}-" in f
                            and f"{algorithm}-" in f
                            and f.split("-")[-1] == str(repetition)
                            and "test" not in f
                            and f"-{p['inputs']}-" in f
                        ):
                            exp_file.append(f)
                    for f in exp_file:
                        result_info = extract_result_info(f)
                        interpretation_pickles = collect_pickle_res_files(
                            os.path.join(f, "interpretation")
                        )
                        for iteration in result_info["iteration"]:
                            it = (
                                iteration // p["save_each"]
                                if iteration % p["save_each"] == 0
                                else (iteration + 1) // p["save_each"]
                            )
                            input_emb = result_info["input_embedding"][it]
                            input_emb_norm = norm(input_emb)
                            input_emb_max = npmax(input_emb)
                            interpretation_pickle = load_pickle(
                                next(
                                    os.path.join(f, "interpretation", p)
                                    for p in interpretation_pickles
                                    if p.split("-")[0] == str(iteration)
                                )
                            )
                            if interpretation_pickle["modified_patches"].dim()>1:
                                interpretation_pickle["modified_patches"] = interpretation_pickle["modified_patches"].flatten() #this should be fixed with later intepretation results
                            results.append(
                                tuple(
                                    [
                                        dataset,  # dataset
                                        p["delta_mult"],  # delta multiplier
                                        p["patches"],  # patch option
                                        img.split(".")[0],  # input name
                                        img_c["explore"],  # explored patches
                                        img_c[
                                            "attrib"
                                        ],  # attribution of explored patches
                                        len(
                                            img_c["attrib"]
                                        ),  # number of explored patches
                                        p[
                                            "save_each"
                                        ],  # results are saved at each save_each iterations
                                        algorithm,  # simec or simexp
                                        iteration,  # iteration
                                        result_info["distance"][it],  # distance
                                        result_info["delta"][it],  # delta
                                        input_emb,  # input_embedding
                                        input_emb_norm, # input_embedding norm
                                        input_emb_max, # input_embedding max
                                        result_info.get("time", None)[it],  # time
                                        repetition,
                                        interpretation_pickle[
                                            "original_image_pred_proba"
                                        ]
                                        .detach()
                                        .cpu()
                                        .squeeze(0)
                                        .numpy(),
                                        interpretation_pickle["original_image_pred"],
                                        interpretation_pickle["embedding_pred_proba"]
                                        .detach()
                                        .cpu()
                                        .squeeze(0)
                                        .numpy(),
                                        interpretation_pickle["embedding_pred"],
                                        interpretation_pickle["modified_image_pred"],
                                        interpretation_pickle[
                                            "modified_image_pred_proba"
                                        ]
                                        .detach()
                                        .cpu()
                                        .squeeze(0)
                                        .numpy(),
                                        interpretation_pickle[
                                            "modified_patches"
                                        ]
                                        .detach()
                                        .cpu()
                                        .squeeze(0)
                                        .numpy(),
                                    ]
                                )
                            )
                    
    results = pd.DataFrame(
        results,
        columns=[
            "dataset",
            "delta_multiplier",
            "patch_option",
            "input_name",
            "explored_patches",
            "patch_attribution",
            "number_explored_patches",
            "save_each",
            "algorithm",
            "iteration",
            "distance",
            "delta",
            "input_embedding",
            "input_embedding_norm",
            "input_embedding_max",
            "time",
            "repetition",
            "original_image_pred_proba",
            "original_image_pred",
            "embedding_pred_proba",
            "embedding_pred",
            "modified_image_pred",
            "modified_image_pred_proba",
            "modified_image",
        ],
    )

    print("Saving embeddings...")
    savez_compressed(
        os.path.join(res_dir, f"{args.out_name.lower()}_embeddings.npz"),
        distance=results["distance"].values,
        original_image_pred_proba=results["original_image_pred_proba"].values,
        embedding_pred_proba=results["embedding_pred_proba"].values,
        modified_image_pred_proba=results["modified_image_pred_proba"].values,
        modified_image=results["modified_image"].values,
    )
    non_array_columns = [
        col
        for col in results.columns
        if col
        not in [
            "distance",
            "original_image_pred_proba",
            "input_embedding",
            "embedding_pred_proba",
            "modified_image_pred_proba",
            "modified_image",
        ]
    ]
    print("Saving parquet...")
    results[non_array_columns].to_parquet(
        os.path.join(res_dir, f"{args.out_name.lower()}_results.parquet"), index=False
    )
