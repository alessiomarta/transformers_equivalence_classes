import os
import argparse
import json
from tqdm import tqdm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pre_probas = []
    cap_probas = []
    post_probas = []
    pre_eq_class_wrds = []
    cap_eq_class_wrds = []
    for res in os.listdir(args.json_dir):
        if os.path.isdir(os.path.join(args.json_dir, res)):
            files = [
                filename
                for filename in os.listdir(
                    os.path.join(args.json_dir, res, "interpretation")
                )
                if os.path.isfile(
                    os.path.join(args.json_dir, res, "interpretation", filename)
                )
                and filename.lower().endswith("-stats.json")
            ]
            for j_file in tqdm(files, desc=res):
                stats = json.load(
                    open(
                        os.path.join(args.json_dir, res, "interpretation", j_file), "r"
                    )
                )
                pre_probas.append(
                    sorted([el[1] for el in stats["pre-cap-probas"]], reverse=True)
                )
                cap_probas.append(
                    sorted([el[1] for el in stats["cap-probas"]], reverse=True)
                )
                post_probas.append(
                    sorted([el[1] for el in stats["mod-probas"]], reverse=True)
                )
                pre_eq_class_wrds_keys = [
                    k for k in stats.keys() if "pre-cap-probas-" in k
                ]
                for k in pre_eq_class_wrds_keys:
                    pre_eq_class_wrds.append(
                        sorted([el[1] for el in stats[k]], reverse=True)
                    )
                cap_eq_class_wrds_keys = [
                    k
                    for k in stats.keys()
                    if "cap-probas-" in k and "pre-cap-probas-" not in k
                ]
                for k in cap_eq_class_wrds_keys:
                    cap_eq_class_wrds.append(
                        sorted([el[1] for el in stats[k]], reverse=True)
                    )

    norm_pre_probas = (np.array(pre_probas) - np.min(np.array(pre_probas))) / (
        np.max(np.array(pre_probas)) - np.min(np.array(pre_probas))
    )
    norm_cap_probas = (np.array(cap_probas) - np.min(np.array(cap_probas))) / (
        np.max(np.array(cap_probas)) - np.min(np.array(cap_probas))
    )
    norm_post_probas = (np.array(post_probas) - np.min(np.array(post_probas))) / (
        np.max(np.array(post_probas)) - np.min(np.array(post_probas))
    )

    norm_pre_eq_class_wrds = (
        np.array(pre_eq_class_wrds) - np.min(np.array(pre_eq_class_wrds))
    ) / (np.max(np.array(pre_eq_class_wrds)) - np.min(np.array(pre_eq_class_wrds)))
    norm_cap_eq_class_wrds = (
        np.array(cap_eq_class_wrds) - np.min(np.array(cap_eq_class_wrds))
    ) / (np.max(np.array(cap_eq_class_wrds)) - np.min(np.array(cap_eq_class_wrds)))

    diff_pre_probas = np.abs(norm_pre_probas[:-1] - norm_pre_probas[1:])
    diff_cap_probas = np.abs(norm_cap_probas[:-1] - norm_cap_probas[1:])
    diff_post_probas = np.abs(norm_post_probas[:-1] - norm_post_probas[1:])
    diff_pre_eq_class_wrds = np.abs(
        norm_pre_eq_class_wrds[:-1] - norm_pre_eq_class_wrds[1:]
    )
    diff_cap_eq_class_wrds = np.abs(
        norm_cap_eq_class_wrds[:-1] - norm_cap_eq_class_wrds[1:]
    )

    pre_means = np.array(diff_pre_probas).mean(0)
    cap_means = np.array(diff_cap_probas).mean(0)
    post_means = np.array(diff_post_probas).mean(0)
    pre_eq_class_wrds_means = np.array(diff_pre_eq_class_wrds).mean(0)
    cap_eq_class_wrds_means = np.array(diff_cap_eq_class_wrds).mean(0)

    print("Differences in probabilities (first - second, second - third...)")
    print(f"Pre cut:\t{pre_means}")
    print(f"Cut:\t\t{cap_means}")
    print(f"Post cut:\t{post_means}")
    print(f"Eq class pre:\t{pre_eq_class_wrds_means}")
    print(f"Eq class cut:\t{cap_eq_class_wrds_means}")


if __name__ == "__main__":
    main()
