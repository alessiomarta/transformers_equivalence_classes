import pickle
import os
import json
import pandas as pd
import argparse
from tqdm import tqdm


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


class ExperimentInput:
    def __init__(
        self, input_name, equivalent_class_patch, explore_patches, attribution_patches
    ):
        self.input_name = input_name
        self.equivalence_class_patch = equivalent_class_patch
        self.explore_patches = explore_patches
        self.attribution_patches = attribution_patches

    def __str__(self):
        return f"ExperimentInput: {self.input_name}"

    def __repr__(self):
        return f"ExperimentInput: {self.input_name}"


class Experiment:
    def __init__(
        self,
        experiment_metadata,
        algorithm,
        input_instance,
        repetition,
        results_path=None,
    ):
        self.metadata = experiment_metadata
        self.algorithm = algorithm
        self.input_instance = input_instance
        self.repetition = repetition
        self.iterations = []
        if results_path:
            self.collect_results(results_path)

    def collect_results(self, results_path):
        single_result_path = os.path.join(
            results_path,
            f"{self.algorithm}-{self.input_instance.input_name.split('.')[0]}-{self.repetition}",
        )
        for root, dirs, files in os.walk(single_result_path):
            for file in files:
                if file.endswith(".pkl") and "stats" not in file:
                    file_path = os.path.join(root, file)
                    results = load_pickle(file_path)
                    iteration = results["iteration"]
                    stat_results = load_pickle(
                        os.path.join(
                            root, "interpretation", f"{str(iteration)}-stats.pkl"
                        )
                    )
                    results = {**results, **stat_results}
                    self.iterations.append(ExperimentIteration(results))

    def export_dataframe(self):
        data = []
        for iteration in self.iterations:
            row = iteration.__dict__.copy()
            row.update(self.metadata.__dict__.copy())
            row.update(self.input_instance.__dict__.copy())
            row.update(
                {
                    "algorithm": self.algorithm,
                    "repetition": self.repetition,
                }
            )
            data.append(row)
        return pd.DataFrame(data)

    def __repr__(self):
        return f"{self.algorithm}-{self.input_instance.input_name.split('.')[0]}-{self.repetition}"

    def __str__(self):
        return f"{self.algorithm}-{self.input_instance.input_name.split('.')[0]}-{self.repetition}"


class ExperimentMetadata:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        parameters = load_json(os.path.join(metadata_path, "parameters.json"))
        for k, v in parameters.items():
            setattr(self, k, v)
        config = load_json(os.path.join(metadata_path, "config.json"))
        self.inputs = []
        for k, v in config.items():
            self.inputs.append(
                ExperimentInput(
                    input_name=k,
                    equivalent_class_patch=config[k]["objective"],
                    explore_patches=config[k]["explore"],
                    attribution_patches=config[k]["attrib"],
                )
            )

    def get_generated_experiments(self):
        algorithms = ["simec"]
        if self.algo == "both":
            algorithms.append("simexp")
        elif self.algo == "simexp":
            algorithms[0] = "simexp"
        for algo in algorithms:
            for input_instance in self.inputs:
                for repetition in range(self.repeat):
                    yield Experiment(
                        experiment_metadata=self,
                        algorithm=algo,
                        input_instance=input_instance,
                        repetition=repetition + 1,
                    )

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, default=str)

    def __repr__(self):
        return f"Experiment: {self.metadata_path.split('/')[-1]}"


class ExperimentIteration:
    def __init__(self, results):
        for k, v in results.items():
            setattr(self, k, v)


def pair_folders(directory_path: str, transformers_directory_path: str) -> dict:
    # List all folders in the directory
    folders = [
        f
        for f in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, f))
    ]

    # Filter folders that contain "20250217" and either "cifar" or "mnist"
    filtered_folders = [
        folder
        for folder in folders
        if "20250217" in folder and ("cifar" in folder or "mnist" in folder)
    ]

    # Filter folders that contain an "interpretation" folder inside them
    final_folders = []
    for folder in filtered_folders:
        subfolders = [
            f
            for f in os.listdir(os.path.join(directory_path, folder))
            if os.path.isdir(os.path.join(directory_path, folder, f))
        ]
        for subfolder in subfolders:
            if os.path.isdir(
                os.path.join(directory_path, folder, subfolder, "interpretation")
            ):
                final_folders.append(folder)
                break

    # List all folders in the transformers directory
    transformers_folders = [
        f
        for f in os.listdir(transformers_directory_path)
        if os.path.isdir(os.path.join(transformers_directory_path, f))
    ]

    # Pair final_folders with folders in transformers_directory_path
    paired_folders = {}
    for folder in final_folders:
        for t_folder in transformers_folders:
            if t_folder in folder:
                paired_folders[os.path.join(transformers_directory_path, t_folder)] = (
                    os.path.join(directory_path, folder)
                )
                break

    return paired_folders


def parse_args():
    parser = argparse.ArgumentParser(description="Process experiment results.")
    parser.add_argument(
        "--results-path",
        type=str,
        default="../res",
        help="Path to the directory containing all experiments results.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="experiments_data/test",
        help="Path to the directory containing experiments metadata.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the concatenated dataframe of all experiment results.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    experiments_df = []
    paired_folders = pair_folders(args.results_path, args.metadata_path)
    for k, v in tqdm(paired_folders.items(), desc="Processing paired folders"):
        exp_metadata = ExperimentMetadata(metadata_path=k)
        for exp in tqdm(
            exp_metadata.get_generated_experiments(),
            desc="Generating experiments",
            leave=False,
        ):
            exp.collect_results(v)
            df = exp.export_dataframe()
            if not os.path.exists(
                os.path.join(args.output_path, v.split("/")[-1], str(exp))
            ):
                os.makedirs(os.path.join(args.output_path, v.split("/")[-1], str(exp)))
            df.to_csv(
                os.path.join(
                    args.output_path,
                    v.split("/")[-1],
                    str(exp),
                    "experiment_results.csv",
                ),
                index=False,
            )
            experiments_df.append(df)
    experiments_df = pd.concat(experiments_df)
    experiments_df.to_csv(
        os.path.join(args.output_path, "all_experiments_results.csv"),
        index=False,
    )
