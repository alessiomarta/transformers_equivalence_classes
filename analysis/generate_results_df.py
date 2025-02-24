"""
Experiment Processing Module 

This module processes experiment results by reading metadata, extracting results,
and saving them directly as Parquet files using PyArrow.

Usage:
---------------
To run the module:
    python experiment_processor.py --results-path ../res --metadata-path experiments_data/test --output-path output/
"""

import os
import json
import pickle
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def load_pickle(filepath: str):
    """Load a pickle file."""
    with open(filepath, "rb") as file:
        return pickle.load(file)


def load_json(filepath: str) -> dict:
    """Load a JSON file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


class ExperimentInput:
    """Represents an input instance for an experiment."""

    def __init__(
        self, input_name, equivalent_class_patch, explore_patches, attribution_patches
    ):
        self.input_name = input_name
        self.equivalence_class_patch = equivalent_class_patch
        self.explore_patches = explore_patches
        self.attribution_patches = attribution_patches

    def __repr__(self):
        return f"ExperimentInput({self.input_name})"


class Experiment:
    """Handles an individual experiment, including metadata and results collection."""

    def __init__(
        self, metadata, algorithm, input_instance, repetition, results_path=None
    ):
        self.metadata = metadata
        self.algorithm = algorithm
        self.input_instance = input_instance
        self.repetition = repetition
        self.iterations = []

        if results_path:
            self.collect_results(results_path)

    def collect_results(self, results_path: str):
        """Collects results for the experiment from the given directory."""
        experiment_path = os.path.join(
            results_path,
            f"{self.algorithm}-{self.input_instance.input_name.split('.')[0]}-{self.repetition}",
        )

        if not os.path.exists(experiment_path):
            print(f"Warning: No results found at {experiment_path}")
            return

        for root, _, files in os.walk(experiment_path):
            for file in files:
                if file.endswith(".pkl") and "stats" not in file:
                    file_path = os.path.join(root, file)
                    results = load_pickle(file_path)
                    iteration = results.get("iteration")

                    stats_path = os.path.join(
                        root, "interpretation", f"{iteration}-stats.pkl"
                    )
                    stat_results = (
                        load_pickle(stats_path) if os.path.exists(stats_path) else {}
                    )
                    # TODO controllare perchè stat_results["modified_patches"].shape a volte è 3 a volte 2
                    # nel frattempo lo tolgo
                    if "modified_patches" in stat_results:
                        del stat_results["modified_patches"]

                    results.update(stat_results)
                    self.iterations.append(ExperimentIteration(results))

    def to_arrow_table(self) -> pa.Table:
        """Converts experiment data into a PyArrow Table."""
        records = [
            {
                **iteration.to_dict(),
                **self.metadata.to_dict(),
                **self.input_instance.__dict__,
                "algorithm": self.algorithm,
                "repetition": self.repetition,
            }
            for iteration in self.iterations
        ]

        # Convert list of dictionaries to Arrow Table
        return (
            pa.Table.from_pydict({k: [d[k] for d in records] for k in records[0]})
            if records
            else None
        )

    def __repr__(self):
        return f"{self.algorithm}-{self.input_instance.input_name.split('.')[0]}-{self.repetition}"


class ExperimentMetadata:
    """Handles experiment metadata and generates experiments from it."""

    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path

        # Load experiment parameters
        parameters = load_json(os.path.join(metadata_path, "parameters.json"))
        for key, value in parameters.items():
            setattr(self, key, value)
        self.inputs = []

        # Load experiment configuration
        config = load_json(os.path.join(metadata_path, "config.json"))
        for input_name, values in config.items():
            if len(values["attrib"]) < 1:
                values["explore"] = list(
                    range(1, (values["img_dim"][-1] // values["patch_dim"]) ** 2)
                )
                values["attrib"] = list(map(float, values["explore"]))
                # TODO sistemare in modo tale da predere l'attribution vera quando esploro tutte le patch
            self.inputs.append(
                ExperimentInput(
                    input_name=input_name,
                    equivalent_class_patch=values["objective"],
                    explore_patches=values["explore"],
                    attribution_patches=values["attrib"],
                )
            )

    def generate_experiments(self):
        """Yields Experiment instances based on metadata configurations."""
        if not hasattr(self, "algo") or not hasattr(self, "repeat"):
            raise AttributeError(
                "Metadata must contain 'algo' and 'repeat' attributes."
            )

        algorithms = ["simec"] if self.algo != "simexp" else ["simexp"]
        if self.algo == "both":
            algorithms.append("simexp")

        for algorithm in algorithms:
            for input_instance in self.inputs:
                for repetition in range(self.repeat):
                    yield Experiment(self, algorithm, input_instance, repetition + 1)

    def to_dict(self):
        """Converts metadata to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if k != "inputs"}

    def __repr__(self):
        return f"ExperimentMetadata({os.path.basename(self.metadata_path)})"


class ExperimentIteration:
    """Represents a single iteration within an experiment."""

    def __init__(self, results: dict):
        for key, value in results.items():
            if hasattr(value, "tolist"):  # Convert tensors to lists
                value = value.tolist()
            setattr(self, key, value)

    def to_dict(self):
        """Converts iteration data to a dictionary."""
        return self.__dict__.copy()


def pair_experiment_folders(results_dir: str, metadata_dir: str) -> dict:
    """
    Pairs experiment result folders with their corresponding metadata folders.
    """

    def list_dirs(path):
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    # Find matching result folders
    result_folders = [
        folder for folder in list_dirs(results_dir) if "20250217" in folder
    ]

    # Ensure each folder contains an "interpretation" subfolder
    valid_result_folders = [
        folder
        for folder in result_folders
        if any(
            os.path.exists(
                os.path.join(results_dir, folder, subfolder, "interpretation")
            )
            for subfolder in os.listdir(os.path.join(results_dir, folder))
            if os.path.isdir(os.path.join(results_dir, folder, subfolder))
        )
    ]

    # Find matching metadata folders
    metadata_folders = list_dirs(metadata_dir)

    # Pair matching folders
    return {
        os.path.join(metadata_dir, meta_folder): os.path.join(results_dir, res_folder)
        for res_folder in valid_result_folders
        for meta_folder in metadata_folders
        if meta_folder in res_folder
    }


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and aggregate experiment results."
    )
    parser.add_argument(
        "--results-path", type=str, default="../res", help="Path to experiment results."
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="experiments_data/test",
        help="Path to metadata files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for processed results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_path, exist_ok=True)

    paired_folders = pair_experiment_folders(args.results_path, args.metadata_path)
    all_experiments = []

    for exp_metadata_path, res_path in tqdm(
        paired_folders.items(), desc="Processing Experiments"
    ):
        exp_metadata = ExperimentMetadata(exp_metadata_path)

        for experiment in tqdm(
            exp_metadata.generate_experiments(),
            desc="Generating Experiments",
            leave=False,
        ):
            experiment.collect_results(res_path)
            table = experiment.to_arrow_table()

            if table:
                output_path = os.path.join(
                    args.output_path, os.path.basename(res_path), str(experiment)
                )
                os.makedirs(output_path, exist_ok=True)
                pq.write_table(
                    table, os.path.join(output_path, "experiment_results.parquet")
                )
                all_experiments.append(table)

    final_table = pa.concat_tables(all_experiments) if all_experiments else None
    if final_table:
        pq.write_table(
            final_table,
            os.path.join(args.output_path, "all_experiments_results.parquet"),
        )
