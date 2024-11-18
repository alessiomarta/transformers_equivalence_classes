"""
Module for Preparing Experimental Configurations

This module is designed to prepare experimental setups for machine learning research. It includes functionalities to 
organize input data, configure experimental parameters, and manage outputs. The results directory is specified 
as an argument when running the experiment script.

Features:
- Dynamically creates experiment directories and populates them with input data and configurations.
- Supports sampling input files from structured or flat data directories.
- Handles multiple experimental setups based on parameter grids or user-specified configurations.
- Saves experimental configurations in JSON format for reproducibility.

Usage:
If running this script from the parent directory and the `experiments_utils` module is not recognized, 
set the `PYTHONPATH` environment variable to the current working directory:
    export PYTHONPATH=$(pwd)
"""

import argparse
import os
import copy
import random
import shutil
from experiments_utils import (
    save_json,
    load_json,
)  # if this is not recognised when run from parent directory, run 'export PYTHONPATH=$(pwd)' in the parent directory before


def generate_experiment(
    input_path: str, exp_dir: str, n_inputs: int, patch_option: str
):
    """
    Generate and configure an experimental directory.

    This function organizes an experimental setup by creating a result directory,
    sampling input files from a source directory, and saving relevant configuration data.

    Parameters:
    ----------
    input_path : str
        Path to the directory containing input data. The structure can either be:
        - Flat structure with all files in one directory and a single configuration file.
        - Hierarchical structure with subdirectories, each containing a configuration file and related input files.
    exp_dir : str
        Path to the result directory where sampled input files and the combined configuration JSON will be saved.
    n_inputs : int
        Number of input files to sample and include in the experiment directory.
        If the directory contains subdirectories, the function distributes the sampling across them.
    patch_option : str
        Specifies the type of configuration file to use in each subdirectory:
        - "target-word" for a specific configuration.
        - Other options correspond to specific file naming conventions, e.g., "config_<patch_option>.json".

    Behavior:
    ---------
    - Checks if the source directory has subdirectories (hierarchical structure).
    - Samples input files evenly across subdirectories or from a flat directory.
    - Copies the sampled input files to the result directory.
    - Reads configuration files and combines them into a single JSON file saved in the result directory.

    Returns:
    --------
    str
        File type (extension) of the input files sampled (e.g., "txt", "png").
        This helps in further processing, such as differentiating between text and image experiments.

    Notes:
    ------
    - If `n_inputs` exceeds the number of available files, all files are included with a warning.
    - Creates the result directory if it does not exist.
    - Handles missing or improperly structured data directories gracefully.

    Example:
    --------
    generate_experiment(
        input_path="./data/inputs",
        exp_dir="./results/experiment_1",
        n_inputs=10,
        patch_option="target-word"
    )
    """
    # Ensure result directory exists
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Check if we have subdirectories (Structure 2) or a flat structure (Structure 1)
    subdirs = [
        os.path.join(input_path, d)
        for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ]
    has_subdirs = bool(subdirs)

    # Initialize the result configuration dictionary
    combined_config = {}

    if has_subdirs:
        # Adjust sampling based on the number of subdirectories
        if n_inputs < len(subdirs):
            # Randomly select `n_inputs` subdirectories if there are more subdirs than needed files
            chosen_subdirs = random.sample(subdirs, n_inputs)
            files_per_subdir = 1
        else:
            # Use all subdirectories and sample approximately `files_per_subdir` files from each
            chosen_subdirs = subdirs
            files_per_subdir = n_inputs // len(subdirs)

        for subdir in chosen_subdirs:
            if patch_option == "target-word":
                config_path = os.path.join(subdir, "config.json")
            else:
                config_path = os.path.join(subdir, f"config_{patch_option}.json")
            config_data = load_json(config_path)

            # Collect eligible input files
            input_files = [
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if f.endswith((".txt", ".png", ".jpg", ".jpeg"))
            ]

            # Determine the number of files to sample in this subdir
            sample_size = min(files_per_subdir, len(input_files))
            if sample_size < files_per_subdir:
                print(
                    f"Warning: Requested sample size per subdirectory ({files_per_subdir}) exceeds available files ({len(input_files)}) in '{subdir}'. "
                    f"Proceeding with all available files instead."
                )

            sampled_files = random.sample(input_files, sample_size)

            # Copy files and collect config data
            for input_file in sampled_files:
                shutil.copy(input_file, exp_dir)
                input_name = os.path.basename(input_file)
                combined_config[input_name] = config_data.get(
                    input_name.split(".")[0], {}
                )

    else:
        # Structure without subdirectories: Single directory with a single config.json and input files
        if patch_option == "target-word":
            config_path = os.path.join(input_path, f"config.json")
        else:
            config_path = os.path.join(input_path, f"config_{patch_option}.json")
        config_data = load_json(config_path)

        # Collect eligible input files
        input_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith((".txt", ".png", ".jpg", ".jpeg"))
        ]
        if n_inputs > len(input_files):
            print(
                f"Warning: Requested sample size ({n_inputs}) exceeds available input files ({len(input_files)}). "
                f"Proceeding with all available files instead."
            )
        sampled_files = random.sample(input_files, min(n_inputs, len(input_files)))

        # Copy files and collect config data
        for input_file in sampled_files:
            shutil.copy(input_file, exp_dir)
            input_name = os.path.basename(input_file)
            combined_config[input_name] = config_data.get(input_name.split(".")[0], {})

    # Save the combined configuration in the result directory
    combined_config_path = os.path.join(exp_dir, "config.json")
    save_json(combined_config_path, combined_config)

    print(f"Successfully copied {n_inputs} files and created config.json in {exp_dir}")

    return input_file.split(".")[-1]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare and configure experimental parameters for running research experiments."
    )

    # Default experiments
    parser.add_argument(
        "--default",
        action="store_true",
        help="If set, generates the default experiments",
    )

    # Default test experiments
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, generates the default test experiments",
    )

    # Experiment name
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        help="(Mandatory if --default is not set) Name of the experiment for organization and identification.",
    )

    # Algorithm selection
    parser.add_argument(
        "-a",
        "--algo",
        choices=["simec", "simexp", "both"],
        default="both",
        help="Algorithm to run. Choose 'simec' to run Simec only, 'simexp' to run Simexp only, or omit to run both.",
    )

    # Number of iterations
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        help="(Mandatory if --default is not set) Number of iterations to run.",
    )

    # Delta multiplier
    parser.add_argument(
        "-d",
        "--delta_mult",
        type=int,
        default=1,
        help="Multiplier for delta adjustment. Defaults to 1 if unspecified.",
    )

    # Threshold
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1e-2,
        help="Threshold value for the experiment. Defaults to 0.01 if unspecified.",
    )

    # Save frequency
    parser.add_argument(
        "-s",
        "--save_each",
        type=int,
        default=1,
        help="Frequency of saving results. Save after every n iterations, where n is this value. Defaults to 1.",
    )

    # Number of inputs per experiment
    parser.add_argument(
        "-i",
        "--inputs",
        type=int,
        help="(Mandatory if --default is not set) Number of inputs to use in each experiment.",
    )

    # Repeat experiment per input
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the experiment for each input. Defaults to 1 if unspecified.",
    )

    # Number of patches to explore
    parser.add_argument(
        "-p",
        "--patches",
        choices=["all", "one", "q1", "q2", "q3", "target-word"],
        help="(Mandatory if --default is not set) Number of patches to explore. Options: 'all', 'one', 'q1', 'q2', 'q3', or 'target-word. This latter option is intended for experiments, such as Winobias, where a specific target word is predefined for exploration. The number of tokens will vary based on how the tokenizer segments the target word.",
    )

    # Number of vocabulary tokens in percentile
    parser.add_argument(
        "-v",
        "--vocab_tokens",
        type=int,
        choices=range(1, 101),
        default=1,
        help="Percentage of vocabulary tokens (1-100) to consider for interpreting textual experiments. Mandatory if conducting a textual experiment.",
    )

    # Original data directory
    parser.add_argument(
        "-od",
        "--orig_data_dir",
        type=str,
        help="(Mandatory if --default is not set) Directory path where the original input data is located.",
    )

    # Model path
    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        help="(Mandatory if --default is not set) Path to the model directory where the pytorch file is, or name of the Huggingface model. Must contain the model config file.",
    )

    # Model task
    parser.add_argument(
        "-o",
        "--objective",
        default="cls",
        choices=["cls", "mlm"],
        type=str,
        help="Type of task that the chosen model is supposed to perform. Options: 'cls' (classification), 'mlm' (fill-mask). Defaults to 'cls' in unspecified",
    )

    # Experiment directory
    parser.add_argument(
        "-ed",
        "--exp_dir",
        type=str,
        help="(Mandatory if --default is not set) Directory path where the configuration and data outputs will be saved.",
    )

    return parser, parser.parse_args()


if __name__ == "__main__":
    argparser, args = parse_arguments()
    if not args.default:
        missing_args = []
        for a_name, a in [
            ("-ed/--exp_dir", args.exp_dir),
            ("-mp/--model_path", args.model_path),
            ("-od/--orig_data_dir", args.orig_data_dir),
            ("-p/--patches", args.patches),
            ("-i/--inputs", args.inputs),
            ("-n/--iterations", args.iterations),
            ("-e/--exp_name", args.exp_name),
        ]:
            if a is None:
                missing_args.append(a_name)
        if len(missing_args) > 0:
            argparser.error(
                f"--default option not selected: preparing individual experiment. The following arguments are required: {', '.join(missing_args)}"
            )
        experiment_dir = os.path.join(args.exp_dir, args.exp_name)
        # prepare selection of input to perform the experiments with based on specified parameters
        input_type = generate_experiment(
            input_path=args.orig_data_dir,
            exp_dir=experiment_dir,
            n_inputs=args.inputs,
            patch_option=args.patches,
        )
        # preparing experiment configuration file, where all parameters will be specified
        if input_type != "txt":
            parameters = vars(args)
            del parameters["vocab_tokens"]
            save_json(os.path.join(experiment_dir, "parameters.json"), parameters)
        else:
            save_json(os.path.join(experiment_dir, "parameters.json"), vars(args))
    else:
        # if not run as individual experiment, prepare every possible experiment given the fixed parameters grid
        if args.test:
            print("Warning: creating experiments with --test option.")
        # Define a base experiment template
        BASE_EXPERIMENT = {
            "algo": "both",
            "iterations": 10 if args.test else 20000,
            "delta_mult": None,
            "threshold": 0.01,
            "save_each": 1,
            "inputs": None,
            "repeat": None,
            "patches": None,
            "objective": "cls",
        }

        # Define specific experiments with overrides
        EXPERIMENTS = {
            "mnist": {
                "exp_name": "mnist",
                "orig_data_dir": "../data/mnist_imgs",
                "model_path": "../models/vit",
            },
            "cifar": {
                "exp_name": "cifar",
                "orig_data_dir": "../data/cifar10_imgs",
                "model_path": "../models/cifarTrain",
            },
            "hatespeech": {
                "exp_name": "hatespeech",
                "orig_data_dir": "../data/measuring-hate-speech_txts",
                "model_path": "ctoraman/hate-speech-bert",
                "vocab_tokens": None,
            },
            "winobias": {
                "exp_name": "winobias",
                "orig_data_dir": "../data/wino_bias_txts",
                "model_path": "bert-base-uncased",
                "patches": "target-word",
                "objective": "mlm",
                "vocab_tokens": None,
            },
        }

        # Define parameter variations
        DELTA_MULT_VALUES = [1, 5]
        INPUT_VALUES = [10, 2] if args.test else [200, 20]
        PATCH_OPTIONS = ["all", "one", "q1", "q2", "q3"]
        VOCAB_TOKENS_VALUES = [1, 5, 10]

        # Iterate through experiments and generate configurations
        for exp_name, exp_overrides in EXPERIMENTS.items():
            base_experiment = copy.deepcopy(BASE_EXPERIMENT)
            base_experiment.update(exp_overrides)

            for delta in DELTA_MULT_VALUES:
                base_experiment["delta_mult"] = delta

                for n_input in INPUT_VALUES:
                    base_experiment["inputs"] = n_input
                    if n_input in [200, 10]:
                        base_experiment["repeat"] = 1
                    else:
                        if args.test:
                            base_experiment["repeat"] = 5
                        else:
                            base_experiment["repeat"] = 10

                    # Use specific patches if defined, otherwise use all options
                    patch_list = (
                        [base_experiment["patches"]]
                        if base_experiment["patches"]
                        else PATCH_OPTIONS
                    )

                    for patch_opt in patch_list:
                        base_experiment["patches"] = patch_opt

                        # Use vocab tokens if relevant to the experiment
                        vocab_token_list = (
                            VOCAB_TOKENS_VALUES
                            if "vocab_tokens" in base_experiment
                            else [None]
                        )

                        for n_vocab in vocab_token_list:
                            if n_vocab:
                                base_experiment["vocab_tokens"] = n_vocab

                            # Generate a descriptive experiment name
                            exp_name_full = f"{base_experiment['exp_name']}-{delta}-{n_input}-{patch_opt}"
                            if n_vocab:
                                exp_name_full += f"-{n_vocab}"
                            if args.test:
                                exp_name_full = "test/" + exp_name_full

                            experiment_dir = os.path.join(
                                "experiments_data", exp_name_full
                            )

                            # Generate experiment and save parameters
                            generate_experiment(
                                input_path=base_experiment["orig_data_dir"],
                                exp_dir=experiment_dir,
                                n_inputs=n_input,
                                patch_option=patch_opt,
                            )

                            # Filter out keys with None values
                            filtered_experiment = {
                                k: v
                                for k, v in base_experiment.items()
                                if v is not None
                            }

                            save_json(
                                os.path.join(experiment_dir, "parameters.json"),
                                filtered_experiment,
                            )
