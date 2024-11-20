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
    input_path: str,
    exp_dir: str,
    n_inputs: int,
    patch_option: str,
    fixed_inputs: list = None,
):
    """
    Generate and configure an experimental directory with enhanced sampling behavior.

    This function organizes input files into a specified experimental directory,
    sampling files from the source directory (flat or hierarchical). It supports:
    - Including a list of fixed input files while ensuring the total sample size matches `n_inputs`.
    - Replacing sampled files with fixed inputs if necessary.
    - Returning the file type and a sample of filenames for inspection.

    Parameters:
    ----------
    input_path : str
        Path to the directory containing input data. Can be:
        - Flat: All files in a single directory.
        - Hierarchical: Subdirectories, each with its configuration and input files.
    exp_dir : str
        Path to the directory where sampled files and the combined configuration JSON will be saved.
    n_inputs : int
        Number of input files to sample for the experiment.
    patch_option : str
        Configuration file type to use in each subdirectory:
        - "target-word" for default config.json.
        - Other options specify files like config_<patch_option>.json.
    fixed_inputs : list, optional
        List of fixed input filenames to include in the sample. Default is an empty list.

    Returns:
    --------
    tuple
        - str: File type of the sampled inputs (e.g., "txt", "png").
        - list: 20% of the sampled file names for inspection.

    Notes:
    ------
    - Creates the output directory (`exp_dir`) if it does not exist.
    - Handles cases where `n_inputs` exceeds available files gracefully.
    - Ensures that fixed inputs are included in the sample.
    - Substitutes files from over-represented directories when needed.
    """

    def collect_config_data(current_folder: str):
        """
        Collect and merge configuration data from a specified folder.

        This function identifies the appropriate configuration file in the given folder
        based on the `patch_option`, loads its contents, and updates the global configuration
        dictionary `all_config_data` with the loaded data.

        Parameters:
        ----------
        current_folder : str
            Path to the folder containing the configuration file.
        """
        # Determine the configuration file based on the patch option
        if patch_option == "target-word":
            config_path = os.path.join(current_folder, "config.json")
        else:
            config_path = os.path.join(current_folder, f"config_{patch_option}.json")

        # Load the configuration data
        config_data = load_json(config_path)
        all_config_data.update(config_data)

    def sample_file(current_folder: str, n_input_in_dir: int):
        """
        Sample files from the given folder and add configuration data.

        Parameters:
        ----------
        current_folder : str
            Directory containing input files and configuration data.
        n_input_in_dir : int
            Number of files to sample from this folder.
        """

        # Gather all eligible input files in the folder
        input_files = [
            os.path.join(current_folder, f)
            for f in os.listdir(current_folder)
            if f.endswith((".txt", ".png", ".jpg", ".jpeg"))
        ]

        # Sample files from the directory (up to the number requested)
        sample_size = min(n_input_in_dir, len(input_files))
        sampled_files = random.sample(input_files, sample_size)

        # Add sampled files to the global list
        all_sampled_files.extend(sampled_files)

    # Ensure result directory exists
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Ensure the output directory exists
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Determine if the source directory has subdirectories (hierarchical structure)
    subdirs = [
        os.path.join(input_path, d)
        for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ]
    has_subdirs = bool(subdirs)

    # Initialize data structures for configuration and sampled files
    combined_config = {}
    all_config_data = {}
    all_sampled_files = []

    # Prepare the fixed input handling
    fixed_inputs = fixed_inputs or []

    if has_subdirs:
        # Collect all needed config
        for subdir in subdirs:
            collect_config_data(subdir)
        # Adjust the sampling distribution across subdirectories
        if n_inputs < len(subdirs):
            # If fewer inputs than subdirectories, randomly select subdirs
            chosen_subdirs = random.sample(subdirs, n_inputs)
            files_per_subdir = 1
        else:
            # Evenly distribute sampling across all subdirectories
            chosen_subdirs = subdirs
            files_per_subdir = n_inputs // len(subdirs)

        # Sample files from the selected subdirectories
        for subdir in chosen_subdirs:
            sample_file(subdir, files_per_subdir)

    else:
        # Collect all needed config
        collect_config_data(input_path)
        # Sample files from a flat directory structure
        sample_file(input_path, n_inputs)

    # Handle fixed inputs: Ensure they are included in the sampled files
    fixed_inputs_to_add = set(fixed_inputs) - set(all_sampled_files)
    for fixed_input in fixed_inputs_to_add:
        # Find files to substitute with fixed inputs
        parent_dir = os.path.dirname(fixed_input)
        same_parent_dir = [
            i for i in all_sampled_files if os.path.dirname(i) == parent_dir
        ]
        if same_parent_dir:
            # Substitute a file from the same directory
            substitute = random.sample(same_parent_dir, 1)[0]
        else:
            # Substitute from the most represented directory
            dirnames = [os.path.dirname(f) for f in all_sampled_files]
            parent_dirs = {f: dirnames.count(f) for f in dirnames}
            max_parent = max(parent_dirs, key=parent_dirs.get)
            substitute = [s for s in all_sampled_files if s.startswith(max_parent)][0]
        substitute_idx = all_sampled_files.index(substitute)
        all_sampled_files[substitute_idx] = fixed_input

    # Copy all sampled files to the experiment directory
    for input_file in all_sampled_files:
        shutil.copy(input_file, exp_dir)
        input_name = os.path.basename(input_file)
        combined_config[input_name] = all_config_data[input_name.split(".")[0]]

    # Save the combined configuration file
    combined_config_path = os.path.join(exp_dir, "config.json")
    save_json(combined_config_path, combined_config)

    # Return the file type and a 20% subset of the sampled files
    file_type = os.path.splitext(all_sampled_files[0])[1][1:]
    inspected_files = random.sample(
        all_sampled_files, max(1, (len(all_sampled_files) * 20) // 100)
    )

    print(f"Successfully copied {n_inputs} files and created config.json in {exp_dir}")
    return file_type, inspected_files


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
                "model_path": "../models/mnist_experiment",
            },
            "cifar": {
                "exp_name": "cifar",
                "orig_data_dir": "../data/cifar10_imgs",
                "model_path": "../models/cifar10_experiment",
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

            for n_input in INPUT_VALUES:
                base_experiment["inputs"] = n_input
                if n_input in [200, 10]:
                    base_experiment["repeat"] = 1
                else:
                    if args.test:
                        base_experiment["repeat"] = 5
                    else:
                        base_experiment["repeat"] = 10

                if args.test:
                    sample = None

                for delta in DELTA_MULT_VALUES:
                    base_experiment["delta_mult"] = delta

                    # Use specific patches if defined, otherwise use all options
                    patch_list = (
                        [base_experiment["patches"]]
                        if base_experiment["patches"] == "target-word"
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
                            if args.test:
                                # If run under test condition, ensure that experiments are done considering at least 20% of the sample fixed to compare results over experiments
                                if sample:
                                    generate_experiment(
                                        input_path=base_experiment["orig_data_dir"],
                                        exp_dir=experiment_dir,
                                        n_inputs=n_input,
                                        patch_option=patch_opt,
                                        fixed_inputs=sample,
                                    )
                                else:
                                    _, sample = generate_experiment(
                                        input_path=base_experiment["orig_data_dir"],
                                        exp_dir=experiment_dir,
                                        n_inputs=n_input,
                                        patch_option=patch_opt,
                                    )
                            else:
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
    print("Done")
