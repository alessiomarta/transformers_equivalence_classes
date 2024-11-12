"""
results dir will be specified as argument in experiment script
"""

import argparse
import os
import random
import shutil
from experiments_utils import save_json, load_json


def gather_files_and_configs(
    input_path: str, exp_dir: str, n_inputs: int, patch_option: str
):
    """
    Gather and copy n_inputs input files from 'input_path' to 'exp_dir' along with relevant configuration data.

    Parameters:
    - input_path: str, the target directory (with either Structure 1 or Structure 2).
    - exp_dir: str, the result directory where files and the JSOn_inputs config will be copied.
    - n_inputs: int, the total number of input files to gather.
    - patch_option: str, used to select the config file variant in each subdirectory.
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
            if patch_option == 'target-word':
                config_path = os.path.join(subdir, f"config.json")
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
        if patch_option == 'target-word':
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

    # Experiment name
    parser.add_argument(
        "-e",
        "--exp_name",
        required=True,
        type=str,
        help="(Mandatory) Name of the experiment for organization and identification.",
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
        required=True,
        type=int,
        help="(Mandatory) Number of iterations to run.",
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
        required=True,
        type=int,
        help="(Mandatory) Number of inputs to use in each experiment.",
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
        required=True,
        choices=["all", "one", "q1", "q2", "q3", "target-word"],
        help="(Mandatory) Number of patches to explore. Options: 'all', 'one', 'q1', 'q2', 'q3', or 'target-word. This latter option is intended for experiments, such as Winobias, where a specific target word is predefined for exploration. The number of tokens will vary based on how the tokenizer segments the target word.",
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
        required=True,
        type=str,
        help="(Mandatory) Directory path where the original input data is located.",
    )

    # Model path
    parser.add_argument(
        "-mp",
        "--model_path",
        required=True,
        type=str,
        help="(Mandatory) Path to the model directory where the pytorch file is, or name of the Huggingface model. Must contain the model config file.",
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
        required=True,
        type=str,
        help="(Mandatory) Directory path where the configuration and data outputs will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    experiment_dir = os.path.join(args.exp_dir, args.exp_name)
    # prepare selection of input to perform the experiments with based on
    input_type = gather_files_and_configs(
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
    # if not run as script, prepare every possible experiment given the fixed parameters grid
    mnist_experiment = {
        "exp_name": "mnist",
        "algo": "both",
        "iterations": 20000,
        "delta_mult": None,
        "threshold": 0.01,
        "save_each": 1,
        "inputs": None,
        "repeat": None,
        "patches": None,
        "orig_data_dir": "../data/mnist_imgs",
        "model_path": "../models/vit",
        "objective": "cls",
        "exp_dir": "../experiments",
    }

    cifar_experiment = {
        "exp_name": "cifar",
        "algo": "both",
        "iterations": 20000,
        "delta_mult": None,
        "threshold": 0.01,
        "save_each": 1,
        "inputs": None,
        "repeat": None,
        "patches": None,
        "orig_data_dir": "../data/cifar10_imgs",
        "model_path": "../models/cifarTrain",
        "objective": "cls",
        "exp_dir": "../experiments/test",
    }

    hate_speech_experiment = {
        "exp_name": "hatespeech",
        "algo": "both",
        "iterations": 20000,
        "delta_mult": None,
        "threshold": 0.01,
        "save_each": 1,
        "inputs": None,
        "repeat": None,
        "patches": None,
        "vocab_tokens": None,
        "orig_data_dir": "../data/measuring-hate-speech_txts",
        "model_path": "ctoraman/hate-speech-bert",
        "objective": "cls",
        "exp_dir": "../experiments/test",
    }

    winobias_experiment = {
        "exp_name": "winobias",
        "algo": "both",
        "iterations": 20000,
        "delta_mult": None,
        "threshold": 0.01,
        "save_each": 1,
        "inputs": None,
        "repeat": None,
        "patches": "target-word",
        "vocab_tokens": None,
        "orig_data_dir": "../data/wino_bias_txts",
        "model_path": "bert-base-uncased",
        "objective": "mlm",
        "exp_dir": "../experiments/test",
    }
    delta_mult = [1, 5]
    inputs=[200, 20]
    patches= []
    vocab_tokens = [1,5,10]
    for experiment in [mnist_experiment, cifar_experiment, hate_speech_experiment, winobias_experiment]:
        
