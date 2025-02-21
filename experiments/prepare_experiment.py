"""
Module for Preparing Experimental Configurations

This module provides tools to prepare experimental setups for machine learning research.
It includes functionalities for organizing input data, configuring experimental parameters,
sampling input files, and measuring embedding distributions from models. It is especially
useful for handling structured datasets and configuring experiments reproducibly.

Features:
---------
- Dynamically creates experimental directories and populates them with input data and configurations.
- Supports sampling from structured or flat data directories, with configurable sampling behaviors.
- Manages input data across multiple experimental setups using parameter grids or predefined configurations.
- Measures embedding distributions for image and text datasets using specified machine learning models.

Usage:
------
To ensure module recognition, set the `PYTHONPATH` environment variable to the working directory:
    export PYTHONPATH=$(pwd)
"""

import argparse
import os
import copy
import random
import shutil
import torch
from transformers import logging
from experiments.experiments_utils import (
    save_json,
    load_json,
    load_model,
    load_bert_model,
    deactivate_dropout_layers,
    load_and_transform_raw_images,
    load_raw_sents,
    save_object,
    compute_embedding_boundaries,
)
from experiments.models.const import CIFAR_MEAN, CIFAR_STD, MNIST_MEAN, MNIST_STD


def mask_random_word(sentences, mask_token, classification_token):
    """
    Replace a random word in each sentence with a specified mask token.

    If the sentence contains the classification token (e.g., `[CLS]`), the first and last
    words are excluded from consideration for replacement.

    Parameters:
    ----------
    sentences : list of str
        List of sentences to process.
    mask_token : str
        Token to replace the selected word, e.g., `[MASK]`.
    classification_token : str
        Special token (e.g., `[CLS]`) that affects replacement rules.

    Returns:
    -------
    list of str
        List of sentences with one word replaced by the mask token in each.
    """
    masked_sentences = []

    for sentence in sentences:
        words = sentence.split()  # Split sentence into words
        num_words = len(words)

        # Determine the range of words eligible for masking
        if classification_token in sentence:
            replaceable_indices = range(1, num_words - 1) if num_words > 2 else []
        else:
            replaceable_indices = range(num_words)

        # Replace a random word if eligible indices exist
        if replaceable_indices:
            mask_index = random.choice(replaceable_indices)
            words[mask_index] = mask_token

        masked_sentences.append(" ".join(words))

    return masked_sentences


def generate_experiment(input_path, exp_dir, n_inputs, patch_option, fixed_inputs=None):
    """
    Organize input files and configurations into an experimental directory.

    Samples input files from the source directory and collects their configurations,
    preparing a reproducible experimental setup.

    Parameters:
    ----------
    input_path : str
        Path to the directory containing input files. Can be:
        - Flat: All files are in a single directory.
        - Hierarchical: Contains subdirectories with input files and configurations.
    exp_dir : str
        Path to the directory where the sampled files and configuration JSON will be saved.
    n_inputs : int
        Number of input files to sample.
    patch_option : str
        Specifies the configuration type for subdirectories:
        - "target-word" for default `config.json`.
        - Other options specify files like `config_<patch_option>.json`.
    fixed_inputs : list, optional
        List of specific input files to include in the experiment. Default is None.

    Returns:
    -------
    tuple:
        - str: File type of the sampled inputs (e.g., `txt`, `png`).
        - list: A sample of filenames (20%) for inspection.
    """

    def collect_config_data(folder):
        """
        Load and merge configuration data from a folder.

        Parameters:
        ----------
        folder : str
            Path to the folder containing configuration files.
        """
        config_path = (
            os.path.join(folder, "config.json")
            if patch_option == "target-word"
            else os.path.join(folder, f"config_{patch_option}.json")
        )
        config_data = load_json(config_path)
        all_config_data.update(config_data)

    def sample_files(folder, sample_size):
        """
        Sample files from a folder and update global sampled files list.

        Parameters:
        ----------
        folder : str
            Directory containing input files to sample.
        sample_size : int
            Number of files to sample.
        """
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith((".txt", ".png", ".jpg", ".jpeg"))
        ]
        sampled = random.sample(files, min(sample_size, len(files)))
        all_sampled_files.extend(sampled)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    subdirs = [
        os.path.join(input_path, d)
        for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ]
    has_subdirs = bool(subdirs)

    combined_config = {}
    all_config_data = {}
    all_sampled_files = []
    fixed_inputs = fixed_inputs or []

    if has_subdirs:
        for subdir in subdirs:
            collect_config_data(subdir)

        chosen_subdirs = (
            random.sample(subdirs, n_inputs) if n_inputs < len(subdirs) else subdirs
        )
        files_per_subdir = n_inputs // len(chosen_subdirs)

        for subdir in chosen_subdirs:
            sample_files(subdir, files_per_subdir)
    else:
        collect_config_data(input_path)
        sample_files(input_path, n_inputs)

    fixed_inputs_to_add = set(fixed_inputs) - set(all_sampled_files)
    for fixed_input in fixed_inputs_to_add:
        parent_dir = os.path.dirname(fixed_input)
        same_dir_files = [
            f for f in all_sampled_files if os.path.dirname(f) == parent_dir
        ]
        substitute_file = (
            random.choice(same_dir_files)
            if same_dir_files
            else random.choice(all_sampled_files)
        )
        substitute_idx = all_sampled_files.index(substitute_file)
        all_sampled_files[substitute_idx] = fixed_input

    for input_file in all_sampled_files:
        shutil.copy(input_file, exp_dir)
        input_name = os.path.basename(input_file)
        combined_config[input_name] = all_config_data.get(input_name.split(".")[0], {})

    combined_config_path = os.path.join(exp_dir, "config.json")
    save_json(combined_config_path, combined_config)

    file_type = os.path.splitext(all_sampled_files[0])[1][1:]
    inspected_files = random.sample(
        all_sampled_files, max(1, len(all_sampled_files) // 5)
    )

    print(f"Created experiment in {exp_dir} with {n_inputs} files.")
    return file_type, inspected_files


def measure_embedding_distribution(
    data_dir, model, device, tokenizer=None, objective=None
):
    """
    Measure the distribution of embeddings for images or text data.

    For images: Computes the min and max embeddings using a loaded model.
    For text: Optionally applies masking for MLM objectives, then computes embeddings.

    Parameters:
    ----------
    data_dir : str
        Path to the dataset directory.
    model : torch.nn.Module
        The model used to compute embeddings.
    device : torch.device
        Device to run the computations (CPU or GPU).
    tokenizer : transformers.PreTrainedTokenizer, optional
        Tokenizer for text data. Required if processing text data.
    objective : str, optional
        Objective for the model (e.g., "mlm" for masked language modeling). Required if processing text data.

    Returns:
    -------
    tuple:
        - torch.Tensor: Minimum embeddings.
        - torch.Tensor: Maximum embeddings.
    """
    if any(k in data_dir for k in ["cifar", "mnist"]):
        images = [
            load_and_transform_raw_images(os.path.join(data_dir, d))[0]
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]

        images = (
            torch.cat(images).to(device)
            if isinstance(images, list)
            else images.to(device)
        )

        patches = model.embedding(model.patcher(images))
        return patches.min(dim=0).values.unsqueeze(0), patches.max(
            dim=0
        ).values.unsqueeze(0)

    logging.set_verbosity_error()

    subdirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    has_subdirs = bool(subdirs)
    if has_subdirs:
        txts = []
        for subdir in subdirs:
            txts += load_raw_sents(subdir)[0]
    else:
        txts, _ = load_raw_sents(data_dir)

    if objective == "mlm":
        txts = mask_random_word(
            sentences=txts,
            mask_token=tokenizer.mask_token,
            classification_token=tokenizer.cls_token,
        )
    tokenized = tokenizer(
        txts,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=False,
        add_special_tokens=False if txts[0].startswith("[CLS]") else True,
    ).to(device)
    embeddings = model.bert.embeddings(**tokenized).detach()

    return embeddings.min(dim=0).values.unsqueeze(0), embeddings.max(
        dim=0
    ).values.unsqueeze(0)


def parse_arguments():
    """
    Parse and handle command-line arguments for configuring and running experiments.

    Returns:
        argparse.ArgumentParser: Argument parser object.
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare and configure experimental parameters for running research experiments."
    )

    # Experiment presets
    parser.add_argument(
        "--default",
        action="store_true",
        help="Generate default experiments with predefined settings.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate default test experiments for debugging or small-scale trials.",
    )

    # Experiment configuration
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        help="Name of the experiment for identification (required if --default is not set).",
    )
    parser.add_argument(
        "-a",
        "--algo",
        choices=["simec", "simexp", "both"],
        default="both",
        help="Algorithm to run. Options: 'simec', 'simexp', or 'both'. Defaults to 'both'.",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        help="Number of iterations to run (required if --default is not set).",
    )
    parser.add_argument(
        "-d",
        "--delta_mult",
        type=int,
        default=1,
        help="Delta multiplier for adjustments. Defaults to 1.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1e-2,
        help="Threshold value for experiments. Defaults to 0.01.",
    )
    parser.add_argument(
        "-s",
        "--save_each",
        type=int,
        default=1,
        help="Frequency to save results, e.g., every n iterations. Defaults to 1.",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        type=int,
        help="Number of inputs per experiment (required if --default is not set).",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=1,
        help="Number of repetitions per input. Defaults to 1.",
    )
    parser.add_argument(
        "-p",
        "--patches",
        choices=["all", "one", "q1", "q2", "q3", "target-word"],
        help=(
            "Patch exploration mode (required if --default is not set). Options: 'all', 'one', 'q1', 'q2', 'q3', or "
            "'target-word'. The latter is specific to text-based experiments like Winobias."
        ),
    )
    parser.add_argument(
        "-v",
        "--vocab_tokens",
        type=int,
        choices=range(1, 101),
        default=1,
        help="Percentage of vocabulary tokens (1-100) for text experiments. Required for text-based experiments.",
    )

    # Data and model paths
    parser.add_argument(
        "-od",
        "--orig_data_dir",
        type=str,
        help="Directory containing original input data (required if --default is not set).",
    )
    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        help="Path to the model directory or Huggingface model name (required if --default is not set).",
    )
    parser.add_argument(
        "-ed",
        "--exp_dir",
        type=str,
        help="Directory to save configuration and output data (required if --default is not set).",
    )
    parser.add_argument(
        "-o",
        "--objective",
        choices=["cls", "mlm"],
        default="cls",
        help="Model task: 'cls' (classification) or 'mlm' (masked language model). Defaults to 'cls'.",
    )

    # Device and other options
    parser.add_argument(
        "--device",
        type=str,
        help="Device for computation (e.g., 'cuda', 'cpu'). Auto-detected if unspecified.",
    )
    parser.add_argument(
        "--cap-ex",
        action="store_true",
        default=True,
        help="Enable or disable capped execution. Defaults to enabled.",
    )

    parser.add_argument(
        "--theoretical-min-max",
        action="store_true",
        default=True,
        help="If set, the experiments will use min and max values for capping embeddings taken from the used model. If not set, the experiments will use min and max values computed on experimentd data.",
    )

    return parser, parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    argparser, args = parse_arguments()

    # Determine computation device
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    else:
        args.device = torch.device(args.device).type

    dev = torch.device(args.device)

    if not args.default:
        # Validate mandatory arguments for individual experiments
        missing_args = []
        required_args = [
            ("-ed/--exp_dir", args.exp_dir),
            ("-mp/--model_path", args.model_path),
            ("-od/--orig_data_dir", args.orig_data_dir),
            ("-p/--patches", args.patches),
            ("-i/--inputs", args.inputs),
            ("-n/--iterations", args.iterations),
            ("-e/--exp_name", args.exp_name),
        ]
        for name, value in required_args:
            if value is None:
                missing_args.append(name)

        if missing_args:
            argparser.error(
                f"--default option not selected: preparing individual experiment. The following arguments are required: {', '.join(missing_args)}"
            )

        if (
            not any(k in args.orig_data_dir for k in ["cifar", "mnist"])
            and args.objective is None
        ):
            argparser.error(
                "You have to specify the objective for BERT experiments [mlm or cls]. The following argument is required: --objective."
            )

        if any(k in args.orig_data_dir for k in ["cifar", "mnist"]):
            model_filename = next(
                f for f in os.listdir(args.model_path) if f.endswith(".pt")
            )
            mdl, _ = load_model(
                model_path=os.path.join(args.model_path, model_filename),
                config_path=os.path.join(args.model_path, "config.json"),
                device=dev,
            )
            deactivate_dropout_layers(mdl)
            tkn = None
            if "cifar" in args.orig_data_dir:
                means = CIFAR_MEAN
                sds = CIFAR_STD
            else:
                means = [
                    MNIST_MEAN,
                ]
                sds = [
                    MNIST_STD,
                ]
        else:
            tkn, mdl = load_bert_model(
                args.model_path, mask_or_cls=args.objective, device=dev
            )
            deactivate_dropout_layers(mdl)
            means = [0.0]
            sds = [1.0]

        if not args.theoretical_min_max:
            print("Measuring input space distribution...")
            min_embeddings, max_embeddings = measure_embedding_distribution(
                data_dir=args.orig_data_dir,
                model=mdl,
                objective=args.objective or None,
                tokenizer=tkn,
                device=dev,
            )
        else:
            print("Using theoretical min and max values for capping embeddings.")
            min_embeddings, max_embeddings = compute_embedding_boundaries(
                model=mdl, means=means, sds=sds
            )

        # Prepare experiment directory and configuration
        experiment_dir = os.path.join(args.exp_dir, args.exp_name)
        input_type = generate_experiment(
            input_path=args.orig_data_dir,
            exp_dir=experiment_dir,
            n_inputs=args.inputs,
            patch_option=args.patches,
        )

        # Prepare configuration
        parameters = vars(args)
        if input_type != "txt":
            parameters.pop("vocab_tokens", None)
        else:
            parameters["vocab_tokens"] = [parameters["vocab_tokens"]]

        save_json(os.path.join(experiment_dir, "parameters.json"), parameters)
        save_object(
            min_embeddings.cpu(), os.path.join(experiment_dir, "min_distribution.pkl")
        )
        save_object(
            max_embeddings.cpu(), os.path.join(experiment_dir, "max_distribution.pkl")
        )
    else:
        # Generate default experiments
        print("Generating default experiments...")
        if args.test:
            print("Warning: creating experiments with --test option.")

        # Base template for an experiment configuration
        BASE_EXPERIMENT = {
            "algo": "both",  # Default algorithm to run: both 'simec' and 'simexp'
            "iterations": (
                10 if args.test else 20000
            ),  # Number of iterations, shorter for test mode
            "delta_mult": None,  # Multiplier for delta adjustment
            "threshold": 0.01,  # Default threshold for experiments
            "save_each": 1,  # Frequency of saving results after iterations
            "inputs": None,  # Number of inputs per experiment (to be defined later)
            "repeat": None,  # Number of times to repeat experiments (to be determined dynamically)
            "patches": None,  # Patch exploration options (set dynamically)
            "objective": "cls",  # Default model objective: classification
            "exploratin_capping": args.cap_ex,  # If True, embeddings are capped during exploration, otherwise they are capped at interpretation
            "theoretical_min_max": args.theoretical_min_max,  # If True, the distribution used to cap the embeddings are taken from the model's theoretical min and max. Otherwise, min and max are computed on the experiment inputs
        }

        # Define specific experiments with fixed configurations
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
                "vocab_tokens": [1, 5, 10],  # Vocab % for eval
            },
            "winobias": {
                "exp_name": "winobias",
                "orig_data_dir": "../data/wino_bias_txts",
                "model_path": "bert-base-uncased",
                "patches": "target-word",  # Target specific patches for tokenized target words
                "objective": "mlm",  # Masked language model objective
                "vocab_tokens": [1, 5, 10],  # Vocab % for eval
            },
        }

        # Define parameter variations to iterate through
        DELTA_MULT_VALUES = [1, 5]
        INPUT_VALUES = [10, 2] if args.test else [200, 20]
        PATCH_OPTIONS = ["all", "one", "q1", "q2", "q3"]

        # Iterate through predefined experiments and generate configurations
        for exp_name, exp_overrides in EXPERIMENTS.items():
            # Create a copy of the base experiment and override with specific settings
            base_experiment = copy.deepcopy(BASE_EXPERIMENT)
            base_experiment.update(exp_overrides)

            if any(k in base_experiment["exp_name"] for k in ["cifar", "mnist"]):
                model_filename = next(
                    f
                    for f in os.listdir(base_experiment["model_path"])
                    if f.endswith(".pt")
                )
                mdl, _ = load_model(
                    model_path=os.path.join(
                        base_experiment["model_path"], model_filename
                    ),
                    config_path=os.path.join(
                        base_experiment["model_path"], "config.json"
                    ),
                    device=dev,
                )
                deactivate_dropout_layers(mdl)
                tkn = None
                if "cifar" == base_experiment["exp_name"]:
                    means = CIFAR_MEAN
                    sds = CIFAR_STD
                else:
                    means = [
                        MNIST_MEAN,
                    ]
                    sds = [
                        MNIST_STD,
                    ]
            else:
                tkn, mdl = load_bert_model(
                    base_experiment["model_path"],
                    mask_or_cls=base_experiment["objective"],
                    device=dev,
                )
                deactivate_dropout_layers(mdl)
                means = [0.0]
                sds = [1.0]

            if not args.theoretical_min_max:
                print("Measuring input space distribution...")
                min_embeddings, max_embeddings = measure_embedding_distribution(
                    data_dir=args.orig_data_dir,
                    model=mdl,
                    objective=args.objective or None,
                    tokenizer=tkn,
                    device=dev,
                )
            else:
                print("Using theoretical min and max values for capping embeddings.")
                min_embeddings, max_embeddings = compute_embedding_boundaries(
                    model=mdl, means=means, sds=sds
                )

            # Iterate over input configurations
            for n_input in INPUT_VALUES:
                base_experiment["inputs"] = n_input
                # Define repetition based on input size and test mode
                base_experiment["repeat"] = (
                    1 if n_input in [200, 10] else (5 if args.test else 10)
                )
                # Initialize variable for fixed sample tracking (test mode)
                sample = None

                # Iterate over delta multiplier values
                for delta in DELTA_MULT_VALUES:
                    base_experiment["delta_mult"] = delta

                    # Define patch exploration options (or use specific ones for targeted experiments)
                    patch_list = (
                        [base_experiment["patches"]]
                        if base_experiment["patches"] == "target-word"
                        else PATCH_OPTIONS
                    )

                    # Iterate over patch options
                    for patch_opt in patch_list:
                        base_experiment["patches"] = patch_opt

                        # Construct a descriptive name for the experiment
                        exp_name_full = f"{base_experiment['exp_name']}-{delta}-{n_input}-{patch_opt}"
                        if args.test:
                            exp_name_full = "test/" + exp_name_full

                        experiment_dir = os.path.join("experiments_data", exp_name_full)

                        # Generate experiment inputs and save parameters
                        if args.test:
                            # Ensure at least 20% of inputs remain fixed for test comparisons
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

                        # Filter out parameters with None values for saving
                        filtered_experiment = {
                            k: v for k, v in base_experiment.items() if v is not None
                        }
                        save_json(
                            os.path.join(experiment_dir, "parameters.json"),
                            filtered_experiment,
                        )
                        save_object(
                            obj=min_embeddings.cpu(),
                            filename=os.path.join(
                                experiment_dir, "min_distribution.pkl"
                            ),
                        )
                        save_object(
                            obj=max_embeddings.cpu(),
                            filename=os.path.join(
                                experiment_dir, "max_distribution.pkl"
                            ),
                        )
        print("Done")
