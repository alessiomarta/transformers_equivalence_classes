import argparse
import os
import copy
import random
import shutil
import torch
from transformers import logging
import sys
sys.path.append("./")

from experiments_utils import (
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
from models.const import *


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
        if not all_config_data[input_name.split(".")[0]]:
            print(f"Warning: No config found for {input_name}")
        combined_config[input_name] = all_config_data.get(input_name.split(".")[0], {})

    combined_config_path = os.path.join(exp_dir, "config.json")
    save_json(combined_config_path, combined_config)

    file_type = os.path.splitext(all_sampled_files[0])[1][1:]
    inspected_files = random.sample(
        all_sampled_files, max(1, len(all_sampled_files) // 5)
    )

    print(f"Created experiment in {exp_dir} with {n_inputs} files.")
    return file_type, inspected_files


def generate_experiment_combinations(
    exp,
    device,
    delta_mult_values=None,
    patch_options=None,
    input_values=None,
    test=None,
):
    if patch_options is None:
        patch_options = exp["patches"]
    if delta_mult_values is None:
        delta_mult_values = exp["delta_mult"]
    if input_values is None:
        input_values = exp["inputs"]
    # loading model
    if any(k in exp["exp_name"] for k in ["cifar", "mnist"]):
        
        model_filename = next(
            f for f in os.listdir(exp["model_path"]) if f.endswith("final.pt")
        )
        mdl, _ = load_model(
            model_path=os.path.join(exp["model_path"], model_filename),
            config_path=os.path.join(exp["model_path"], "config.json"),
            device=device,
        )
        deactivate_dropout_layers(mdl)
        tkn = None
        # Give default values to means ans sds if normalize is set to False
        if not normalize:
            CIFAR_MEAN = [0.0, 0.0, 0.0]
            MNIST_MEAN = 0.0
            CIFAR_STD = [1.0, 1.0, 1.0]
            MNIST_STD = 1.0

        if "cifar" in exp["exp_name"]:
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
            exp["model_path"],
            mask_or_cls=exp["objective"],
            device=device,
        )
        deactivate_dropout_layers(mdl)
        means = [0.0]
        sds = [1.0]

    # computing min and max embeddings
    if not exp["theoretical_min_max"]:
        print("Measuring input space distribution...")
        min_embeddings, max_embeddings = measure_embedding_distribution(
            data_dir=exp["orig_data_dir"],
            model=mdl,
            objective=exp["objective"] or None,
            tokenizer=tkn,
            device=device,
        )
    else:
        print("Using theoretical min and max values for capping embeddings.")
        min_embeddings, max_embeddings = compute_embedding_boundaries(
            model=mdl, means=means, sds=sds
        )

    # Initialize variable for fixed sample tracking (test mode)
    sample = None

    # Iterate over delta multiplier values
    for delta in delta_mult_values:
        exp["delta_mult"] = delta

        # Iterate over patch options
        if isinstance(patch_options, str):
            patch_options = [patch_options]
        for patch_opt in patch_options:
            exp["patches"] = patch_opt

            # Construct a descriptive name for the experiment
            exp_name_full = f"{exp['exp_name']}-{str(delta).replace('.','p')}-{input_values}-{patch_opt}"
            if test:
                exp_name_full = "test/" + exp_name_full

            if "exp_dir" in exp:
                experiment_dir = os.path.join(exp["exp_dir"], exp_name_full)
            else:
                experiment_dir = os.path.join("experiments_data", exp_name_full)

            # Generate experiment inputs and save parameters
            if test or not exp["sample_images"]:
                # Ensure at least 20% of inputs remain fixed for test comparisons
                if sample:
                    generate_experiment(
                        input_path=exp["orig_data_dir"],
                        exp_dir=experiment_dir,
                        n_inputs=input_values,
                        patch_option=patch_opt,
                        fixed_inputs=sample,
                    )
                else:
                    _, sample = generate_experiment(
                        input_path=exp["orig_data_dir"],
                        exp_dir=experiment_dir,
                        n_inputs=input_values,
                        patch_option=patch_opt,
                    )
            else:
                generate_experiment(
                    input_path=exp["orig_data_dir"],
                    exp_dir=experiment_dir,
                    n_inputs=input_values,
                    patch_option=patch_opt,
                )

            # Filter out parameters with None values for saving
            filtered_experiment = {k: v for k, v in exp.items() if v is not None}
            save_json(
                os.path.join(experiment_dir, "parameters.json"),
                filtered_experiment,
            )
            save_object(
                obj=min_embeddings.cpu(),
                filename=os.path.join(experiment_dir, "min_distribution.pkl"),
            )
            save_object(
                obj=max_embeddings.cpu(),
                filename=os.path.join(experiment_dir, "max_distribution.pkl"),
            )


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

    parser.add_argument(
        "--default",
        action="store_true",
        help="Generate default experiments with predefined settings, without the interactive environment.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device for computation (e.g., 'cuda', 'cpu'). Auto-detected if unspecified.",
    )

    return parser, parser.parse_args()


def get_user_input(prompt, default=None, choices=None, multiple=False):
    """
    Interactive function to get user input with a default option and multiple-choice support.

    Args:
        prompt (str): The question to ask the user.
        default (any, optional): Default value if the user presses Enter.
        choices (list, optional): List of allowed choices.
        multiple (bool, optional): Whether to allow multiple values (comma-separated).

    Returns:
        str or list: User input as a string or list of choices.
    """
    while True:
        user_input = input(
            f"{prompt} {'(default: ' + str(default) + ')' if default is not None else ''}: "
        ).strip()

        if not user_input and default is not None:
            return default

        if multiple:
            # Handle case where input is either a single value or a comma-separated list
            user_choices = [item.strip() for item in user_input.split(",")]

            # If there's only one input value, treat it as a list of one
            if len(user_choices) == 1:
                return user_choices[0]

            if choices and not all(choice in choices for choice in user_choices):
                print(f"Invalid choices! Available options: {choices}")
                continue

            return user_choices

        if choices and user_input not in choices:
            print(f"Invalid choice! Available options: {choices}")
            continue

        return user_input


def interactive_argument_parser():
    """
    Interactive argument input instead of command-line parsing.

    Returns:
        dict: Parsed arguments as a dictionary.
    """
    args = {}

    print("\nWelcome to the interactive experiment setup!")
    print("Press Enter to use the default value where available.")

    args["exp_name"] = get_user_input("Experiment name")
    args["algo"] = get_user_input(
        "Algorithm(s)",
        default=["both"],
        choices=["simec", "simexp", "both"],
        multiple=True,
    )
    args["iterations"] = int(get_user_input("Number of iterations", default=1000))

    # Fixing the issue for 'multiple=True' for delta_mult
    delta_mult_input = get_user_input("Delta multiplier", default=[1,10], multiple=True)
    if isinstance(delta_mult_input, list):
        args["delta_mult"] = [float(x) for x in delta_mult_input]
    else:
        args["delta_mult"] = [float(delta_mult_input)]  # Single value, but in a list

    args["threshold"] = float(get_user_input("Threshold value", default= 1e-6))
    args["save_each"] = int(
        get_user_input("Save results every n iterations", default=10)
    )
    args["inputs"] = int(get_user_input("Number of inputs per experiment", default=16))
    args["repeat"] = int(get_user_input("Number of repetitions", default=3))

    args["patches"] = get_user_input(
        "Patch mode(s)",
        default=["all"],
        choices=["all", "one", "q1", "q2", "q3", "target-word"],
        multiple=True,
    )

    args["orig_data_dir"] = get_user_input("Original data directory")
    args["model_path"] = get_user_input("Model path")
    args["exp_dir"] = get_user_input("Experiment directory")
    args["objective"] = get_user_input(
        "Objective", default="cls", choices=["cls", "mlm"]
    )

    args["cap_ex"] = (
        get_user_input(
            "Enable capped execution? (yes/no)", default="yes", choices=["yes", "no"]
        )
        == "yes"
    )

    args["degrowth"] = (
        get_user_input(
            "Check eigenvector is opposite to the gradient? (yes/no)", default="no", choices=["yes", "no"]
        )
        == "yes"
    )

    args["sample_images"] = (
        get_user_input(
            "Sample 100% new images? (yes/no)", default="no", choices=["yes", "no"]
        )
        == "yes"
    )

    args["theoretical_min_max"] = (
        get_user_input(
            "Use theoretical min/max? (yes/no)", default="yes", choices=["yes", "no"]
        )
        == "yes"
    )

    return args


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
        args = interactive_argument_parser()
        generate_experiment_combinations(args, dev)
    else:
        # Generate default experiments
        print("Generating default experiments...")
        if args.test:
            print("Warning: creating experiments with --test option.")

        # Base template for an experiment configuration
        BASE_EXPERIMENT = {
            "algo": "both",  # Default algorithm to run: both 'simec' and 'simexp'
            "iterations": (
                10 if args.test else 1000
            ),  # Number of iterations, shorter for test mode
            "delta_mult": None,  # Multiplier for delta adjustment
            "threshold": 0.01,  # Default threshold for experiments
            "save_each": 10,  # Frequency of saving results after iterations
            "inputs": None,  # Number of inputs per experiment (to be defined later)
            "repeat": 3,  # Number of times to repeat experiments (to be determined dynamically)
            "patches": None,  # Patch exploration options (set dynamically)
            "objective": "cls",  # Default model objective: classification
            "exploration_capping": True,  # If True, embeddings are capped during exploration, otherwise they are capped at interpretation
            "theoretical_min_max": True,  # If True, the distribution used to cap the embeddings are taken from the model's theoretical min and max. Otherwise, min and max are computed on the experiment inputs
            "sample_images": False,
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
                "vocab_tokens": [5],  # Vocab % for eval
            },
            "winobias": {
                "exp_name": "winobias",
                "orig_data_dir": "../data/wino_bias_txts",
                "model_path": "bert-base-uncased",
                "patches": "target-word",  # Target specific patches for tokenized target words
                "objective": "mlm",  # Masked language model objective
                "vocab_tokens": [5],  # Vocab % for eval
            },
        }

        # Define parameter variations to iterate through
        DELTA_MULT_VALUES = [1, 10]
        INPUT_VALUES = 16
        PATCH_OPTIONS = ["all", "one", "q2"]

        # Iterate through predefined experiments and generate configurations
        for exp_name, exp_overrides in EXPERIMENTS.items():
            # Create a copy of the base experiment and override with specific settings
            base_experiment = copy.deepcopy(BASE_EXPERIMENT)
            base_experiment.update(exp_overrides)

            generate_experiment_combinations(
                base_experiment,
                dev,
                DELTA_MULT_VALUES,
                PATCH_OPTIONS,
                INPUT_VALUES,
                args.test,
            )

        print("Done")
