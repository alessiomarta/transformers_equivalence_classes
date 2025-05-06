"""
A utility module for handling various operations needed in machine learning experiments.
This includes functions for saving and loading objects, loading and preprocessing data, 
configuring models, and manipulating model training states.
"""

import os
import json
import string
from typing import Any, List, Tuple
import pickle
from PIL import Image
import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizerFast,
)
from numpy import load

import sys
sys.path.append(".")
from models.vit import ViTForClassification
from models.const import *

def load_metadata_tensors(filename):
    with open(filename + "/metadata.json", "r") as f:  # Overwrites any existing file.
        metadata = json.load(f)
    numpy_tensors = load(filename + "/embeddings.npz")
    tensor_dict = {k:torch.from_numpy(numpy_tensors[k]) for k in numpy_tensors.keys()}
    return metadata | tensor_dict

def collect_pkl_res_files(exploration_result_dir: str) -> list:
    """
    Collects paths to all `.pkl` files in a given exploration result directory.

    Parameters:
    ----------
    exploration_result_dir : str
        The root directory to search for `.pkl` files.

    Returns:
    --------
    list
        A list of paths to `.pkl` files found in the directory tree, without considering min and max distribution files.
    """
    pkl_paths = []
    for root, _, files in os.walk(exploration_result_dir):
        # Filter and collect only `.pkl` files
        pkl_paths.extend(
            os.path.join(root, f)
            for f in files
            if f.lower().endswith(".pkl")
            and "distribution" not in f
            and "interpretation" not in root
        )
    return pkl_paths


def save_object(obj: Any, filename: str) -> None:
    """
    Save a Python object to a file using pickle.

    Args:
        obj: The Python object to save.
        filename: The path to the file where the object will be saved.
    """
    with open(filename, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename: str) -> Any:
    """
    Load a Python object from a file using pickle.

    Args:
        filename: The path to the file from which the object will be loaded.

    Returns:
        The Python object loaded from the file.
    """
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


def save_json(filename: str, object_to_save: Any) -> dict:
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(object_to_save, file)


def load_raw_images(img_dir: str) -> Tuple[torch.Tensor, List[str]]:
    """
    Load images from a directory, convert them to grayscale, resize to 28x28, and apply a standard transformation.

    Args:
        img_dir: The directory from which images are loaded.

    Returns:
        A tuple containing a batch of tensor images and their corresponding names.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    images = []
    images_names = []
    for filename in os.listdir(img_dir):
        if os.path.isfile(
            os.path.join(img_dir, filename)
        ) and filename.lower().endswith(image_extensions):
            if "mnist" in img_dir:
                image = read_image(
                    os.path.join(img_dir, filename), mode=ImageReadMode.GRAY
                ).to(torch.float32)
            else:
                image = read_image(
                    os.path.join(img_dir, filename), mode=ImageReadMode.RGB
                ).to(torch.float32)
            images.append(image)
            images_names.append(filename)
    return torch.stack(images), images_names


def load_and_transform_raw_images(img_dir: str) -> Tuple[torch.Tensor, List[str]]:
    """
    Load images from a directory, convert them to grayscale, resize to 28x28, and apply a standard transformation.

    Args:
        img_dir: The directory from which images are loaded.

    Returns:
        A tuple containing a batch of tensor images and their corresponding names.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    images = []
    images_names = []
    for filename in os.listdir(img_dir):
        if os.path.isfile(
            os.path.join(img_dir, filename)
        ) and filename.lower().endswith(image_extensions):
            image = load_and_transform_raw_image(os.path.join(img_dir, filename))
            images.append(image)
            images_names.append(filename)
    return torch.stack(images), images_names


def load_and_transform_raw_image(img_path: str) -> torch.Tensor:
    """
    Load a single image from a file, convert it to grayscale if necessary, and apply a standard transformation.

    Args:
        img_path: The path to the image file.

    Returns:
        A tensor containing the transformed image.
    """
    if normalize:
        if "cifar" in img_path.lower():
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=CIFAR_MEAN,
                        std=CIFAR_STD,
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
            )
    else:
        transform = transforms.ToTensor()

    if os.path.isfile(img_path):
        image = Image.open(img_path).convert("L" if "mnist" in img_path.lower() else "RGB")
        image = transform(image).to(torch.float32)
        return image
    else:
        raise FileNotFoundError(f"No such file: '{img_path}'")


def load_raw_image(img_dir: str, image_filename: str) -> Tuple[torch.Tensor, List[str]]:
    """
    Load a single image from a specific image file within a directory

    Args:
        img_dir: The directory from which images are loaded.

    Returns:
        A tuple containing a batch of tensor images and their corresponding names.
    """
    if os.path.isfile(os.path.join(img_dir, image_filename + ".jpg")):
        image = read_image(os.path.join(img_dir, image_filename)).to(torch.float32)
        return image, image_filename.split(".")[0]
    if os.path.isfile(os.path.join(img_dir, image_filename + ".png")):
        image = read_image(os.path.join(img_dir, image_filename)).to(torch.float32)
        return image, image_filename.split(".")[0]


def load_raw_sents(txt_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load sentences from text files within a specified directory.

    Args:
        txt_dir: The directory from which text files are loaded.

    Returns:
        A tuple containing lists of sentences and their corresponding filenames without extension.
    """
    txts = []
    txts_names = []
    for filename in os.listdir(txt_dir):
        if os.path.isfile(
            os.path.join(txt_dir, filename)
        ) and filename.lower().endswith(".txt"):
            with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as file:
                txts.append(file.readlines()[0])
            txts_names.append(filename)
    return txts, txts_names


def load_raw_sent(txt_dir: str, sentence_filename: str) -> Tuple[str, str]:
    """
    Load a single sentence from a specific text file within a directory.

    Args:
        txt_dir: The directory containing the text file.
        sentence_filename: The name of the text file (without the extension).

    Returns:
        A tuple containing the sentence and the filename without extension.
    """
    with open(
        os.path.join(txt_dir, sentence_filename + ".txt"), "r", encoding="utf-8"
    ) as file:
        txt = file.readlines()[0]
    txt_name = sentence_filename.split(".")[0]
    return txt, txt_name


def load_config(config_path: str) -> dict:
    """
    Load a configuration file in JSON format.

    Args:
        config_path: The file path of the configuration file.

    Returns:
        The configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_model(
    model_path: str, config_path: str, device: torch.device
) -> Tuple[ViTForClassification, dict]:
    """
    Load a ViT classification model with a specified configuration, adapted for a specific computing device.

    Args:
        model_path: The file path to the model's state dictionary.
        config_path: The configuration file path for initializing the model.
        device: The torch device on which the model will operate.

    Returns:
        A tuple containing the loaded model and its configuration.
    """
    if device.type == "cpu":
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(model_path)
    config = load_config(config_path)
    model = ViTForClassification(config)
    if len(checkpoint.keys()) > len(model.state_dict().keys()):
        checkpoint = {k: checkpoint[k] for k, v in model.state_dict().items()}
    model.load_state_dict(checkpoint)
    return model.to(device), config


def deactivate_dropout_layers(model: torch.nn.Module) -> None:
    """
    Deactivate dropout layers in a model to potentially improve its performance during inference.

    Args:
        model: The model from which dropout layers will be deactivated.
    """
    if isinstance(model, ViTForClassification):
        model.embedding.dropout.p = 0.0
        for block in model.encoder.blocks:
            block.attention.attn_dropout.p = 0.0
            block.attention.output_dropout.p = 0.0
            block.mlp.dropout.p = 0.0
    elif isinstance(model, BertForSequenceClassification) or isinstance(
        model, BertForMaskedLM
    ):
        model.bert.embeddings.dropout.p = 0.0
        if hasattr(model, "dropout"):
            model.dropout.p = 0.0
        for layer in model.bert.encoder.layer:
            layer.attention.self.dropout.p = 0.0
            layer.attention.output.dropout.p = 0.0
            layer.output.dropout.p = 0.0
            if hasattr(layer, "crossattention"):
                layer.crossattention.self.dropout.p = 0.0
                layer.crossattention.output.dropout.p = 0.0


def load_bert_model(
    model_name: str, mask_or_cls: str, device: torch.device
) -> Tuple[BertTokenizerFast, BertForSequenceClassification]:
    """
    Load a BERT model for either masked language modeling or sequence classification based on the model name.

    Args:
        model_name: The name of the BERT model variant to load.
        mask_or_cls: A string indicating whether to load a model for "mask" (masked LM) or "cls" (classification).

    Returns:
        A tuple of a tokenizer and the loaded BERT model.
    """
    if mask_or_cls in ["mask", "mlm"]:
        bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
        bert_model = BertForMaskedLM.from_pretrained(model_name).to(device)
        return bert_tokenizer, bert_model
    
    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_name).to(device)
    decoder = BertForMaskedLM.from_pretrained("bert-base-uncased")
    deactivate_dropout_layers(decoder)
    decoder = decoder.to(device)
    #bert_model.decoder = lambda x: decoder.cls(decoder.bert.encoder(x)[0])
    bert_model.decoder = decoder.to(device)
    
    return bert_tokenizer, bert_model


def is_punctuation(token: str) -> bool:
    """
    Check if a token consists solely of punctuation characters.

    Args:
        token: The string token to check.

    Returns:
        True if the token consists only of punctuation, otherwise False.
    """
    return all(char in string.punctuation for char in token)


def get_allowed_tokens(tokenizer: BertTokenizerFast) -> List[int]:
    """
    Retrieve a list of token IDs that are not punctuation, special, or numeric tokens from a tokenizer's vocabulary.

    Args:
        tokenizer: The tokenizer whose vocabulary is to be filtered.

    Returns:
        A list of allowable token IDs.
    """
    return [
        idx
        for token, idx in tokenizer.vocab.items()
        if not (
            is_punctuation(token)
            or token in tokenizer.all_special_tokens
            or token.startswith("[unused")
            # or token.startswith("##")
            # or token.isdigit()
        )
    ]


def compute_embedding_boundaries(
    model: torch.nn.Module, means: List[float] = [0.0], sds: List[float] = [1.0]
):
    """Computes the embedding minima and maxima for a given model.

    Args:
        model (torch.nn.Module): Model.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: One-dimensional vectors of theoretical embedding minima and maxima.
    """

    if isinstance(model, ViTForClassification):
        if hasattr(model.embedding, "patch_embeddings"):
            position_embedding = model.embedding.position_embeddings
            # shape = (1, n_patch**2+1, hidden_size)
            token_embedding = model.embedding.patch_embeddings.weight.permute(1, 0)
            # shape = (n_channels*patch_width*patch_height, hidden_size)
        else:
            raise AttributeError(
                "Model .embeddings layer hasn't patch_embeddings attribute."
            )
    else:
        if hasattr(model.bert.embeddings, "word_embeddings"):
            position_embedding = model.bert.embeddings.position_embeddings.weight
            # shape = (max_length, hidden_size)
            token_embedding = model.bert.embeddings.word_embeddings.weight
            # shape = (vocab_size, hidden_size)
        else:
            raise AttributeError(
                "Model .embeddings layer hasn't word_embeddings attribute."
            )

    if position_embedding.dim() > 2:
        position_embedding = position_embedding.squeeze(0)

    max_pos_embedding = position_embedding.max(dim=0).values
    min_pos_embedding = position_embedding.min(dim=0).values

    positive_embeddings = torch.where(token_embedding >= 0, token_embedding, 0)
    negative_embeddings = torch.where(token_embedding < 0, token_embedding, 0)

    max_input = (255 - min(means)) / min(sds)
    min_input = (0 - max(means)) / min(sds)

    # The embedding layer computes e_i = \sum_j E_{ij}*x_j + p^{(k)}_i, where E is the embedding matrix, x_j are the features of the input, and p_i are those of the positional encoding
    # --> e_i <= \sum_{j: E_{ij} >= 0} E_{ij}*max_l(x_l) + \sum_{j: E_{ij} < 0} E_{ij}*min_l(x_l) + max_k (p^{(k)}_i)
    # --> e_i >= \sum_{j: E_{ij} >= 0} E_{ij}*min_l(x_l) + \sum_{j: E_{ij} < 0} E_{ij}*max_l(x_l) + min_k (p^{(k)}_i)
    max_embeddings = (
        max_input * positive_embeddings.sum(dim=0)
        + min_input * negative_embeddings.sum(dim=0)
        + max_pos_embedding
    )
    min_embeddings = (
        min_input * positive_embeddings.sum(dim=0)
        + max_input * negative_embeddings.sum(dim=0)
        + min_pos_embedding
    )

    return min_embeddings, max_embeddings
