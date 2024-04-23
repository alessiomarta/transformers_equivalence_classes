import os
import json
import string
from PIL import Image
import torch
from torchvision import datasets, transforms
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizerFast,
)

import pickle
from models.vit import ViTForClassification


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as outp:
        obj = pickle.load(outp)
    return obj


def load_raw_images(img_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    images = []
    images_names = []
    for filename in os.listdir(img_dir):
        if os.path.isfile(
            os.path.join(img_dir, filename)
        ) and filename.lower().endswith(image_extensions):
            image = Image.open(os.path.join(img_dir, filename)).convert("L")
            if image.size != (28, 28):
                image = image.resize((28, 28))
            images.append(transform(image))
            images_names.append(filename.split(".")[0])
    return torch.stack(images), images_names


def load_raw_sents(txt_dir):
    txts = []
    txts_names = []
    for filename in os.listdir(txt_dir):
        if os.path.isfile(
            os.path.join(txt_dir, filename)
        ) and filename.lower().endswith(".txt"):
            with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as file:
                txts.append(file.readlines()[0])
            txts_names.append(filename.split(".")[0])
    return txts, txts_names


def prepare_data(
    out_dir=".data/MNIST/",
    batch_size=128,
    num_workers=2,
    test=True,
    train_sample_size=None,
    test_sample_size=None,
):

    classes = tuple(range(10))

    mean, std = (0.5,), (0.5,)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    trainset = datasets.MNIST(out_dir, download=True, train=True, transform=transform)

    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    if test:
        testset = datasets.MNIST(
            out_dir, download=True, train=False, transform=transform
        )

        if test_sample_size is not None:
            # Randomly sample a subset of the test set
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return trainloader, testloader, classes

    return trainloader, classes


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_model(model_path, config_path, device):
    if device.type == "cpu":
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    elif device.type == "mps":
        checkpoint = torch.load(model_path, map_location=torch.device("mps"))
    else:
        checkpoint = torch.load(model_path)
    config = load_config(config_path)
    model = ViTForClassification(config)
    model.load_state_dict(checkpoint)
    return model, config


def deactivate_dropout_layers(model):
    """Deactivate the dropout layers of the model after training."""
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


def load_bert_model(model_name, mask_or_cls):
    """Load pre-trained model (either bert-base or bert-mini)

    Args:
        model_name: Either "bert-base" (768-d embedding space) or "bert-mini" (256-d embedding space).

    Returns:
        The loaded model.
    """
    if mask_or_cls == "mask":
        bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
        bert_model = BertForMaskedLM.from_pretrained(model_name)
        return bert_tokenizer, bert_model
    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_name)
    bert_model.decoder = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return bert_tokenizer, bert_model


# Helper function to identify punctuation
def is_punctuation(token):
    return all(char in string.punctuation for char in token)


def get_allowed_tokens(tokenizer):
    # Identify punctuation and special tokens
    return [
        idx
        for token, idx in tokenizer.vocab.items()
        if not (
            is_punctuation(token)
            or token in tokenizer.all_special_tokens
            or token.startswith("[unused")
            or token.startswith("##")
            or token.isdigit()
        )
    ]
