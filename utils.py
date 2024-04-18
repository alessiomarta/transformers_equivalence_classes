import json
import torch
from torchvision import datasets, transforms
from vit import ViTForClassfication


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
    model = ViTForClassfication(config)
    model.load_state_dict(checkpoint)
    return model, config


def deactivate_dropout_layers(model):
    """Deactivate the dropout layers of the model after training."""
    model.embedding.dropout.p = 0.0
    for block in model.encoder.blocks:
        block.attention.attn_dropout.p = 0.0
        block.attention.output_dropout.p = 0.0
        block.mlp.dropout.p = 0.0
