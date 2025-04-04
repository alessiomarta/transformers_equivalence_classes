"""
This script trains a Vision Transformer (ViT) model for MNIST or CIFAR-10 datasets. It includes:
1. Loading configurations from an external JSON file.
2. Preprocessing images using Hugging Face's ViTImageProcessorFast.
3. Training and evaluating a ViT model.
4. Saving checkpoints, metrics, and the final model.
"""

import json
import os
import argparse
import torch
from torchvision import datasets, transforms
from tqdm.auto import trange
import sys
sys.path.append(".")
from time import sleep
from vit import ViTForClassification
from const import *
sleep(2)

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_experiment(
    experiment_name, config, model, train_losses, test_losses, accuracies, base_dir
):
    """
    Save experiment data, including configuration, metrics, and model checkpoints.

    Args:
        experiment_name (str): Name of the experiment directory.
        config (dict): Experiment configuration.
        model (torch.nn.Module): Trained model.
        train_losses (list[float]): Training losses for each epoch.
        test_losses (list[float]): Test losses for each epoch.
        accuracies (list[float]): Test accuracies for each epoch.
        base_dir (str): Base directory to save the experiment.
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    configfile = os.path.join(outdir, "config.json")
    with open(configfile, "w", encoding="utf-8") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    metrics_file = os.path.join(outdir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        data = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "accuracies": accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir):
    """
    Save a model checkpoint.

    Args:
        experiment_name (str): Name of the experiment directory.
        model (torch.nn.Module): Trained model.
        epoch (int): Epoch number for checkpoint naming.
        base_dir (str): Base directory to save the checkpoint.
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    checkpoint_path = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)


class Trainer:
    """
    Trainer class to handle training and evaluation of the model.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device, config):
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            loss_fn (torch.nn.Module): Loss function.
            exp_name (str): Experiment name.
            device (str): Device to run the computation on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.config = config

    def train(
        self, trainloader, testloader, epochs, save_model_every_n_epochs, base_dir
    ):
        """
        Train the model and save experiment results.

        Args:
            trainloader (torch.utils.data.DataLoader): DataLoader for training data.
            testloader (torch.utils.data.DataLoader): DataLoader for test data.
            epochs (int): Number of training epochs.
            save_model_every_n_epochs (int): Save a checkpoint every N epochs.
            base_dir (str): Base directory for saving experiments.
        """
        train_losses, test_losses, accuracies = [], [], []

        for epoch in trange(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

            if (
                save_model_every_n_epochs > 0
                and (epoch + 1) % save_model_every_n_epochs == 0
            ):
                save_checkpoint(self.exp_name, self.model, epoch + 1, base_dir)

        save_experiment(
            self.exp_name,
            self.config,
            self.model,
            train_losses,
            test_losses,
            accuracies,
            base_dir,
        )

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.

        Args:
            trainloader (torch.utils.data.DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(images)[0], labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        """
        Evaluate the model on the test set.

        Args:
            testloader (torch.utils.data.DataLoader): DataLoader for test data.

        Returns:
            tuple: Accuracy and average test loss.
        """
        self.model.eval()
        total_loss = 0
        correct = 0

        for images, labels in testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits, _ = self.model(images)
            total_loss += self.loss_fn(logits, labels).item() * len(images)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()

        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def prepare_data(
    batch_size, dataset, base_dir, train_sample_size=None, test_sample_size=None
):
    """
    Prepare data loaders for training and testing.

    Args:
        batch_size (int): Batch size for DataLoaders.
        dataset (str): Name of the dataset ('MNIST' or 'CIFAR10').
        device (str): Device to perform operations ('cpu' or 'cuda').
        train_sample_size (int, optional): Number of training samples to use.
        test_sample_size (int, optional): Number of test samples to use.

    Returns:
        tuple: Data loaders for training and testing.
    """
    if normalize:
        if "cifar" in dataset.lower():
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
                [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
            )
    else:
        transform = transforms.ToTensor()

    data_dir = f"{base_dir}/{dataset.lower()}_data"
    train_dataset = getattr(datasets, dataset.upper())(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = getattr(datasets, dataset.upper())(
        root=data_dir, train=False, download=True, transform=transform
    )

    if train_sample_size:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_sample_size))
    if test_sample_size:
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_sample_size))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ViT model on MNIST or CIFAR-10."
    )

    # Command-line arguments
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MNIST", "CIFAR10"],
        required=True,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for DataLoader."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="Save a checkpoint every N epochs."
    )
    parser.add_argument(
        "--train_sample_size",
        type=int,
        default=None,
        help="Number of training samples to use.",
    )
    parser.add_argument(
        "--test_sample_size",
        type=int,
        default=None,
        help="Number of test samples to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to train on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name for saving results.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./experiments",
        help="Base directory to save experiments.",
    )

    return parser.parse_args()


def main():
    """
    Main function to parse arguments, load configuration, prepare data, and train the ViT model.
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:", config)

    # Prepare dataset and data loaders
    train_loader, test_loader = prepare_data(
        batch_size=args.batch_size,
        dataset=args.dataset,
        base_dir=args.base_dir + "/data",
        train_sample_size=args.train_sample_size,
        test_sample_size=args.test_sample_size,
    )
    print(
        f"Data prepared. Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}"
    )

    # Initialize the model, optimizer, and loss function
    model = ViTForClassification(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        exp_name=args.exp_name,
        device=args.device,
        config=config,
    )
    trainer.train(
        trainloader=train_loader,
        testloader=test_loader,
        epochs=args.epochs,
        save_model_every_n_epochs=args.save_every,
        base_dir=args.base_dir + "/models",
    )
    print("Training complete!")


if __name__ == "__main__":
    main()
