import time, json, os, argparse
import torch
from torchvision import datasets, transforms
from vit import ViTForClassification


CONFIG = {
    "patch_size": 2,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.01,
    "attention_probs_dropout_prob": 0.01,
    "initializer_range": 0.02,
    "image_size": 28,
    "num_classes": 10,
    "num_channels": 1,
    "qkv_bias": True,
}


def save_experiment(
    experiment_name,
    config,
    model,
    train_losses,
    test_losses,
    accuracies,
    base_dir="../../mnist_experiments",
):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    configfile = os.path.join(outdir, "config.json")
    with open(configfile, "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = os.path.join(outdir, "metrics.json")
    with open(jsonfile, "w") as f:
        data = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "accuracies": accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(
    experiment_name, model, epoch, base_dir="../../mnist_experiments"
):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(
        self,
        trainloader,
        testloader,
        epochs,
        save_model_every_n_epochs=0,
        measure_time=False,
        base_dir="../../mnist_experiments",
    ):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            if measure_time:
                start_time = time.time()
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print("---------------------------------------------------------")
            print(
                f"Epoch: {i+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            if measure_time:
                print(f"Epoch ended in {(time.time() - start_time)} seconds")
            if (
                save_model_every_n_epochs > 0
                and (i + 1) % save_model_every_n_epochs == 0
                and i + 1 != epochs
            ):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1, base_dir)
            print("---------------------------------------------------------")
        # Save the experiment
        save_experiment(
            self.exp_name,
            CONFIG,
            self.model,
            train_losses,
            test_losses,
            accuracies,
            base_dir=base_dir,
        )

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def prepare_data(
    batch_size=128,
    num_workers=2,
    train_sample_size=None,
    test_sample_size=None,
    data_dir="../../mnist_data",
):
    mean, std = (0.5,), (0.5,)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    trainset = datasets.MNIST(
        root=data_dir,
        download=True,
        train=True,
        transform=transform,
    )

    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = datasets.MNIST(
        root=data_dir,
        download=True,
        train=False,
        transform=transform,
    )

    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    classes = tuple(range(10))

    return trainloader, testloader, classes


def main(args, model_path=None):
    # Training parameters
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    lr = args["lr"]
    device = args["device"]
    save_model_every_n_epochs = args["save_model_every"]
    # Load the MNIST dataset
    print("Preparing data ...")
    trainloader, testloader, _ = prepare_data(
        batch_size=batch_size, data_dir=args["datadir"]
    )
    # Create the model, optimizer, loss function and trainer
    print("Creating the model, optimizer, loss function and trainer ...")
    model = ViTForClassification(CONFIG)
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args["exp_name"], device=device)
    print("Training starts!")
    trainer.train(
        trainloader,
        testloader,
        epochs,
        save_model_every_n_epochs=save_model_every_n_epochs,
        measure_time=True,
        base_dir=args["basedir"],
    )


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, help="Batch size.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--device", type=str, help="Device for the computation.")
    parser.add_argument(
        "--savemodelevery",
        type=int,
        help="Number of epochs after which a checkpoint is saved.",
    )
    parser.add_argument(
        "--expname", type=str, help="Name of the experiment, where to save the model."
    )
    parser.add_argument("--basedir", type=str, help="Base directory.")
    parser.add_argument("--datadir", type=str, help="Data directory.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    arg_dict = {
        "batch_size": args.batchsize,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": args.device,
        "save_model_every": args.savemodelevery,
        "exp_name": args.expname,
        "basedir": args.basedir,
        "datadir": args.datadir,
    }

    main(arg_dict)
