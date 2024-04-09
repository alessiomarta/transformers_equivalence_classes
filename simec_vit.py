import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import save_image
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from transformer_package.models import ViT

from jacobian_function import jacobian

import matplotlib.pyplot as plt

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(0)
# np.random.seed(0)

BATCH_SIZE = 128
LR = 5e-5


def load_model(fname="mnist_vit.mdl"):
    model = torch.load(fname)
    return model


def train_model(
    device,
    num_epochs,
    trainloader,
    image_size=28,
    channel_size=1,
    patch_size=7,
    embed_size=512,
    num_heads=8,
    classes=10,
    num_layers=3,
    hidden_size=256,
    dropout=0.2,
    save=True,
):

    model = ViT(
        image_size,
        channel_size,
        patch_size,
        embed_size,
        num_heads,
        classes,
        num_layers,
        hidden_size,
        dropout=dropout,
    ).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    loss_hist = {}
    loss_hist["train accuracy"] = []
    loss_hist["train loss"] = []

    for epoch in range(num_epochs):
        model.train()

        epoch_train_loss = 0

        y_true_train = []
        y_pred_train = []

        for batch_idx, (img, labels) in enumerate(trainloader):
            img = img.to(device)
            labels = labels.to(device)

            preds = model(img)

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())

            epoch_train_loss += loss.item()

        loss_hist["train loss"].append(epoch_train_loss)
        total_correct = len(
            [True for x, y in zip(y_pred_train, y_true_train) if x == y]
        )
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total

        loss_hist["train accuracy"].append(accuracy)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
        print("-------------------------------------------------")

    if save:
        torch.save(model, "mnist_vit.mdl")

    return model


def deactivate_dropout_layers(model):
    """Deactivate the dropout layers of the model after the training."""
    model.dropout_layer.in_place = False
    model.dropout_layer.p = 0.0
    for i in range(3):
        model.encoders[i].attention.dropout_layer.p = 0.0
        model.encoders[i].mlp[2].p = 0.0
        model.encoders[i].mlp[4].p = 0.0


def simec_vit(
    model,
    starting_img,
    n_iterations,
    delta=5e-2,
    threshold=1e-2,
    print_every_n_iter=100,
):

    # 10x10 identity matrix that we use as standard Riemannain metric of the embedding space.
    g = torch.eye(10).to(device)

    # Clone and require gradient of the embedded input and prepare for the first iteration
    simec_input = starting_img.clone().view(1, 1, 28, 28)
    simec_input = simec_input.requires_grad_(True)
    output = model(simec_input)

    distance = 0.0
    for i in range(n_iterations):

        # Compute the pullback metric
        jac = jacobian(output[0], simec_input)[0]
        dimensions = jac.size()
        jac = jac.resize(dimensions[0] * dimensions[1], dimensions[2])
        jac_t = torch.transpose(jac, 0, 1)
        tmp = torch.mm(jac, g)
        pullback_metric = torch.mm(tmp, jac_t).type(torch.double)
        # The conversion to double is done in order to avoid the following error:
        # The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric, UPLO="U")

        # Select a random eigenvectors corresponding to a null eigenvalue.
        # We consider an eigenvalue null if it is below a threshold value.
        zero_eigenvalues = eigenvalues < threshold
        number_null_eigenvalues = torch.count_nonzero(zero_eigenvalues)
        id_eigen = torch.randint(0, number_null_eigenvalues, (1,)).item()
        # id_eigen = torch.randint(number_null_eigenvalues+1, 28*28-1, (1,)).item()
        null_vec = eigenvectors[:, id_eigen].resize(28, 28).type(torch.float)

        # Proceeed along a null direction
        simec_input = simec_input.requires_grad_(False)
        simec_input[0, 0] = simec_input[0, 0] + null_vec * delta
        distance += eigenvalues[id_eigen].item() * delta

        # Clamp the tensor in [-1,1].
        clamp_lower = simec_input < -1.0
        simec_input[clamp_lower] = -1.0
        clamp_upper = simec_input > 1.0
        simec_input[clamp_upper] = 1.0

        # Prepare for the next iteration.
        simec_input = simec_input.requires_grad_(True)
        output = model(simec_input)

        if i % print_every_n_iter == 0:
            # print(int(i/250))
            print("Length of the polygonal:", distance)
            print(eigenvalues[id_eigen])
            print(torch.nn.functional.softmax(output))
            print("-------------------------------------------------")
            fname = str(int(i / print_every_n_iter)) + ".png"
            fig = plt.figure
            plt.imshow(simec_input[0, 0].cpu().detach().numpy(), cmap="gray")
            plt.savefig(fname)
            plt.clf()


if __name__ == "main":
    mean, std = (0.5,), (0.5,)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    trainset = datasets.MNIST(
        "data/MNIST/", download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )

    testset = datasets.MNIST(
        "data/MNIST/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = load_model(fname="mnist_vit.mdl")

    deactivate_dropout_layers(model)
    # Get a mini-batch of train data loaders.
    imgs, labels = next(iter(trainloader))

    imgs = imgs.to(device)
    print(torch.max(imgs))
    print(torch.min(imgs))

    # print(imgs[0,0])
    # print((testloader.dataset.test_data[0]))

    simec_vit(model, imgs[0], 100000)
