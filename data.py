import torch
from torchvision import datasets, transforms


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
