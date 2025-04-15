from torch import from_numpy, float32
from torchvision import datasets

normalize = False

if normalize:
    cifar10_dir = "../data/cifar10_data"
    cifar10 = datasets.CIFAR10(
        root=cifar10_dir, train=True, download=True
    )
    cifar10_data = cifar10.data
    cifar10_data = from_numpy(cifar10_data).flatten(0,-2).to(dtype = float32)

    mnist_dir = "../data/mnist_data"
    mnist = datasets.MNIST(
        root=mnist_dir, train=True, download=True
    )
    mnist_data = mnist.data.to(dtype = float32)

    CIFAR_MEAN = cifar10_data.mean(dim = 0)
    CIFAR_STD = cifar10_data.std(dim = 0)
    print("CIFAR10 mean and sd:", CIFAR_MEAN, CIFAR_STD)

    MNIST_MEAN = mnist_data.mean()
    MNIST_STD = mnist_data.std()
    print("MNIST mean and sd:", MNIST_MEAN, MNIST_STD)