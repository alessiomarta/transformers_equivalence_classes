import argparse
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from typing import Tuple, List
import os
from tqdm.auto import tqdm


def load_raw_images(img_dir: str) -> Tuple[torch.Tensor, List[str]]:
    """
    Load images from a directory, convert them to grayscale, resize to 28x28, and apply a standard transformation.

    Args:
        img_dir: The directory from which images are loaded.

    Returns:
        A tuple containing a batch of tensor images and their corresponding names.
    """
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


def image_filter(dir, tol):

    print(dir)

    image_tensor, _ = load_raw_images(dir)

    image_tensor = image_tensor.reshape(image_tensor.shape[0], -1).numpy()

    variances = [np.var(v) for v in tqdm(image_tensor)]

    return sum(map(lambda x: int(x > tol), variances))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--tol", type=float, default=0.01)
    args = parser.parse_args()

    counts = []

    for experiment in os.listdir(args.dir):
        for folder in os.listdir(os.path.join(args.dir, experiment)):
            if "." in folder:
                continue
            N = image_filter(os.path.join(args.dir, experiment, folder), args.tol)
            counts.append(N)

    print(np.mean(counts), np.std(counts))


if __name__ == "__main__":
    main()
