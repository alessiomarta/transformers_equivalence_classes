from utils import *
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import json


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--img-dir", type = str, required = True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--timer", type=float, required=True)
    parser.add_argument("--pert-step", type=float, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    args = parser.parse_args()
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ).type
    return args


def orthogonal_vector(v):

    M = torch.eye(len(v))
    random_dim = torch.randint(high=len(v))
    M[random_dim, :] = -1 * v / v[random_dim]
    M[random_dim, random_dim] = 0

    return M @ v


def perturbation(model, starting_img, step, direction, y):

    noise = direction.reshape(starting_img.shape)

    new_img = starting_img + step * noise

    logits = model(new_img).flatten().numpy()
    new_y = np.argmax(logits)

    orthogonal_direction = 0.1 * orthogonal_vector(direction)
    new_direction = direction + orthogonal_direction

    while new_y != y:

        noise = new_direction.reshape(starting_img.shape)
        new_img = starting_img + step * noise

        logits = model(new_img).flatten().numpy()
        new_y = np.argmax(logits)

    return new_img, new_direction


def main():

    args = parse_args()
    model_path = args.model_path
    config_path = args.config_path
    device = torch.device(args.device)
    initial_time = args.time
    step = args.pert_step
    img_out_dir = args.out_dir
    img_in_dir = args.img_dir

    # Load images
    images, names = load_raw_images(args.img_dir)
    images = images.to(device)

    model, _ = load_model(
        model_path=model_path,
        config_path=config_path,
        device=device,
    )

    deactivate_dropout_layers(model)

    
    for name, img in zip(names, images):
        # Initialize mean and standard deviation of noise to 0 and 1
        mu = 0
        sd = 1
        direction = torch.normal(mean=mu, std=sd, size=img.shape).flatten()
        count = 0
        norm = Normalize(vmin=-1, vmax=1)

        timer = initial_time

        while timer > 0:

            tic = time()

            logits = model(img).flatten().numpy()
            y = np.argmax(logits)

            img, direction = perturbation(
                model=model, starting_img=img, step=step, direction=direction, y=y
            )

            timedelta = time() - tic
            timer -= timedelta

            fname = os.path.join(img_out_dir, str(count) + ".png")
            _, ax = plt.subplots()
            ax.imshow(img.squeeze().cpu().numpy(), cmap="gray", norm=norm)
            if not os.path.exists(img_out_dir):
                os.makedirs(img_out_dir)
            plt.savefig(fname)
            plt.close()

            count += 1

        stats = {"time": int(initial_time - timer), "count": count}

        with open(os.path.join(img_out_dir, name, "stats.json"), "w") as f:
            json.dump(stats, f)


if __name__ == "__main__":
    main()
