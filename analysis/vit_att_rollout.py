import argparse
import torch
from vit_explain.vit_grad_rollout import VITAttentionGradRollout
from experiments_utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--img-dir", type = str, required = True)
    parser.add_argument("--device", type=str)
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


def main():

    args = parse_args()
    model_path = args.model_path
    config_path = args.config_path
    device = torch.device(args.device)
    img_in_dir = args.img_dir
    img_out_dir = args.out_dir

    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    # Load images
    images, names = load_raw_images(img_in_dir)
    images = images.to(device)

    model, _ = load_model(
        model_path=model_path,
        config_path=config_path,
        device=device,
    )
    model = model.to(device)

    deactivate_dropout_layers(model)

    grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.9)

    for name, img in zip(names, images):

        fname = os.path.join(img_out_dir, f"{name}.png")

        print("Image:", name)

        pred = model(img.unsqueeze(0))[0].flatten().cpu().detach().numpy().argmax()

        mask = grad_rollout(img.unsqueeze(0), category_index = pred)

        fig = plt.figure(figsize=(8, 4))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        ax1, ax2, cax = (
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
        )

        ax1.imshow(img.squeeze().cpu().detach().numpy(), cmap="gray")
        ax1.axis("off")

        feature_importance = ax2.imshow(mask)
        ax2.axis("off")

        cbar = plt.colorbar(feature_importance, cax=cax)
        cbar.set_label("Attention rollout")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(fname)
        plt.close()


if __name__ == "__main__":

    main()