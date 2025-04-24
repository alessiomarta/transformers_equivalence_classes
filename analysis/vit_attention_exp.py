from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import numpy as np
import cv2
import argparse
import sys

sys.path.append("../")
from experiments.experiments_utils import *


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
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


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).to(cam.device)
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1]).to(grad.device)
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model, input, device, index=None):
    output = model(input, save_attn_gradients=True)[0].to(device)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad(set_to_none=True)
    one_hot.backward(retain_graph=True)

    num_tokens = model.encoder.blocks[0].attention.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).to(device)
    for blk in model.encoder.blocks:
        grad = blk.attention.get_attn_gradients().to(device)
        cam = blk.attention.get_attention_map().to(device)
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R, cam)
        del grad
        del cam
    return R[0, :]


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

    for name, img in zip(names, images):

        fname = os.path.join(img_out_dir, f"{name}.png")

        print("Image:", name)

        pred = model(img.unsqueeze(0))[0].flatten().cpu().detach().numpy().argmax()

        transformer_attribution = generate_relevance(
            model, img.unsqueeze(0), index=pred, device=device
        ).detach()
        transformer_attribution = transformer_attribution.reshape(img.shape[:2])

        fig = plt.figure(figsize=(8, 4))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        ax1, ax2, cax = (
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
        )

        ax1.imshow(img.squeeze().cpu().detach().numpy(), cmap="gray")
        ax1.axis("off")

        feature_importance = ax2.imshow(transformer_attribution)
        ax2.axis("off")

        cbar = plt.colorbar(feature_importance, cax=cax)
        cbar.set_label("Relevancy")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(fname)
        plt.close()


if __name__ == "__main__":

    main()
