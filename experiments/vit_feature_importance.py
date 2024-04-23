import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from simec.logics import pullback_eigenvalues
from utils import (
    load_raw_images,
    deactivate_dropout_layers,
    load_model,
)


# The operator 'aten::_linalg_eigh.eigenvalues' is not currently
# implemented for the MPS device. If you want this op to be added
# in priority during the prototype phase of this feature, please
# comment on https://github.com/pytorch/pytorch/issues/77764. As a
# temporary fix, you can set the environment variable
# `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
# WARNING: this will be slower than running natively on MPS.
# Exeption when executing
# eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric, UPLO="U")
# export PYTORCH_ENABLE_MPS_FALLBACK=1


def simec_feature_importance_vit(model, starting_img, device, img_out_dir="."):
    def pullback(input_simec, output_simec):
        # Compute the pullback metric
        jac = jacobian(output_simec, input_simec)
        jac_t = torch.transpose(jac, -2, -1)
        tmp = torch.bmm(jac, g)
        if device.type == "mps":
            # mps doen't support float64, must convert in float32
            pullback_metric = torch.bmm(tmp, jac_t).type(torch.float32)
        else:
            # The conversion to double is done in order to avoid the following error:
            # The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues
            pullback_metric = torch.bmm(tmp, jac_t).type(torch.double)
        return torch.linalg.eigh(pullback_metric, UPLO="U")

    # Clone and require gradient of the embedded input and prepare for the first iteration
    emb_inp_simec = model.embedding(model.patcher(starting_img)).requires_grad_(True)

    # Build the identity matrix that we use as standard Riemannain metric of the output embedding space.
    g = (
        torch.eye(model.hidden_size)
        .unsqueeze(0)
        .repeat(model.embedding.num_patches + 1, 1, 1)
    )

    # Compute the output of the encoder. This is the output which we want to keep constant
    encoder_output = model.encoder(emb_inp_simec)[0]

    eigenvalues, _ = pullback(
        output_simec=encoder_output[0, 0],
        input_simec=emb_inp_simec,
    )
    max_eigenvalues = [
        torch.tensor(v) for v in torch.max(eigenvalues, dim=1).values.tolist()
    ]

    max_eigenvalues = (
        torch.stack(max_eigenvalues[1:])
        .reshape(14, 14)
        .repeat_interleave(2, dim=0)
        .repeat_interleave(2, dim=1)
    )
    fname = os.path.join(img_out_dir, time.strftime("%Y%m%d-%H%M%S") + ".png")
    fig = plt.figure(figsize=(8, 4))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    ax1, ax2, cax = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    )

    ax1.imshow(starting_img.squeeze().cpu().detach().numpy(), cmap="gray")
    ax1.axis("off")
    feature_importance = ax2.imshow(max_eigenvalues.squeeze().cpu().detach().numpy())
    ax2.axis("off")
    cbar = plt.colorbar(feature_importance, cax=cax)
    cbar.set_label("Eigenvalues")
    plt.subplots_adjust(wspace=0, hspace=0)
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    plt.savefig(fname)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    # MNIST data
    images, names = load_raw_images(args.img_dir)
    images = images.to(device)

    # load modified ViT model and deactivate it dropout layers
    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
    )
    deactivate_dropout_layers(model)

    # for naming results directories
    str_time = time.strftime("%Y%m%d-%H%M%S")

    for idx, img in enumerate(images):

        # Clone and require gradient of the embedded input and prepare for the
        # first iteration
        input_patches = model.patcher(img.unsqueeze(0))
        input_embedding = model.embedding(input_patches)

        eigenvalues = pullback_eigenvalues(
            model=model.encoder,
            input_embedding=input_embedding,
            pred_id=0,
            device=device,
            out_dir=os.path.join(
                args.out_dir,
                "feature-importance",
                args.exp_name + "-" + str_time,
                names[idx],
            ),
        )


if __name__ == "__main__":
    main()
