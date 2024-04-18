import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from jacobian_function import jacobian
from utils import prepare_data, deactivate_dropout_layers, load_model


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
    def simec(input_simec, output_simec):
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

    max_eigenvalues = []
    eigenvalues, _ = simec(
        output_simec=encoder_output[0, 0],
        input_simec=emb_inp_simec,
    )
    max_eigenvalues = [
        torch.tensor(v) for v in torch.max(eigenvalues, dim=1).values.tolist()
    ]
    first_col_cls = torch.zeros(starting_img.size(-1))
    first_col_cls[0] = max_eigenvalues.pop(0)
    max_eigenvalues = (
        torch.stack(max_eigenvalues)
        .reshape(14, 14)
        .repeat_interleave(2, dim=0)
        .repeat_interleave(2, dim=1)
    )
    max_eigenvalues = torch.concat([first_col_cls.unsqueeze(0), max_eigenvalues], dim=0)
    fname = os.path.join(img_out_dir, time.strftime("%Y%m%d-%H%M%S") + ".png")
    fig, ax = plt.subplots()
    ax.imshow(starting_img.squeeze().cpu().detach().numpy() * (-1), cmap="gray")
    custom_cmap = LinearSegmentedColormap.from_list(
        "white_to_pink", ["white", "mediumvioletred"]
    )
    colors = custom_cmap(np.arange(256))
    # Modify alpha values linearly across the colormap
    alphas = np.linspace(0, 0.9, 256)
    colors[:, -1] = alphas  # Replace the alpha channel with the new alpha values
    alpha_cmap = LinearSegmentedColormap.from_list("alpha_cmap", colors, N=256)
    feature_importance = ax.imshow(
        max_eigenvalues.squeeze().cpu().detach().numpy(), cmap=alpha_cmap
    )
    cbar = fig.colorbar(feature_importance, ax=ax)
    cbar.set_label("Eigenvalues")
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    plt.savefig(fname)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--device", type=str)

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
    experiment_name = args.exp_name
    model_path = args.model_path
    config_path = args.config_path
    out_dir = args.out_dir
    device = torch.device(args.device)

    # MNIST data
    trainloader, _ = prepare_data(test=False)

    model, _ = load_model(
        model_path=model_path,
        config_path=config_path,
        device=device,
    )

    deactivate_dropout_layers(model)

    # Get a mini-batch of train data loaders
    imgs, _ = next(iter(trainloader))
    imgs = imgs.to(device)
    # take first image keeping batch dimension
    img = imgs[0].unsqueeze(0)

    simec_feature_importance_vit(
        model=model,
        starting_img=img,
        device=device,
        img_out_dir=os.path.join(
            out_dir,
            "feature-importance",
            experiment_name,
            time.strftime("%Y%m%d-%H%M%S"),
        ),
    )


if __name__ == "__main__":
    main()
