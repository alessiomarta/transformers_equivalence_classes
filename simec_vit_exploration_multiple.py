import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import torch
from jacobian_function import jacobian
from utils import prepare_data, deactivate_dropout_layers, load_model
from vit import PatchDecoder


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


def simec_vit(
    model,
    starting_img,
    n_iterations,
    eq_class_patch_ids,
    device,
    delta=9e-1,
    threshold=1e-2,
    print_every_n_iter=4,
    img_out_dir=".",
):
    def simec(input_simec, output_simec, eq_class_patch_ids):
        # Compute the pullback metric
        jac = jacobian(output_simec, input_simec)[eq_class_patch_ids]
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
    input_patches = model.patcher(starting_img)
    emb_inp_simec = model.embedding(input_patches)
    emb_inp_simec = emb_inp_simec.requires_grad_(True)

    # Build the identity matrix that we use as standard Riemannain metric of the output embedding space.

    g = torch.eye(model.hidden_size).unsqueeze(0).repeat(len(eq_class_patch_ids), 1, 1)

    # Compute the output of the encoder. This is the output which we want to keep constant
    encoder_output = model.encoder(emb_inp_simec)[0]

    # define decoder to plot images from embeddings
    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=model.embedding.patch_size,
        model_embedding_layer=model.embedding,
    )

    # Keep track of the length of the polygonal
    distance = torch.zeros(len(eq_class_patch_ids))
    for i in range(n_iterations):
        # simec --------------------------------------------------------------
        eigenvalues, eigenvectors = simec(
            output_simec=encoder_output[0, 0],
            input_simec=emb_inp_simec,
            eq_class_patch_ids=eq_class_patch_ids,
        )

        # random walk step ---------------------------------------------------

        # Select a random eigenvectors corresponding to a null eigenvalue.
        # We consider an eigenvalue null if it is below a threshold value.
        number_null_eigenvalues = torch.count_nonzero(eigenvalues < threshold, dim=1)
        null_vecs, zero_eigenvals = [], []
        for emb in range(eigenvalues.size(0)):
            if number_null_eigenvalues[emb]:
                id_eigen = torch.randint(0, number_null_eigenvalues[emb], (1,)).item()
                null_vecs.append(eigenvectors[emb, :, id_eigen].type(torch.float))
                zero_eigenvals.append(eigenvalues[emb, id_eigen].type(torch.float))
            else:
                null_vecs.append(torch.zeros(1).type(torch.float))
                zero_eigenvals.append(torch.zeros(1).type(torch.float))
        null_vecs = torch.stack(null_vecs, dim=0)
        zero_eigenvals = torch.stack(zero_eigenvals, dim=0)

        with torch.no_grad():
            # Proceeed along a null direction
            emb_inp_simec[0, eq_class_patch_ids] = (
                emb_inp_simec[0, eq_class_patch_ids] + null_vecs * delta
            )
            distance += zero_eigenvals * delta
            # Clamp the tensor in [-1,1].
            # emb_inp_simec[emb_inp_simec < -1.0] = -1.0
            # emb_inp_simec[emb_inp_simec > 1.0] = 1.0

            norm = Normalize(vmin=-1, vmax=1)

            if i % print_every_n_iter == 0:
                print("Length of the polygonal:", distance)
                print(eigenvalues[:, id_eigen])
                fname = os.path.join(
                    img_out_dir, str(int(i / print_every_n_iter)) + ".png"
                )
                image = decoder(emb_inp_simec)
                print("-------------------------------------------------")
                _, ax = plt.subplots()
                ax.imshow(image.squeeze().cpu().numpy(), cmap="gray", norm=norm)
                for p in eq_class_patch_ids:
                    image_index = (
                        np.array(
                            np.unravel_index(
                                p - 1,
                                (
                                    28 // model.embedding.patch_size,
                                    28 // model.embedding.patch_size,
                                ),
                            )
                        )
                        * model.embedding.patch_size
                    )[::-1]
                    ax.add_patch(
                        Rectangle(
                            tuple(image_index + np.array([-0.5, -0.5])),
                            model.embedding.patch_size,
                            model.embedding.patch_size,
                            linewidth=1,
                            edgecolor="r",
                            facecolor="none",
                        )
                    )
                if not os.path.exists(img_out_dir):
                    os.makedirs(img_out_dir)
                plt.savefig(fname)
                plt.close()

        # Prepare for the next iteration.
        emb_inp_simec = emb_inp_simec.requires_grad_(True)
        encoder_output = model.encoder(emb_inp_simec)[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--which-patch", type=int, default=90)
    parser.add_argument("--iter", type=int, default=100)
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
    iterations = args.iter
    eq_class_patch = args.which_patch
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

    simec_vit(
        model=model,
        starting_img=img,
        n_iterations=iterations,
        eq_class_patch_ids=torch.randint(1, 197, (10,)).tolist(),
        device=device,
        img_out_dir=os.path.join(
            out_dir,
            "input-space-exploration",
            experiment_name,
            time.strftime("%Y%m%d-%H%M%S"),
        ),
    )


if __name__ == "__main__":
    main()
