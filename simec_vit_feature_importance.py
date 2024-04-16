import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from jacobian_function import jacobian, jacobian2D
from data import prepare_data
from vit import ViTForClassfication

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

CONFIG = {
    "patch_size": 2,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.01,
    "attention_probs_dropout_prob": 0.01,
    "initializer_range": 0.02,
    "image_size": 28,
    "num_classes": 10,
    "num_channels": 1,
    "qkv_bias": True,
}


class PatchDecoder(nn.Module):
    """
    Convert the embeddings so that they can be displayed as images
    """

    def __init__(
        self, image_size, patch_size, num_channels, hidden_size, positional_embeddings
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.positional_embeddings = positional_embeddings
        # Calculate the number of patches from the image size and patch size
        self.num_patches = self.image_size // self.patch_size
        # Create a projection layer to convert embeddings into images
        self.projection = nn.ConvTranspose2d(
            self.hidden_size,
            self.num_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        # reverting the operations done by PatchEmbedder
        x = x - self.positional_embeddings
        # removing first embedding because it is the classifier's
        x = (
            x[:, 1:, :]
            .transpose(1, 2)
            .reshape(1, self.hidden_size, self.num_patches, self.num_patches)
        )
        x = self.projection(x)
        return x


def load_model(fname, device):
    if device.type == "cpu":
        checkpoint = torch.load(fname, map_location=torch.device("cpu"))
    elif device.type == "mps":
        checkpoint = torch.load(fname, map_location=torch.device("mps"))
    else:
        checkpoint = torch.load(fname)
    model = ViTForClassfication(CONFIG)
    model.load_state_dict(checkpoint)
    return model


def deactivate_dropout_layers(model):
    """Deactivate the dropout layers of the model after training."""
    model.embedding.dropout.p = 0.0
    for block in model.encoder.blocks:
        block.attention.attn_dropout.p = 0.0
        block.attention.output_dropout.p = 0.0
        block.mlp.dropout.p = 0.0


def simec_vit(
    model,
    starting_img,
    n_iterations,
    eq_class_patch_id,
    patch_size,
    device,
    delta=5e-2,
    threshold=1e-2,
    print_every_n_iter=100,
    img_out_dir=".",
):
    def simec(input_simec, output_simec, select_patch=True):
        # Compute the pullback metric
        if select_patch:
            jac = jacobian(output_simec, input_simec)[eq_class_patch_id]
            jac_t = torch.transpose(jac, -2, -1)
            tmp = torch.mm(jac, g)
            if device.type == "mps":
                # mps doen't support float64, must convert in float32
                pullback_metric = torch.mm(tmp, jac_t).type(torch.float32)
            else:
                # The conversion to double is done in order to avoid the following error:
                # The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues
                pullback_metric = torch.mm(tmp, jac_t).type(torch.double)
            return torch.linalg.eigh(pullback_metric, UPLO="U")
        else:
            jac = jacobian2D(output_simec, input_simec)
            jac_t = jac.permute(-1, -2, -3, -4)
            eigenvalues, eigenvectors = [], []
            for h in range(jac.size(0)):
                for w in range(jac.size(1)):
                    tmp = torch.mm(jac[h, w], g)
                    if device.type == "mps":
                        pullback_metric = torch.mm(tmp, jac_t[:, :, w, h]).type(
                            torch.float32
                        )
                        eigenvalues.append()
                    else:
                        pullback_metric = torch.mm(tmp, jac_t[:, :, w, h]).type(
                            torch.double
                        )
                    # Compute eigenvalues and eigenvectors
                    values, vectors = torch.linalg.eigh(pullback_metric, UPLO="U")
                    eigenvalues.append(values)
                    eigenvectors.append(vectors)
            eigenvectors = torch.stack(eigenvectors, dim=-1).reshape(28, 28, 28, 28)
            eigenvalues = torch.stack(eigenvalues, dim=-1).reshape(28, 28, 28)
            return eigenvectors, eigenvalues

    # fname = img_out_dir + "start.png"
    image = starting_img.squeeze().cpu().detach().numpy()
    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    # plt.savefig(fname)
    # plt.clf()

    # define decoder to plot images from embeddings
    decoder = PatchDecoder(
        image_size=model.image_size,
        patch_size=patch_size,
        num_channels=starting_img.shape[1],
        hidden_size=model.hidden_size,
        positional_embeddings=model.embedding.position_embeddings,
    )

    # prepare eigenvectors to project embeddings back to the image space
    output_decoder = decoder(model.embedding(starting_img.requires_grad_(True)))
    g = torch.eye(output_decoder.shape[-1])  # , output_emb.shape[-2] - 1)
    eigenvectors_decoder, _ = simec(
        input_simec=starting_img,
        output_simec=output_decoder.squeeze(),
        select_patch=False,
    )

    # Clone and require gradient of the embedded input and prepare for the first iteration
    emb_inp_simec = model.embedding(starting_img).requires_grad_(True)

    # Build the identity matrix that we use as standard Riemannain metric of the output embedding space.
    embedding_dimension = emb_inp_simec.shape[-1]
    g = torch.eye(embedding_dimension)

    # Plot embeddings as image with decoder
    image_emb_inp_simec = decoder(emb_inp_simec.clone()).squeeze().type(torch.double)
    basis_changes = []
    for h in range(image_emb_inp_simec.size(0)):
        for w in range(image_emb_inp_simec.size(1)):
            eigevec = eigenvectors_decoder[h, w]
            eigevec_inverse = torch.inverse(eigevec)
            basis_change = torch.mm(eigevec_inverse, image_emb_inp_simec)
            basis_change = torch.mm(basis_change, eigevec)
            basis_changes.append(basis_change)
    # fname = img_out_dir + ".png"
    image = basis_change.squeeze().cpu().detach().numpy()
    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    # plt.savefig(fname)
    # plt.clf()

    # Compute the output of the encoder. This is the output which we want to keep constant
    encoder_output = model.encoder(emb_inp_simec)[0]

    # Keep track of the length of the polygonal
    distance = 0.0
    for i in range(n_iterations):
        # simec --------------------------------------------------------------
        eigenvalues, eigenvectors = simec(
            output_simec=encoder_output[0, 0], input_simec=emb_inp_simec
        )

        # random walk step ---------------------------------------------------

        # Select a random eigenvectors corresponding to a null eigenvalue.
        # We consider an eigenvalue null if it is below a threshold value.
        zero_eigenvalues = eigenvalues < threshold
        number_null_eigenvalues = torch.count_nonzero(zero_eigenvalues)
        id_eigen = torch.randint(0, number_null_eigenvalues, (1,)).item()
        null_vec = eigenvectors[:, id_eigen].type(torch.float)
        # Proceeed along a null direction
        emb_inp_simec = emb_inp_simec.detach()
        emb_inp_simec[0, eq_class_patch_id] = (
            emb_inp_simec[0, eq_class_patch_id] + null_vec * delta
        )
        distance += eigenvalues[id_eigen].item() * delta
        # Clamp the tensor in [-1,1].
        clamp_lower = emb_inp_simec < -1.0
        emb_inp_simec[clamp_lower] = -1.0
        clamp_upper = emb_inp_simec > 1.0
        emb_inp_simec[clamp_upper] = 1.0

        # Prepare for the next iteration.
        emb_inp_simec = emb_inp_simec.requires_grad_(True)
        encoder_output = model.encoder(emb_inp_simec)[0]

        if i % print_every_n_iter == 0:
            print("Length of the polygonal:", distance)
            print(eigenvalues[id_eigen])
            print("-------------------------------------------------")
            fname = img_out_dir + str(int(i / print_every_n_iter)) + ".png"

            image = emb_inp_simec - model.embedding.position_embeddings
            image = image[:, 1:, :].transpose(1, 2).reshape(1, 48, 14, 14)
            image = decoder(image)
            image = starting_img + image_emb_inp_simec - image
            image = image.squeeze().cpu().detach().numpy()
            _, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.add_patch(
                Rectangle(
                    tuple(
                        (
                            np.array(
                                np.unravel_index(
                                    eq_class_patch_id,
                                    (28 // patch_size, 28 // patch_size),
                                )
                            )
                            * patch_size
                        )
                        + np.array([-0.5, -0.5])
                    )[::-1],
                    patch_size,
                    patch_size,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            plt.savefig(fname)
            plt.clf()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--which-patch", type=int, default=0)
    parser.add_argument("--iter", type=int, default=10)
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

    model = load_model(
        fname=model_path,
        device=device,
    )

    with open(config_path, "r") as c:
        model_config = json.load(c)

    patch_size = model_config["patch_size"]

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
        eq_class_patch_id=eq_class_patch,
        patch_size=patch_size,
        device=device,
        img_out_dir=os.path.join(out_dir, experiment_name),
    )


if __name__ == "__main__":
    main()
