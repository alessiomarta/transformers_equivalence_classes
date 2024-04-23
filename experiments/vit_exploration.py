import argparse
from collections import defaultdict
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image
import torch
from torchvision import transforms
from simec.logics import explore
from models.vit import PatchDecoder
from utils import prepare_data, deactivate_dropout_layers, load_model


def load_raw_images(img_dir):
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


def vit_exploration_experiment(
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
    def pullback(input_simec, output_simec):
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

    times = defaultdict(float)
    times["n_iterations"] = n_iterations

    # Keep track of the length of the polygonal
    distance = torch.zeros(len(eq_class_patch_ids))
    for i in range(n_iterations):

        tic = time.time()

        # simec --------------------------------------------------------------
        eigenvalues, eigenvectors = pullback(
            output_simec=encoder_output[0, 0],
            input_simec=emb_inp_simec,
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

            times["time"] += time.time() - tic

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
    parser.add_argument("--exp-type", type=str, choices=["same", "diff"], required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--which-patch", nargs="+", default=90)
    parser.add_argument("--keep-constant", default=0)
    parser.add_argument("--delta", type=float, default=9e-1)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--iter", type=int, default=100)
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
    eq_class_patch = args.which_patch if args.which_patch[0].isdigit() else None
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

    str_time = time.strftime("%Y%m%d-%H%M%S")

    for idx, img in enumerate(images):

        # Clone and require gradient of the embedded input and prepare for the first iteration
        input_patches = model.patcher(img.unsqueeze(0))
        input_embedding = model.embedding(input_patches)

        # input exploration
        explore(
            same_equivalence_class=args.exp_type == "same",
            input_embedding=input_embedding,
            model=model.encoder,
            delta=args.delta,
            threshold=args.threshold,
            n_iterations=args.iter,
            pred_id=args.keep_constant,
            eq_class_emb_ids=eq_class_patch,
            device=device,
            out_dir=os.path.join(
                args.out_dir,
                "input-space-exploration",
                args.exp_name + "-" + str_time,
                names[idx],
            ),
        )


if __name__ == "__main__":
    main()
