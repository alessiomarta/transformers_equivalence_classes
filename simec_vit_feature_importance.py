import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from jacobian_function import jacobian
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(0)
# np.random.seed(0)

BATCH_SIZE = 128
LR = 5e-5

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


def load_model(fname="mnist_vit.mdl", device="cuda"):
    if device.type == "cpu":
        return torch.load(fname, map_location=torch.device("cpu"))
    if device.type == "mps":
        return torch.load(fname, map_location=torch.device("mps"))
    return torch.load(fname)


def deactivate_dropout_layers(model):
    """Deactivate the dropout layers of the model after the training."""
    model.dropout_layer.in_place = False
    model.dropout_layer.p = 0.0
    for i in range(3):
        model.encoders[i].attention.dropout_layer.p = 0.0
        model.encoders[i].mlp[2].p = 0.0
        model.encoders[i].mlp[4].p = 0.0


def simec_vit(
    model,
    starting_img,
    n_iterations,
    eq_class_patch_id,
    patch_size=None,
    delta=5e-2,
    threshold=1e-2,
    print_every_n_iter=100,
    img_out_dir="",
):
    # if image is not already patched, set to image size
    if not patch_size:
        patch_size = 28
    # 10x10 identity matrix that we use as standard Riemannain metric of the embedding space.
    g = torch.eye(10).to(device)

    # Clone and require gradient of the embedded input and prepare for the first iteration
    simec_input = starting_img.clone()
    simec_input = simec_input.requires_grad_(True)

    output = model(simec_input)

    distance = 0.0
    for i in range(n_iterations):

        # Compute the pullback metric
        jac = jacobian(output[0], simec_input)[eq_class_patch_id]
        # if input has not been patched
        if jac.dim() > 2:
            dimensions = jac.size()
            jac = jac.resize(dimensions[0] * dimensions[1], dimensions[2])
        jac_t = torch.transpose(jac, 0, 1)
        tmp = torch.mm(jac, g)
        if device.type == "mps":
            # mps doen't support float64, must convert in float32
            pullback_metric = torch.mm(tmp, jac_t).type(torch.float32)
        else:
            pullback_metric = torch.mm(tmp, jac_t).type(torch.double)
        # The conversion to double is done in order to avoid the following error:
        # The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric, UPLO="U")

        # Select a random eigenvectors corresponding to a null eigenvalue.
        # We consider an eigenvalue null if it is below a threshold value.
        zero_eigenvalues = eigenvalues < threshold
        number_null_eigenvalues = torch.count_nonzero(zero_eigenvalues)
        id_eigen = torch.randint(0, number_null_eigenvalues, (1,)).item()
        # id_eigen = torch.randint(number_null_eigenvalues+1, 28*28-1, (1,)).item()
        if not patch_size:
            null_vec = eigenvectors[:, id_eigen].resize(28, 28).type(torch.float)
        else:
            null_vec = eigenvectors[:, id_eigen].type(torch.float)

        # Proceeed along a null direction
        simec_input = simec_input.requires_grad_(False)
        simec_input[0, eq_class_patch_id] = (
            simec_input[0, eq_class_patch_id] + null_vec * delta
        )
        distance += eigenvalues[id_eigen].item() * delta

        # Clamp the tensor in [-1,1].
        clamp_lower = simec_input < -1.0
        simec_input[clamp_lower] = -1.0
        clamp_upper = simec_input > 1.0
        simec_input[clamp_upper] = 1.0

        # Prepare for the next iteration.
        simec_input = simec_input.requires_grad_(True)
        output = model(simec_input)

        if i % print_every_n_iter == 0:
            # print(int(i/250))
            print("Length of the polygonal:", distance)
            print(eigenvalues[id_eigen])
            print(torch.nn.functional.softmax(output))
            print("-------------------------------------------------")
            fname = img_out_dir + str(int(i / print_every_n_iter)) + ".png"
            image = simec_input.reshape(1, 1, 28, 28)[0, 0].cpu().detach().numpy()
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            if patch_size:
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


if __name__ == "__main__":
    mean, std = (0.5,), (0.5,)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    trainset = datasets.MNIST(
        "data/MNIST/", download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )

    testset = datasets.MNIST(
        "data/MNIST/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        fname="transformers_equivalence_classes/mnist_vit_trained_mod.mdl",
        device=device,
    )

    deactivate_dropout_layers(model)
    # Get a mini-batch of train data loaders.
    imgs, labels = next(iter(trainloader))

    imgs = imgs.to(device)
    print(torch.max(imgs))
    print(torch.min(imgs))
    n = False
    # print(imgs[0,0])
    # print((testloader.dataset.test_data[0]))
    if not n:
        patch_size = 7
        img = imgs[0].unsqueeze(0)
        b, c, h, w = img.size()
        img = img.reshape(
            b, int((h / patch_size) * (w / patch_size)), c * patch_size * patch_size
        )
        simec_vit(
            model,
            img,
            100000,
            6,
            patch_size=patch_size,
            img_out_dir="transformers_equivalence_classes/data/res_img/exp0/",
        )
    if n:
        simec_vit(model, imgs[0].unsqueeze(0), 100000, 0, img_out_dir="data/res_img/")
