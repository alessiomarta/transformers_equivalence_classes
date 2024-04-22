import os
import time
from collections import defaultdict
import torch
from utils import save_object


def jacobian(nn_output, nn_input):
    """
    Explicitly compute the full Jacobian matrix.

    Args:
        nn_output (torch.Tensor): A model output with gradient attached
        nn_input (torch.Tensor): A model input with gradient attached

    Returns:

    torch.Tensor: The Jacobian matrix, of dimensions torch.Size([len(nn_output), len(nn_input)])
    """

    return torch.stack(
        [
            torch.autograd.grad([nn_output[i]], nn_input, retain_graph=True)[0]
            for i in range(nn_output.size(0))
        ],
        dim=-1,
    )[0].detach()


def pullback(input_simec, output_simec, g, eq_class_emb_ids=None):
    # Compute the pullback metric
    jac = jacobian(output_simec, input_simec)
    if eq_class_emb_ids:
        jac = jac[eq_class_emb_ids]
    jac_t = torch.transpose(jac, -2, -1)
    tmp = torch.bmm(jac, g)
    pullback_metric = torch.bmm(tmp, jac_t).type(torch.double)
    return torch.linalg.eigh(pullback_metric, UPLO="U")


def pullback_eigenvalues(input_embedding, model, pred_id, device):
    # input embedding : (batch, number of embeddings, hidden size)

    # Clone and require gradient of the embedded input and prepare for the first iteration
    input_emb = input_embedding.clone().to(device).requires_grad_(True)
    output_emb = model(input_emb)

    # Build the identity matrix that we use as standard Riemannain metric of the output embedding space.
    g = (
        torch.eye(input_embedding.size(-1))
        .unsqueeze(0)
        .repeat(
            output_emb.size(1),
            1,
            1,
        )
    ).to(device)

    # Compute the pullback metric and its eigenvalues and eigenvectors
    eigenvalues, _ = pullback(
        output_simec=output_emb[0, pred_id].squeeze(),
        input_simec=input_emb,
        g=g,
    )

    return eigenvalues


def explore(
    same_equivalence_class,
    input_embedding,
    model,
    delta,
    threshold,
    n_iterations,
    pred_id,
    device,
    eq_class_emb_ids=None,
    keep_timing=False,
    save_each=10,
    out_dir=".",
):
    # input embedding : (batch, number of embeddings, hidden size)

    # Clone and require gradient of the embedded input and prepare for the first iteration
    input_emb = input_embedding.clone().to(device).requires_grad_(True)
    output_emb = model(input_emb).item()

    # Build the identity matrix that we use as standard Riemannain metric of the output embedding space.
    g = (
        torch.eye(input_embedding.size(-1))
        .unsqueeze(0)
        .repeat(
            input_emb.size(1) if not eq_class_emb_ids else len(eq_class_emb_ids),
            1,
            1,
        )
    ).to(device)

    # Keep track of the length of the polygonal
    distance = torch.zeros(
        input_emb.size(1) if not eq_class_emb_ids else len(eq_class_emb_ids)
    )
    if keep_timing:
        times = defaultdict(float)
        times["n_iterations"] = n_iterations

    for i in range(n_iterations):
        if keep_timing:
            tic = time.time()
        # Compute the pullback metric and its eigenvalues and eigenvectors
        eigenvalues, eigenvectors = pullback(
            output_simec=output_emb[0, pred_id].squeeze(), input_simec=input_emb, g=g
        )

        # Select a random eigenvectors corresponding to a null eigenvalue.
        # We consider an eigenvalue null if it is below a threshold value.
        if same_equivalence_class:
            number_eigenvalues = torch.count_nonzero(eigenvalues < threshold, dim=1)
        else:
            number_eigenvalues = torch.count_nonzero(eigenvalues > threshold, dim=1)
        eigenvecs, eigenvals = [], []
        for emb in range(eigenvalues.size(0)):
            if number_eigenvalues[emb]:
                if same_equivalence_class:
                    id_eigen = torch.randint(0, number_eigenvalues[emb], (1,)).item()
                else:
                    id_eigen = torch.argmax(number_eigenvalues[emb], dim=-1).item()
                eigenvecs.append(eigenvectors[emb, :, id_eigen].type(torch.float))
                eigenvals.append(eigenvalues[emb, id_eigen].type(torch.float))
            else:
                eigenvecs.append(torch.zeros(1).type(torch.float))
                eigenvals.append(torch.zeros(1).type(torch.float))
        eigenvecs = torch.stack(eigenvecs, dim=0)
        eigenvals = torch.stack(eigenvals, dim=0)

        with torch.no_grad():
            # Proceeed along a null direction
            if eq_class_emb_ids:
                input_emb[0, eq_class_emb_ids] = (
                    input_emb[0, eq_class_emb_ids] + eigenvecs * delta
                )
            else:
                input_emb[0] = input_emb[0] + eigenvecs * delta
            distance += eigenvals * delta

            if i % save_each == 0:
                if keep_timing:
                    tic_save = time.time()
                save_object(
                    {
                        "input_embedding": input_emb,
                        "output_embedding": output_emb,
                        "distance": distance,
                        "iteration": i,
                    },
                    os.path.join(out_dir, f"{i}.pkl"),
                )
                if keep_timing:
                    diff = time.time() - tic_save

        # Prepare for next iteration
        input_emb = input_emb.to(device).requires_grad_(True)
        output_emb = model(input_emb).item()
        if keep_timing:
            times["time"] += time.time() - tic
            if i % save_each == 0:
                times["time"] -= diff

    save_object(
        {
            "input_embedding": input_emb,
            "output_embedding": output_emb,
            "distance": distance,
            "iteration": "final",
            "time": times["time"],
        },
        os.path.join(out_dir, "final.pkl"),
    )
