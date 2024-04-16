import torch


def jacobian(output_, input_):
    """
    Explicitly compute the full Jacobian matrix.

    Args:
        output (torch.Tensor): A model output with gradient attached
        input (torch.Tensor): A model input with gradient attached

    Returns:

    torch.Tensor: The Jacobian matrix, of dimensions torch.Size([len(output), len(input)])
    """

    du = torch.stack(
        [
            torch.autograd.grad([output_[i]], input_, retain_graph=True)[0]
            for i in range(output_.size(0))
        ],
        dim=-1,
    )
    J = du[0]

    # Return a new Tensor detached from the current graph.
    return J.detach()


def jacobian2D(output_, input_):
    grad = []
    for h in range(output_.size(0)):
        for w in range(output_.size(1)):
            grad.append(
                torch.autograd.grad(
                    outputs=output_[h, w], inputs=input_, retain_graph=True
                )[0]
            )
    grad = torch.stack(grad, dim=-1).squeeze()
    # no batch
    grad = grad.reshape(*tuple(output_.size()), *tuple(output_.size())).detach()
    return grad
