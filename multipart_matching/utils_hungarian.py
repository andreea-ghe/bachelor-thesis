import torch
from torch import Tensor
import numpy as np
import scipy.optimize as opt


def hungarian(m: Tensor, n1: Tensor, n2: Tensor):
    """
    Solves the Linear Assignment Problem (LAP) to find the optimal discrete
    (binary) matching that maximizes the total affinity. This is O(n³) but
    produces a hard assignment (0 or 1 for each entry).

    Input:
        m: [B, N, M] - input affinity matrix (logits)
        n1: [B] - number of valid rows for each batch
        n2: [B] - number of valid columns for each batch

    Output:
        m: [B, N, M] - optimal discrete matching matrix
    """
    if len(m.shape) == 2:
        m = m.unsqueeze(0)  # add batch dimension if missing
        matrix_was_2d = True
    elif len(m.shape) == 3:
        matrix_was_2d = False
    else:
        raise ValueError("Input affinity matrix must be 2D or 3D tensor.")

    device = m.device
    B = m.shape[0]

    # convert logits to numpy array for scipy linear_sum_assignment processing
    m = m.cpu().detach().numpy() * -1 # we minimize cost, so negate the affinity matrix
    n1 = n1.cpu().detach() if n1 is not None else [None] * B
    n2 = n2.cpu().detach() if n2 is not None else [None] * B

    m = np.stack(
        [hungarian_kernel(m[b], n1[b], n2[b]) for b in range(B)],
    )

    # convert back to torch tensor
    m = torch.from_numpy(m).to(device)

    if matrix_was_2d:
        # squeeze_ removes the dimension in-place; squeeze removes and returns a new tensor
        m = m.squeeze_(0)  # remove batch dimension if it was originally missing

    return m

def hungarian_kernel(m: Tensor, n1: int = None, n2: int = None) -> np.ndarray:
    """
    Solves the Linear Assignment Problem (LAP) for a single affinity matrix
    using the Hungarian algorithm from scipy.

    Input:
        m: [N, M] - input affinity matrix (logits)
        n1: int - number of valid rows
        n2: int - number of valid columns

    Output:
        perm_m: [N, M] - optimal discrete matching matrix
    """
    if n1 is None:
        n1 = m.shape[0]
    if n2 is None:
        n2 = m.shape[1]

    # solve linear sum assignment problem
    row, col = opt.linear_sum_assignment(m[:n1, :n2])

    # create permutation matrix from row, col indices
    perm_m = np.zeros_like(m)
    perm_m[row, col] = 1
    return perm_m