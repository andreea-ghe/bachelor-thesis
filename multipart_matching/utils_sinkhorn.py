import torch
import numpy as np
from torch import Tensor


class Sinkhorn(nn.Module):
    """
    Wrap Sinkhorn algorithm as a nn.Module for easy integration into neural networks.

    Input:
        max_iter: int - maximum number of Sinkhorn iterations
        tau: float - temperature parameter for scaling logits
    """
    def __init__(self, max_iter: int = 20, tau: float = 1.0):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau

    def forward(self, m: Tensor, nrows: Tensor = None, ncols: Tensor = None) -> Tensor:
        return sinkhorn(m, nrows, ncols, self.max_iter, self.tau)


def sinkhorn(m: Tensor, nrows: Tensor = None, ncols: Tensor = None, max_iter: int = 20, tau: float = 1.0) -> Tensor:
    """
    Perform the Sinkhorn algorithm: it iteratively normalizes the rows and columns
    to convert an affinity matrix into a doubly stochastic matrix (matrix where rows and columns sum to 1).

    Steps:
    1. Start with affinity matrix M
    2. Apply temperature scaling: log_M = M / tau
    3. Alternately normalize rows and columns in log space:
        - even iterations: log_M -= logsumexp(log_M, dim=cols)
        - odd iterations: log_M -= logsumexp(log_M, dim=rows)
    4. Repeat for max_iter iterations
    5. Convert back: M = exp(log_M)

    Input:
        m: [B, N, M] - input affinity matrix (logits)
        nrows: [B] - number of valid rows for each batch
        ncols: [B] - number of valid columns for each batch
        max_iter: int - maximum number of Sinkhorn iterations
        tau: float - temperature parameter for scaling logits

    Output:
        m: [B, N, M] - normalized doubly stochastic matrix
    """
    batch_size = m.size(0)

    # ensure nrow <= ncol for efficient processing because the most expensive processing is done over the smaller dimension
    if m.shape[2] >= m.shpe[1]:
        transpose = False
    else:
        m = m.transpose(1, 2)
        nrows, ncols = ncols, nrows
        transpose = True

    if nrows is None: # if no valid rows: assume all rows are valid
        nrows = torch.tensor([m.shape[1] for _ in range(batch_size)], device=m.device)
    if ncols is None: # if no valid columns: assume all columns are valid
        ncols = torch.tensor([m.shape[2] for _ in range(batch_size)], device=m.device)

    # apply temperature scaling 
    log_m = m / tau

    # create masks for valid rows and columns
    row_mask = torch.zeros(batch_size, log_m.shape[1], 1, dtype=torch.bool, device=log_m.device)
    col_mask = torch.zeros(batch_size, 1, log_m.shape[2], dtype=torch.bool, device=log_m.device)
    
    for b in range(batch_size):
        row_mask[b, :nrows[b], 0] = 1
        col_mask[b, 0, :ncols[b]] = 1

    # initialize regularized log-matrix
    reg_log_m = torch.full(
        (batch_size, log_m.shape[1], log_m.shape[2]),
        -float('inf'),
        device=log_m.device,
        dtype=log_m.dtype
    )

    # each batch element gets its own Sinkhorn normalization
    for b in range(batch_size):
        # slice out the valid submatrix
        row_slice = slice(0, nrows[b])
        col_slice = slice(0, ncols[b])
        log_m_b = log_m[b, row_slice, col_slice]
        # get the corresponding masks
        row_mask_b = row_mask[b, row_slice, :]
        col_mask_b = col_mask[b, :, col_slice]

        for i in range(max_iter):
            # we alternate because normalizing rows breaks column sum and vice versa
            if i % 2 == 0: # normalize rows
                log_sum = torch.logsumexp(log_m_b, dim=1, keepdim=True) # log of sum over columns
                log_m_b = log_m_b - torch.where(row_mask_b, log_sum, torch.zeros_like(log_sum))
            else: # normalize columns
                log_sum = torch.logsumexp(log_m_b, dim=0, keepdim=True) # log of sum over rows
                log_m_b = log_m_b - torch.where(col_mask_b, log_sum, torch.zeros_like(log_sum))

        # write back the regularized log-matrix
        reg_log_m[b, row_slice, col_slice] = log_m_b

        # after one iteration: ret_log_s[b, :nrows[b], :ncols[b]] is doubbly stochastic in log space
        # everything else is masked out

    if transpose: # if we transposed at the beginning: transpose back
        reg_log_m = reg_log_m.transpose(1, 2)

    # convert from log space back to probability space
    return torch.exp(reg_log_m)

