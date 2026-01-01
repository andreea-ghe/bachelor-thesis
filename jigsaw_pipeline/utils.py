import torch


def get_batch_length_from_part_points(n_pcs, n_valids):
    """
    Given number of points in each part and number of valid parts,
    return a 1D tensor of lengths of all valid parts in the batch.

    Input:
        n_pcs: [B, P] - number of points in each part
        n_valids: [B] - number of valid parts in each batch

    Return:
        batch_length: [sum(n_valids)] - lengths of all valid parts in the batch
    """
    B, P = n_pcs.shape
    if n_valids is None:
        n_valids = torch.ones(B, device=n_pcs, dtype=torch.long) * P

    batch_lengths = []
    for b in range(B):
        batch_lengths.append(n_pcs[b, :n_valids[b]])

    batch_length = torch.cat(batch_lengths)
    assert batch_length.shape[0] == torch.sum(n_valids)

    return batch_length