import torch


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, -1)[:, :, None]  # [B, N, 1]
    dist += torch.sum(dst ** 2, -1)[:, None, :]  # [B, 1, M]

    dist = torch.clamp(dist, min=1e-12, max=None)

    return dist


@torch.no_grad() # this function must not participate in backpropagation
def diagonal_square_matrix(shape, nr_points_piece, nr_valid_pieces, pos_msk=0.0, neg_msk=1000.0, device=None):
    """
    Generate a block-diagonal mask to prevent self-matching.
    Creates a matrix where:
    - Diagonal blocks (within same piece): filled with neg_msk
    - Off-diagonal blocks (different pieces): filled with pos_msk
    
    This prevents points from the same piece from matching each other,
    which is crucial for both matching and label computation.

    Input:
        device: torch.device - device to create the tensor on
        shape: tuple (B, N, N) - desired output shape
        nr_points_piece: [B, P] - number of points per piece in each batch
        nr_valid_pieces: [B] - number of valid pieces in each batch
        pos_msk: float - value for off-diagonal blocks (inter-piece)
        neg_msk: float - value for diagonal blocks (intra-piece)
    """
    device = nr_points_piece.device if device is None else device
    
    B = nr_points_piece.shape[0]

    # cumulative sum that will help us identify piece boundaries
    nr_points_piece_cumsum = torch.cumsum(nr_points_piece, dim=-1)  # [B, P]
    
    if nr_valid_pieces is None:
        # assume all pieces are valid
        P = nr_points_piece_cumsum.shape[-1]
        nr_valid_pieces = torch.tensor([P for _ in range(B)], dtype=torch.long)

    # start with all neg_msk
    msk = torch.ones(shape).to(device) * neg_msk
    for b in range(B):
        nr_pieces = nr_valid_pieces[b] # remove padding pieces that were added for batching

        # set to pos_mask, meaning allowed, everything in this object
        msk[b, : nr_points_piece_cumsum[b, nr_pieces - 1], : nr_points_piece_cumsum[b, nr_pieces - 1]] = pos_msk

        # for every piece, set the diagonal block to unallowed (neg_msk)
        for p in range(nr_pieces):
            start = 0 if p == 0 else nr_points_piece_cumsum[b, p - 1]
            end = nr_points_piece_cumsum[b, p]
            msk[b, start:end, start:end] = neg_msk # set diagonal block to neg_msk

    return msk
