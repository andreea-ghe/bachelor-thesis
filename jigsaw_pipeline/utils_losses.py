import torch
import torch.nn.functional as F
import itertools
from .utils_pairwise_alignment import pairwise_alignment


def permutation_loss(pred_matching, gt_matching, n_source, n_target):
    """
    Matching Loss (Permutation Loss) between predicted and ground truth correspondence matrices.
    It uses binary cross-entropy to supervise the soft matching matrix produced by the Sinkhorn algorithm.
    
    The loss encourages the predicted matching matrix to match the ground truth matching matrix,
    which is computed based on nearest neighbors in the original (untransformed) point clouds.

    L_mat = -∑_{i,j} x_ij^gt log(x̂_ij) + (1-x_ij^gt)log(1-x̂_ij)

    This loss is applied starting from epoch 0, after the segmentation module has had time to learn.
    
    Input:
        pred_matching: [B, N_source, N_target] - predicted soft correspondence matrix from Sinkhorn
        gt_matching: [B, N_source, N_target] - ground truth correspondence matrix
        n_source: [B] - number of source points in each batch
        n_target: [B] - number of target points in each batch
    Output:
        L_mat: matching loss
    """
    B = pred_matching.shape[0]
    pred_matching_fp32 = pred_matching.to(dtype=torch.float32) 

    # make sure matrices represent probabilities
    try:
        assert torch.all((gt_matching >= 0) * (gt_matching <= 1))
        assert torch.all((pred_matching_fp32 >= 0) * (pred_matching_fp32 <= 1))
    except AssertionError as e:
        print("Error in permutation_loss: gt_matching and pred_matching_fp32 must be in [0, 1]")
        raise e

    loss = torch.tensor(0.0).to(pred_matching_fp32.device) # accumulated loss over batch
    total_num_points = torch.zeros_like(loss) # normalization term, because different batches may have different number of points

    # accumulate loss over batches
    for b in range(B):
        batch_slice = [b, slice(n_source[b]), slice(n_target[b])]
        # compute loss for current batch slice
        loss += F.binary_cross_entropy(pred_matching_fp32[batch_slice], gt_matching[batch_slice], reduction='sum')
        total_num_points += n_source[b].to(total_num_points.dtype).to(pred_matching_fp32.device)

    # average over total number of points
    avg_loss = loss / total_num_points

    return avg_loss


def rigidity_loss(n_pcs, n_valid, gt_pcs, part_pcs, n_critical_pcs, critical_pcs_idx, ds_mat):
    """
    It enforces that matched fracture points should align geometrically after applying the optimal
    transformation between pieces.
    For each pair of matches pieces (Pi, Pj) we compute the optimal rigid transformation T_ij using 
    RANSAC on matched points. We then measure how well the transformed points from Pi align with
    matched points in Pj. We penalize misalignment.

    T̂_ij = argmin_{T_ij} ∑_p ||T_ij(p) - p̄||₂
    L_rig = ∑_{i<j≤n} ℛ_ij, where ℛ_ij measures alignment error

    This loss is applied starting from epoch 199.

    Input:
        n_pcs: [B, P] - number of points per piece in each batch
        n_valid: [B] - number of valid pieces in each batch
        gt_pcs: [B, N_SUM, 3] - ground truth point clouds
        part_pcs: [B, N_SUM, 3] - point clouds for each piece in the batch
        n_critical_pcs: [B, P] - number of critical fracture points per piece
        critical_pcs_idx: [B, N_SUM] - indices of critical fracture points
        ds_mat: [B, N_SUM, N_SUM] - soft correspondence matrix between all

    Output:
        L_rig: rigidity loss
    """
    B, N, _ = gt_pcs.shape
    loss = torch.tensor(0.0).to(ds_mat.device)  # accumulated loss over batch

    # compute cumulative sums for indexing in concatenated point clouds
    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)  # [B, P]
    n_critical_pcs_cumsum = torch.cumsum(n_critical_pcs, dim=-1)  # [B, P]
    n_sum = torch.zeros_like(loss)

    for b in range(B):
        sum_full_matched = torch.sum(ds_mat[b]) # total matching confidence for all pairs in batch b

        for idx1, idx2 in itertools.combinations(torch.arange(n_valid[b]), 2):
            # get indices for fracture points of piece 1
            critical_start1 = 0 if idx1 == 0 else n_critical_pcs_cumsum[b, idx1 - 1]
            critical_end1 = n_critical_pcs_cumsum[b, idx1]

            # get indices for fracture points of piece 2
            critical_start2 = 0 if idx2 == 0 else n_critical_pcs_cumsum[b, idx2 - 1]
            critical_end2 = n_critical_pcs_cumsum[b, idx2]

            # get indices for all points of piece 1
            pc_start1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
            pc_end1 = n_pcs_cumsum[b, idx1]

            # get indices for all points of piece 2
            pc_start2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
            pc_end2 = n_pcs_cumsum[b, idx2]

            n1 = n_critical_pcs[b, idx1]
            n2 = n_critical_pcs[b, idx2]

            if n1 == 0 or n2 == 0:
                continue  # skip if no critical points in either piece

            # Extract the soft matching probabilities between fracture points of piece i and piece j.
            # ds_mat contains point-to-point matching probabilities (Sinkhorn output).
            # We consider both directions:
            #   - i -> j  (piece i matching to piece j)
            #   - j -> i  (piece j matching to piece i)
            # and symmetrize them to enforce mutual agreement.
            match_submatrix = ds_mat[b, critical_start1:critical_end1, critical_start2:critical_end2] + ds_mat[b, critical_start2:critical_end2, critical_start1:critical_end1].transpose(1, 0)
            # measures how confident the model is that piece i and piece j belong together
            n_matches = torch.sum(match_submatrix)

            # Convert the matching matrix to NumPy for geometric alignment.
            ds_mat_d = ds_mat.detach().cpu().numpy()
            match_submatrix_d = ds_mat_d[b, critical_start1:critical_end1, critical_start2:critical_end2] + ds_mat_d[b, critical_start2:critical_end2, critical_start1:critical_end1].transpose(1, 0)

            # skip pairs with no matching if there are other matches in the object
            if n_valid[b] > 2 and n_matches == 0 and sum_full_matched > 0:
                continue

            # get point coulds for both pieces
            pc1 = part_pcs[b, pc_start1:pc_end1]  # [n1, 3]
            pc2 = part_pcs[b, pc_start2:pc_end2]  # [n2, 3]

            if critical_pcs_idx is not None:
                # extract fracture points
                critical_pcs_source = pc1[critical_pcs_idx[b, pc_start1:pc_start1 + n1]]  # [n1, 3]
                critical_pcs_target = pc2[critical_pcs_idx[b, pc_start2:pc_start2 + n2]]  # [n2, 3]

                # compute optimal rigid transformation using RANSAC
                rot, trans = pairwise_alignment(critical_pcs_source.cpu().numpy(), critical_pcs_target.cpu().numpy(), match_submatrix_d)
                rot = torch.tensor(rot, dtype=torch.float32, device=match_submatrix.device).reshape(3, 3)
                trans = torch.tensor(trans, dtype=torch.float32, device=match_submatrix.device).reshape(3)

                # apply transformation to all points in source piece
                new_critical_pcs_source = torch.matmul(rot, critical_pcs_source.transpose(1, 0)).transpose(1, 0)
                new_critical_pcs_source += trans
                # weight transformed points by matching probabilities: if a point has no match, it should not contribute to the loss
                new_critical_pcs_source = new_critical_pcs_source * torch.sum(match_submatrix, dim=-1).reshape(-1, 1)
                
                # compute weighted average of target points
                new_critical_pcs_target = torch.matmul(match_submatrix, critical_pcs_target)  # [n1, 3]

                pair_loss = new_critical_pcs_source ** 2 + new_critical_pcs_target ** 2 - 2 * new_critical_pcs_source * new_critical_pcs_target
                pair_loss = torch.sum(pair_loss)
                n_sum += critical_pcs_source.shape[0]

                # weight by total matching probability: stronger matches should contribute more to the loss
                loss += pair_loss * n_matches

    return loss / n_sum # average over total number of matched critical points  
