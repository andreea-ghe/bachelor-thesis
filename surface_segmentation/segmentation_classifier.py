import torch
import torch.nn as nn
from surface_segmentation.utils import square_distance, diagonal_square_matrix


class SegmentationClassifier(nn.Module):
    def __init__(self, model_point: str, pc_feat_dim: int, num_classes: int):
        super().__init__()
        self.model_point = model_point

        output_dim = 1 if model_point == "binary" else num_classes
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(pc_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(pc_feat_dim, output_dim, 1)
        )

    def forward(self, feat_bfn):
        return self.classifier(feat_bfn)

@torch.no_grad()
def compute_label(points, nr_points_piece, nr_valid_pieces, dist_thresholds):
    """
    Compute ground truth labels for fracture surface segmentation.
    A point is labeled as a fracture point if it is close to any point
    from a different piece.

    Input:
        points: [B, N, 3] - concatenated point clouds from all pieces
        nr_points_piece: [B, P] - number of points per piece in each batch
        nr_valid_pieces: [B] - number of valid pieces in each batch
        dist_thresholds: list of float - distance thresholds for labeling

    Output:
        all_labels: [B, N] - binary labels for each point (1: fracture, 0: non-fracture)
    """
    B, N, _ = points.shape

    # Compute pairwise distances between all points
    distances = torch.sqrt(square_distance(points, points))  # [B, N, N]

    # Mask out points from the same piece
    # ensure that nearest neighbor comes from a different piece
    neg_mask = diagonal_square_matrix(
        shape=(B, N, N),
        nr_points_piece=nr_points_piece,
        nr_valid_pieces=nr_valid_pieces,
        pos_msk=0.0,
        neg_msk=1e6,
        device=points.device
    )
    distances = distances + neg_mask

    # Find minimum distance to any point in a different piece
    min_distance, _ = torch.min(distances, dim=-1)  # [B, N]
    min_distance = min_distance.reshape(B, N)

    # Label as fracture if distance < threshold
    labels = (min_distance < dist_thresholds).to(torch.int64)
    
    return labels

def get_fracture_points_from_label(n_pcs, critical_labels):
    """
    Given critical point labels for all points in the concatenated point clouds,
    compute the number of fracture points per piece and their indices.

    Input:
        n_pcs: [B, P] - number of points in each piece
        critical_labels: [B, N_SUM] - binary labels for each point (1: fracture, 0: non-fracture)

    Output:
        n_critical_pcs: [B, P] - number of fracture points per piece
        critical_pcs_idx: [B, N_SUM] - indices of fracture points in each piece
    """
    B, N_SUM = critical_labels.shape
    P = n_pcs.shape[-1]

    # compute cumulative sum of number of points per piece to identify indices for each piece
    n_pcs_cumsum = torch.cumsum(n_pcs, dim=1).to(torch.int64) # [B, P]

    n_critical_pcs = torch.zeros_like(n_pcs)  # [B, P] number of fracture points per piece
    critical_pcs_idx = torch.zeros_like(critical_label).to(torch.int64)  # [B, N_SUM] indices of fracture points in each piece

    for b in range(B): # for each object in the batch
        for p in range(P): # for each piece

            # we find the start and end indices of this piece in the concatenated point cloud
            start_idx = 0 if p == 0 else n_pcs_cumsum[b, p - 1]
            end_idx = n_pcs_cumsum[b, p]

            piece_labels = critical_labels[b, start_idx:end_idx]  # get labels for this piece for each point 
            fracture_point_indices = piece_labels.nonzero().reshape(-1)# indices of fracture points in this piece

            nr_fracture_points = fracture_point_indices.shape[0]
            n_critical_pcs[b, p] = nr_fracture_points # number of fracture points in this piece
            critical_pcs_idx[b, start_idx:start_idx + nr_fracture_points] = fracture_point_indices # store indices of fracture points

    return n_critical_pcs, critical_pcs_idx