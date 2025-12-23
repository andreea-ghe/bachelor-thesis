import torch
import torch.nn as nn
from surface_segmentation.utils import square_distance, diagonal_square_matrix


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
