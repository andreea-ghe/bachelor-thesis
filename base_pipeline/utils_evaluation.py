import torch
from pytorch3d.loss import chamfer_distance

from .utils_transform import transform_point_clouds


@torch.no_grad()
def part_acc_and_cd(point_cloud, predicted_trans, gt_trans, predicted_rot, gt_rot, valid_pieces):
    """
    Compute part-wise accuracy and Chamfer Distance for transformed point clouds.

    1. Part Accuracy (PA): Percentage of correctly assembled pieces
    A piece is considered "correct" if:
    - Chamfer Distance between predicted and ground truth pose < 0.01
    
    2. Chamfer Distance (CD): Average point cloud distance
    Chamfer Distance measures the average distance between point clouds:
    CD(A, B) = mean(min_dist(a→B)) + mean(min_dist(b→A))
    """
    B, P = point_cloud.shape[:2]

    # transform point clouds using predicted and ground truth poses
    pred_pc = transform_point_clouds(point_cloud, predicted_trans, predicted_rot)  # [B, P, N, 3] - predicted poses
    gt_pc = transform_point_clouds(point_cloud, gt_trans, gt_rot)  # [B, P, N, 3] - ground truth poses

    # flatten to compute pairwise distances
    pred_pc = pred_pc.flatten(0, 1)  # [B*P, N, 3]
    gt_pc = gt_pc.flatten(0, 1)      # [B*P, N, 3]

    # compute chamfer distance between predicted and ground truth pieces
    cd_per_piece, _ = chamfer_distance(pred_pc, gt_pc, batch_reduction=None, point_reduction='mean')  # [B*P]
    cd_per_piece = cd_per_piece.view(B, P).type_as(point_cloud)  # [B, P]

    # part accuracy threshold: CD < 0.01
    threshold = 0.01
    accuracy = (cd_per_piece < threshold) & (valid_pieces == 1)

    # compute per object accuracy (not per part)
    accuracy = accuracy.sum(-1) / valid_pieces.sum(-1)  # [B]
    cd_mean = cd_per_piece.sum(-1) / (valid_pieces == 1).sum(-1)

    return accuracy, cd_mean


@torch.no_grad()
def trans_metrics(predicted_trans, gt_trans, valid_pieces, metric):
    """
    Evaluation metrics for transformation.
    
    Input:
        predicted_trans: [B, P, 3] - predicted translations
        gt_trans: [B, P, 3] - ground truth translations
        valid_pieces: [B, P] - mask indicating valid pieces
        metric: str - 'mse', 'rmse', or 'mae'

    Output:
        metric_per_data: [B] - computed metric per batch item
    """
    assert metric in ['mse', 'rmse', 'mae']
    
    if metric == 'mse':
        metric_per_data = (predicted_trans - gt_trans).pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = (predicted_trans - gt_trans).pow(2).mean(dim=-1) ** 0.5  # [B, P]
    else:
        metric_per_data = (predicted_trans - gt_trans).abs().mean(dim=-1)
        
    # mask invalid pieces so they don't affect the metric
    metric_per_data = average_loss(metric_per_data, valid_pieces)  # [B]
    return metric_per_data


@torch.no_grad()
def rot_metrics(predicted_rot, gt_rot, valid_pieces, metric):
    """
    Evaluation metrics for rotation in euler angle (degree) space.

    Input:
        predicted_rot: [B, P, 4/(3, 3)] - predicted rotations (quaternion or rotation matrix)
        gt_rot: [B, P, 4/(3, 3)] - ground truth rotations (quaternion or rotation matrix)
        valid_pieces: [B, P] - mask indicating valid pieces
        metric: str - 'mse', 'rmse', or 'mae'

    Output:
        metric_per_data: [B] - computed metric per batch item
    """
    assert metric in ['mse', 'rmse', 'mae']

    pred_deg = predicted_rot.to_euler(to_degree=True) # [B, P, 3]
    gt_deg = gt_rot.to_euler(to_degree=True)

    # since angles loop around at 360 degrees
    naive_angular_diff = (pred_deg - gt_deg).abs()
    wrap_around_diff = 360.0 - naive_angular_diff

    min_angular_diff  = torch.minimum(naive_angular_diff, wrap_around_diff)  # [B, P, 3]

    if metric == 'mse':
        metric_per_data = min_angular_diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = min_angular_diff.pow(2).mean(dim=-1) ** 0.5  # [B, P]
    else:
        metric_per_data = min_angular_diff.mean(dim=-1)  # [B, P]

    metric_per_data = average_loss(metric_per_data, valid_pieces)  # [B]
    return metric_per_data


def average_loss(loss_per_piece, valid_pieces):
    """
    Average loss values according to the valid parts.
    
    Since objects have variable numbers of pieces (2-20), and batches are padded
    to max_num_part, we need to mask out the padded entries when computing loss.
  
    Input:
        loss_per_piece: [B, P] - loss values per piece
        valid_pieces: [B, P] - binary mask indicating valid pieces (1) and padded pieces (0)

    Output:
        loss: [B] - averaged loss per batch item

    Example:
    loss_per_piece =
    [[0.2, 0.4, 0.3, 0.0, 0.0],
    [0.1, 0.6, 0.5, 0.2, 0.3]]

    valid_pieces =
    [[1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1]]

    (loss_per_piece * valid_pieces).sum(1) = loss_{b,p} ​⋅ valid_{b,p} = [0.9, 1.7]
    valid_pieces.sum(1) = [3, 5]
    """
    if valid_pieces is None:
        loss = loss_per_piece.sum(1) / loss_per_piece.shape[1]
    else:
        valid_pieces = valid_pieces.float().detach() # detach = treat the mask as a constant, not something the model could optimize
        loss = (loss_per_piece * valid_pieces).sum(1) / valid_pieces.sum(1)

    return loss
