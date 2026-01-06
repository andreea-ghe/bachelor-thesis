import torch
from torch_geometric.nn.pool import knn
from torch_geometric.utils import to_dense_batch
from torch import Tensor


def group_knn_features(all_features: Tensor, all_coords: Tensor, k: int, query_coords: Tensor, batch_idx: Tensor, include_relative_pos: bool) -> Tensor:
    """
    Finds k nearest neighbors for each point and groups their features together.
    This is used for local feature aggregation in the Point Transformer layer.
    Input:
        all_features: features of all points [N, D]
        all_coords: coordinates of all points [N, 3]
        k: number of nearest neighbors to find
        query_coords: coordinates of points we want to find neighbors for [N, 3]
        batch_idx: batch assignment per point [N]
        include_relative_pos: whether to append (p_j - p_i)
    """
    device = all_features.device
    N, feat_dim = all_features.shape

    if batch_idx is None:
        batch_idx = torch.zeros(N, dtype=torch.long).to(device) # all points belong to the same batch
    
    if query_coords is None:
        query_coords = all_coords

    # Find k nearest neighbors
    idx = knn(all_features, all_features, k=k, batch_x=batch_idx, batch_y=batch_idx) # [N, k]
    idx, mask = to_dense_batch(idx[1], idx[0], fill_value=N, max_num_nodes=k) # [N, k]
    all_features = torch.cat([all_features, torch.zeros(1, feat_dim).to(device)], dim=0) # add zero padding for invalid indices

    neighbor_idx = idx.view(-1).long() # [N * k]
    neighbor_features = all_features[neighbor_idx, :] # [N * k, D]
    neighbor_features = neighbor_features.view(N, k, feat_dim) # [N, k, D]
    
    if include_relative_pos:
        assert query_coords.is_contiguous() # needs to be contiguous for view()
        
        # all_coords[N] = (0, 0, 0): only invalid indices have position N so we add 0 padding for invalid indices
        all_coords = torch.cat([all_coords, torch.zeros(1, 3).to(device)], dim=0) 
        
        # compute the relative positions of the neighbors regarding the query point
        relative_coords = all_coords[neighbor_idx, :].view(N, k, 3) - query_coords.unsqueeze(1) # [N, k, 3]
        # converts boolean mask to float
        mask = mask.to(torch.float32)
        # zero out invalid positions
        # "n s c" is the shape of relative_coords and "n s" is the shape of mask => we multiply each coordinate by the mask and define the output shape accordingly
        relative_coords = torch.einsum("n s c, n s -> n s c", relative_coords, mask) 
        neighbor_features = torch.cat([relative_coords, neighbor_features], dim=-1) # [N, k, D + 3]

        return neighbor_features, idx

    return neighbor_features, idx


def select_points(points, idx):
    """
    Select specific points from the point cloud based on provided indices.
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device # location of the points tensor
    B = points.shape[0] # get batch size

    # idx.shape = [B, S]
    # view_shape = [B, 1]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
 
    # repeat_shape = [1, S]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points
