import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import fps, knn
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

class PointNetEncoder(nn.Module):
    """Encodes point features by multi-scale grouping based on geometric distances."""
    def __init__(self, ratio, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetEncoder, self).__init__()
        self.ratio = ratio # ratio to downsample points
        self.radius_list = radius_list # list of radii for each scale
        self.nsample_list = nsample_list # number of samples for each scale
        self.conv_blocks = nn.ModuleList() # list of convolutional layers for each scale
        self.bn_blocks = nn.ModuleList() # list of batch normalization layers for each scale
        
        for i in range(len(mlp_list)):
            convolutions = nn.ModuleList()
            batch_norms = nn.ModuleList()
            current_channel = in_channel + 3
            
            for out_channel in mlp_list[i]:
                # we use 1x1 convolutions to simulate MLPs
                convolutions.append(nn.Conv2d(current_channel, out_channel, 1))
                batch_norms.append(nn.BatchNorm2d(out_channel))
                current_channel = out_channel
            
            self.conv_blocks.append(convolutions)
            self.bn_blocks.append(batch_norms)

    def forward(self, xyz, points, piece_id):
        """
        Encoder computes strong features on a small set of centroid points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            piece_id: piece id of each point, [B, 1, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # [B, N, C]
        piece_id = piece_id.permute(0, 2, 1) # [B, N, 1]
        if points is not None:
            points = points.permute(0, 2, 1) # [B, N, D]

        B, N, C = xyz.shape
        assert B == 1

        # sample points using piece-aware farthest point sampling (it selects a subset of points that are as far apart from each other as possible)
        # uniform spatial coverage
        # avoids clustering in dense regions
        # preserves global shape with fewer points
        # xyz shape should be just [num_points, num_features]
        # we only consider in a batch the points from the same piece
        # this ensures every fragment gets its own centroids
        # .unsqueeze(0) → [1, S]
        centroids = fps(xyz[0, :, :], batch=piece_id.reshape(B * N), ratio=self.ratio).unsqueeze(0) # [B, S, C]
        S = centroids.shape[1] # number of sampled points

        # select centrioids' xyz coordinates
        centroids_xyz = select_points(xyz, centroids) # [B, S, C]
        # select metadata for the sampled points: piece id
        centroids_piece_id = select_points(piece_id, centroids) # [B, S, 1]

        scale_features = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i] # number of samples in each local region
            # find neighboring points for each centroid within the specified radius on the same piece only
            neighborhood_idx = knn(xyz[0, :, :], centroids_xyz[0, :, :], k=K, batch_x=piece_id.reshape(-1), batch_y=centroids_piece_id.reshape(-1)) # [S, K]
            # groups neighbors by centroid
            # fill missing neighbors with N
            neighborhood_idx = to_dense_batch(neighborhood_idx[1], neighborhood_idx[0], fill_value=-1, max_num_nodes=K)[0].unsqueeze(0) # [B, S, K]

            # replace invalid indices (-1) with the first neighbor's index
            neighborhood_first = neighborhood_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K]) # [B, S, K]
            mask = neighborhood_idx == -1
            neighborhood_idx[mask] = neighborhood_first[mask]

            neighborhood_xyz = select_points(xyz, neighborhood_idx) # [B, S, K, C]
            # removes global position
            # convert to relative coordinates
            # enforces translation invariance
            # encodes local geometry
            neighborhood_xyz -= centroids_xyz.view(B, S, 1, C) # translate to local coordinates
            if points is not None:
                neighborhood_points = select_points(points, neighborhood_idx) # [B, S, K, D]
                # concatenate point features with relative coordinates
                neighborhood_points = torch.cat([neighborhood_points, neighborhood_xyz], dim=-1) # [B, S, K, D + C]
            else:
                neighborhood_points = neighborhood_xyz # [B, S, K, C]

            neighborhood_points = neighborhood_points.permute(0, 3, 2, 1) # [B, D + C, K, S]
            for j in range(len(self.conv_blocks[i])):
                convolution = self.conv_blocks[i][j]
                batch_norm = self.bn_blocks[i][j]
                neighborhood_points = F.relu(batch_norm(convolution(neighborhood_points))) # [B, D', K, S]
            # max pooling over K neighbors
            pooled_points = torch.max(neighborhood_points, 2)[0] # [B, D', S]
            scale_features.append(pooled_points)

        centroids_xyz = centroids_xyz.permute(0, 2, 1) # [B, C, S]
        centroids_piece_id = centroids_piece_id.permute(0, 2, 1) # [B, 1, S]
        features_concat = torch.cat(scale_features, dim=1) # [B, D'', S]

        return centroids_xyz, centroids_piece_id, features_concat
    

class PointNetDecoder(nn.Module):
    """Decodes point features by feature propagation layers."""
    def __init__(self, in_channel, mlp):
        super(PointNetDecoder, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        current_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(current_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            current_channel = out_channel

    def forward(self, fine_xyz, coarse_xyz, fine_piece_id, coarse_piece_id, fine_features, coarse_features):
        """
        Decoder upsamples points' features by interpolating from centroid points.
        Input:
            fine_xyz: input points position data, [B, C, N]
            coarse_xyz: centroid points position data, [B, C, S]
            fine_piece_id: piece id of input points, [B, N]
            coarse_piece_id: piece id of centroid points, [B, S]
            fine_features: input points data, [B, D, N]
            coarse_features: centroid points data, [B, D', S]
        Return:
            new_points: upsampled points data, [B, D'', N]
        """
        fine_xyz = fine_xyz.permute(0, 2, 1) # [B, N, C]
        coarse_xyz = coarse_xyz.permute(0, 2, 1) # [B, S, C]
        fine_piece_id = fine_piece_id.permute(0, 2, 1) # [B, N, 1]
        coarse_piece_id = coarse_piece_id.permute(0, 2, 1) # [B, S, 1]
        if fine_features is not None:
            fine_features = fine_features.permute(0, 2, 1) # [B, N, D]
        coarse_features = coarse_features.permute(0, 2, 1) # [B, S, D']

        B, N, C = fine_xyz.shape
        _, S, _ = coarse_xyz.shape

        if S == 1: # if there is only one centroid, replicate its features for all points
            interpolated_points = coarse_features.repeat(1, N, 1) # [B, N, D']
        else: 
            neighborhood_idx = knn(coarse_xyz[0, :, :], fine_xyz[0, :, :], k=3, batch_x=coarse_piece_id.reshape(-1), batch_y=fine_piece_id.reshape(-1))
            
            # for each fine point get the indices of its 3 nearest centroids
            idx = to_dense_batch(neighborhood_idx[1], neighborhood_idx[0], fill_value=-1, max_num_nodes=3)[0].unsqueeze(0) # [B, N, 3]
            # replace invalid indices (-1) with the first neighbor's index
            neighborhood_first = idx[:, :, 0].view(B, N, 1).repeat([1, 1, 3]) # [B, N, 3]
            mask = idx == -1
            idx[mask] = neighborhood_first[mask]

            distances = fine_xyz[:, neighborhood_idx[0], :] ** 2 + coarse_xyz[:, neighborhood_idx[1], :] ** 2
            distances -= 2 * fine_xyz[:, neighborhood_idx[0], :] * coarse_xyz[:, neighborhood_idx[1], :]
            distances = torch.sum(distances, -1).reshape(-1)
            distances = to_dense_batch(distances, neighborhood_idx[0], fill_value=1e8, max_num_nodes=3)[0].unsqueeze(0) # [B, N, 3]
            
            dist_inverse = 1.0 / (distances + 1e-8) # if centroid is close, weight is high
            norm = torch.sum(dist_inverse, dim=2, keepdim=True) # normalize such that weights sum to 1
            weight = dist_inverse / norm
            interpolated_points = torch.sum(select_points(coarse_features, idx) * weight.view(B, N, 3, 1), dim=2)

        if fine_features is not None: # refine already existing features with skip connections
            interpolated_points = torch.cat([interpolated_points, fine_features], dim=-1) # [B, N, D + D']

        interpolated_points = interpolated_points.permute(0, 2, 1) # [B, D'', N]
        for i in range(len(self.mlp_convs)):
            convolution = self.mlp_convs[i]
            batch_norm = self.mlp_bns[i]
            interpolated_points = F.relu(batch_norm(convolution(interpolated_points))) # [B, D''', N]
        
        return interpolated_points