import torch


def get_features_of_fracture_points(features, nr_critical_points, critical_label, B, N, F, device=None):
    """
    Extract features of fracture points based on segmentation labels.
    After segmentation, we only use fracture points for matching.
    This function gathers the features of points labeled as fracture points.

    Input:
        features: [B, N, F] - features of all points
        nr_critical_points: [B] - number of fracture points in each batch
        critical_label: [B, N] - binary labels indicating fracture points
        B: int - batch size
        N: int - max number of fracture points across batch
        F: int - feature dimension

    Output:
        critical_features: [B, N, F] - features of fracture poitns
    """
    critical_features = torch.zeros(B, N, F, device=device, dtype=features.dtype)
    
    for b in range(B):
        # select only features where label is 1 (fracture points)
        critical_features[b, :nr_critical_points[b]] = features[b, critical_label[b] == 1]

    return critical_features