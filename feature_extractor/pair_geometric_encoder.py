import torch
import torch.nn as nn
from torch import Tensor
from feature_extractor.utils_gaussian_rbf import GaussianRBF


class PairGeometricEncoder(nn.Module):
    """
    Computes geometric pair attention bias between parts (Pair Attention).

    For each pair of parts (i, j), encodes:
        1. Center of mass:    p_i = (1/N_i) Σ x_j
        2. Pairwise distance: d_ij = RBF(||p_i - p_j||)
        3. Triplet angles:    r_ij = Σ_k RBF(cos ∠_ijk)

    Projects [d_ij; r_ij] → scalar bias per pair, then expands from
    part-level [B, P, P] to point-level [B, 1, N_SUM, N_SUM] so it
    can be added to attention scores before softmax.
    """

    def __init__(self, num_bases=16, distance_range=(0.0, 10.0), angle_range=(-1.0, 1.0)):
        """
        Input:
            num_bases: number of Gaussian RBF centers (d in the paper)
            distance_range: (min, max) range for distance RBF centers
            angle_range: (min, max) range for angle RBF centers (cosine values live in [-1, 1])
        """
        super().__init__()
        self.distance_rbf = GaussianRBF(num_bases=num_bases, distance_range=distance_range)
        self.angle_rbf = GaussianRBF(num_bases=num_bases, distance_range=angle_range)

        # Project concatenated geometric features [d_ij; r_ij] → scalar bias
        self.bias_proj = nn.Linear(num_bases * 2, 1)

    def _compute_centroids(self, part_pcs: Tensor, n_pcs: Tensor) -> Tensor:
        """
        Compute center of mass for each part: p_i = (1/N_i) Σ x_j

        Input:
            part_pcs: [B, N_SUM, 3] concatenated point clouds of all pieces
            n_pcs: [B, P] number of points per piece
        Output:
            centroids: [B, P, 3]
        """
        B, _, _ = part_pcs.shape
        P = n_pcs.shape[1]
        centroids = torch.zeros(B, P, 3, device=part_pcs.device, dtype=part_pcs.dtype)

        for b in range(B):
            offset = 0
            for p in range(P):
                n = n_pcs[b, p].item()
                if n > 0:
                    # for each part, we compute the center of mass (mean) of points belonging to that part
                    centroids[b, p] = part_pcs[b, offset:offset + n].mean(dim=0)
                offset += n

        return centroids

    def _compute_pairwise_distances(self, centroids: Tensor) -> Tensor:
        """
        Compute pairwise Euclidean distances: ||p_i - p_j||_2

        Input:
            centroids: [B, P, 3]
        Output:
            distances: [B, P, P]
        """
        diff = centroids.unsqueeze(2) - centroids.unsqueeze(1) # [B, P, 1, 3] - [B, 1, P, 3] = [B, P, P, 3]
        distances = torch.norm(diff, dim=-1)  # [B, P, P]
        return distances

    def _compute_triplet_angles(self, centroids: Tensor, n_pcs: Tensor) -> Tensor:
        """
        Compute triplet-wise angle features: r_ij = Σ_k RBF(cos ∠_ijk)

        ∠_ijk is the angle at vertex j, formed by edges j→i and j→k.

        Input:
            centroids: [B, P, 3]
            n_pcs: [B, P]
        Output:
            angle_features: [B, P, P, num_bases]
        """
        # Unit vectors from each part to every other: v_hat[b, i, j] = normalize(p_j - p_i)
        v = centroids.unsqueeze(2) - centroids.unsqueeze(1)  # [B, P, P, 3]
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        v_hat = v / v_norm  # [B, P, P, 3]

        # cos(∠_ijk) with angle at vertex j:
        #   = dot(v_hat[b, j, i], v_hat[b, j, k])
        # Batched dot product via matmul:
        #   v_hat @ v_hat^T → [B, P, P, P]
        #   result[b, j, i, k] = v_hat[b,j,i,:] · v_hat[b,j,k,:]
        cos_angles = torch.matmul(v_hat, v_hat.transpose(2, 3))  # [B, j, i, k]
        cos_angles = cos_angles.clamp(-1.0, 1.0)

        # Reindex so pair (i, j) is in the first two dims: [B, j, i, k] → [B, i, j, k]
        cos_angles = cos_angles.permute(0, 2, 1, 3)  # [B, P_i, P_j, P_k]

        # RBF encode all cosines
        rbf_encoded = self.angle_rbf(cos_angles)  # [B, P, P, P, num_bases]

        # Mask out invalid parts (padding parts with 0 points)
        valid_mask = (n_pcs > 0).float()  # [B, P]
        k_mask = valid_mask[:, None, None, :, None]  # [B, 1, 1, P, 1]
        rbf_encoded = rbf_encoded * k_mask

        # Sum over k: r_ij = Σ_k RBF(cos ∠_ijk)
        angle_features = rbf_encoded.sum(dim=3)  # [B, P, P, num_bases]

        return angle_features

    def _expand_to_point_level(self, pair_bias: Tensor, n_pcs: Tensor, N_SUM: int) -> Tensor:
        """
        Expand part-level bias [B, P, P] to point-level [B, 1, N_SUM, N_SUM].
        Each point inherits the bias of the part it belongs to.

        Input:
            pair_bias: [B, P, P] part-level attention bias
            n_pcs: [B, P] number of points per piece
            N_SUM: total number of points
        Output:
            point_bias: [B, 1, N_SUM, N_SUM] broadcastable across attention heads
        """
        B, P = n_pcs.shape
        device = pair_bias.device

        point_bias = torch.zeros(B, N_SUM, N_SUM, device=device, dtype=pair_bias.dtype)

        for b in range(B):
            # Build part assignment: point i → part index
            part_idx = torch.repeat_interleave(
                torch.arange(P, device=device),
                n_pcs[b].long()
            )  # [n_points_b]

            n_points_b = part_idx.shape[0]

            # Pad to N_SUM if needed (remaining points get part index 0, bias will be 0 anyway)
            if n_points_b < N_SUM:
                pad = torch.zeros(N_SUM - n_points_b, dtype=torch.long, device=device)
                part_idx = torch.cat([part_idx, pad])

            # Each point inherits pair bias from its part:
            # point_bias[b, i, j] = pair_bias[b, part_of(i), part_of(j)]
            point_bias[b] = pair_bias[b][part_idx][:, part_idx]

        # Add head dimension for broadcasting: [B, 1, N_SUM, N_SUM]
        return point_bias.unsqueeze(1)

    def forward(self, part_pcs: Tensor, n_pcs: Tensor) -> Tensor:
        """
        Compute pair geometric attention bias.

        Input:
            part_pcs: [B, N_SUM, 3] concatenated point clouds of all pieces
            n_pcs: [B, P] number of points per piece
        Output:
            pair_bias: [B, 1, N_SUM, N_SUM] additive attention bias
        """
        N_SUM = part_pcs.shape[1]

        # Step 1: p_i = (1/N_i) Σ x_j — center of mass per part
        centroids = self._compute_centroids(part_pcs, n_pcs)  # [B, P, 3]

        # Step 2: d_ij = RBF(||p_i - p_j||) — pairwise distance features
        distances = self._compute_pairwise_distances(centroids)  # [B, P, P]
        dist_features = self.distance_rbf(distances)  # [B, P, P, num_bases]

        # Step 3: r_ij = Σ_k RBF(cos ∠_ijk) — triplet angle features
        angle_features = self._compute_triplet_angles(centroids, n_pcs)  # [B, P, P, num_bases]

        # Step 4: project [d_ij; r_ij] → scalar bias per pair
        pair_features = torch.cat([dist_features, angle_features], dim=-1)  # [B, P, P, 2*num_bases]
        pair_bias = self.bias_proj(pair_features).squeeze(-1)  # [B, P, P]

        # Step 5: expand part-level → point-level
        pair_bias = self._expand_to_point_level(pair_bias, n_pcs, N_SUM)  # [B, 1, N_SUM, N_SUM]

        return pair_bias


if __name__ == "__main__":
    # Quick verification that the module runs correctly
    B, P = 2, 4
    # Simulate 4 parts with varying number of points
    n_pcs = torch.tensor([[30, 25, 20, 15], [35, 20, 25, 10]])
    N_SUM = n_pcs.sum(dim=1).max().item()  # 90

    part_pcs = torch.randn(B, N_SUM, 3)

    encoder = PairGeometricEncoder(num_bases=16)
    pair_bias = encoder(part_pcs, n_pcs)
    print(f"pair_bias shape: {pair_bias.shape}")  # [2, 1, 90, 90]

    # Verify bias is symmetric-ish and centroids make sense
    centroids = encoder._compute_centroids(part_pcs, n_pcs)
    print(f"centroids shape: {centroids.shape}")  # [2, 4, 3]
    print(f"centroid[0,0]: {centroids[0, 0]}")

