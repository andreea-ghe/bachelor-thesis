import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


class AffinityDual(nn.Module):
    """
    Primal-Dual Affinity Layer
    Problem it solves:
    Traditional matching uses the same descriptor for all pieces, which can lead to
    self-alignment (pieces matching themselves instead of their counterparts).
    
    Solution:
    Split features into two complementary descriptors:
    - Primal descriptor (first half): represents the "convex" or "outward" surface
    - Dual descriptor (second half): represents the "concave" or "inward" surface
    
    Matching rule:
    Affinity(X, Y) = X_primal * A * Y_dual^T
    
    This ensures pieces only match with their complements, not themselves.
    
    """
    def __init__(self, feature_dim: int):
        super(AffinityDual, self).__init__()
        self.feature_dim = feature_dim
        assert feature_dim % 2 == 0, "Feature dimension must be even to split into primal and dual."
        self.half_dim = feature_dim // 2 # half feature dimension for primal and dual descriptors

        self.A = Parameter(Tensor(self.half_dim, self.half_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize affinity matrix with small random values plus identity.
        Identity initialization provides a good starting point for learning.
        """
        start = 1.0 / self.half_dim**0.5
        self.A.data.uniform_(-start, start) # small random values
        self.A.data += torch.eye(self.half_dim) # add identity matrix

    def forward(self, X, Y):
        """
        The affinity is computed by matching:
        - Primal descriptor of X (first half of features)
        - Dual descriptor of Y (second half of features)
        
        This asymmetric matching ensures complementary geometry matching.
        
        Input:
            X: features from first piece [B, N1, F]
                X[:, :, :hd] = primal descriptor
                X[:, :, hd:] = dual descriptor (not used here)
            Y: features from second piece [B, N2, F]
                Y[:, :, :hd] = primal descriptor (not used here)
                Y[:, :, hd:] = dual descriptor
        """
        assert X.shape[-1] == Y.shape[-1] == self.feature_dim # we expect same feature dimension for consistent splitting

        X_primal = X[:, :, :self.half_dim]  # extract primal descriptor from X (first half) [B, N1, F/2]
        Y_dual = Y[:, :, self.half_dim:]    # extract dual descriptor from Y (second half) [B, N2, F/2]

        # compute affinity matrix: X_primal * A * Y_dual^T
        M = torch.matmul(X_primal, self.A)
        M = torch.matmul(M, Y_dual.transpose(1, 2))

        return M  # [B, N1, N2]
