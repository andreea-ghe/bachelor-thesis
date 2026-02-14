import torch
from torch import Tensor
import torch.nn as nn

class GaussianRBF(nn.Module):
    def __init__(self, num_bases=16, distance_range=(0.0, 10.0)):
        super().__init__()
        centers = torch.linspace(distance_range[0], distance_range[1], num_bases)
        self.register_buffer("centers", centers)
        self.width = (centers[1] - centers[0]) * 0.5 # width of the Gaussian function; controls smoothness of decision boundary

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: x: scalar values [N]
        Output: rbf features [N, num_bases]
            - each feature represents the similarity between the scalar value and the center of the Gaussian function
            - the features are normalized to sum to 1
        """
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.width)**2)