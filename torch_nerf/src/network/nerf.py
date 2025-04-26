"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim

        # Input + 9 FC layers with ReLU, skip at layer 5 (i=4)
        self.layers = nn.ModuleList()
        for i in range(10):  # input + 9 layers
            in_dim = pos_dim if i == 0 else feat_dim
            if i == 5:
                in_dim += pos_dim  # Skip connection
            self.layers.append(nn.Linear(in_dim, feat_dim))

        # Output head: density and feature
        self.density_layer = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.ReLU()
        )
        self.feature_layer = nn.Linear(feat_dim, feat_dim)

        # RGB head
        self.view_fc = nn.Sequential(
            nn.Linear(feat_dim + view_dir_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 3),
            nn.Sigmoid()
        )

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        x = pos
        for i, layer in enumerate(self.layers):
            if i == 5:
                x = torch.cat([x, pos], dim=-1)  # Skip connection
            x = torch.relu(layer(x))

        sigma = self.density_layer(x)         # Shape: [N, 1]
        feature = self.feature_layer(x)       # Shape: [N, feat_dim]

        # RGB output using view direction
        h = torch.cat([feature, view_dir], dim=-1)
        rgb = self.view_fc(h)                 # Shape: [N, 3]

        return sigma, rgb
