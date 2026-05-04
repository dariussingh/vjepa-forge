from __future__ import annotations

import torch
from torch import nn


class SemanticSegmentationHead(nn.Module):
    """Implements a Forge-native semantic segmentation decoder over V-JEPA feature maps."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(input_dim, num_classes, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.decoder(feature_map)


class VideoSemanticSegmentationHead(nn.Module):
    """Extends Forge semantic decoding to video by predicting per-frame class masks from 3D features."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(input_dim, num_classes, kernel_size=1),
        )

    def forward(self, feature_volume: torch.Tensor) -> torch.Tensor:
        return self.decoder(feature_volume)
