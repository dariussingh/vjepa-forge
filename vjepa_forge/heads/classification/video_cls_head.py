from __future__ import annotations

import torch
from torch import nn


class VideoClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, feature_volume: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feature_volume).flatten(1)
        return self.classifier(pooled)
