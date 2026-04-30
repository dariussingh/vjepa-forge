from __future__ import annotations

import torch
from torch import nn


class ImageClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feature_map).flatten(1)
        return self.classifier(pooled)
