from __future__ import annotations

import torch
from torch import nn


class ForgeAnomalyHead(nn.Module):
    def __init__(self, input_dim: int, media: str) -> None:
        super().__init__()
        self.media = media
        pool = nn.AdaptiveAvgPool2d(1) if media == "image" else nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool = pool
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        feature = features[-1]
        pooled = self.pool(feature).flatten(1)
        return self.classifier(pooled).squeeze(-1)
