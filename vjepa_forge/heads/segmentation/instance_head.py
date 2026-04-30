from __future__ import annotations

import torch
from torch import nn


class InstanceSegmentationHead(nn.Module):
    def __init__(self, input_dim: int, num_queries: int, num_classes: int) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, input_dim)
        self.classifier = nn.Linear(input_dim, num_classes + 1)
        self.mask_proj = nn.Conv2d(input_dim, num_queries, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = feature_map.mean(dim=(-1, -2))
        queries = self.query_embed.weight.unsqueeze(0).expand(pooled.shape[0], -1, -1)
        logits = self.classifier(queries + pooled.unsqueeze(1))
        masks = self.mask_proj(feature_map)
        return {"pred_logits": logits, "pred_masks": masks}
