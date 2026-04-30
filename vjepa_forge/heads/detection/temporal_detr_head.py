from __future__ import annotations

import torch
from torch import nn


class TemporalDETRHead(nn.Module):
    def __init__(self, input_dim: int, num_queries: int, num_classes: int) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, input_dim)
        self.classifier = nn.Linear(input_dim, num_classes + 1)
        self.box_head = nn.Linear(input_dim, 4)

    def forward(self, feature_volume: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = feature_volume.mean(dim=(-1, -2)).mean(dim=2)
        queries = self.query_embed.weight.unsqueeze(0).expand(pooled.shape[0], -1, -1)
        latent = queries + pooled.unsqueeze(1)
        return {"pred_logits": self.classifier(latent), "pred_boxes": self.box_head(latent).sigmoid()}
