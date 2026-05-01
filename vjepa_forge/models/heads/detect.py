from __future__ import annotations

import torch
from torch import nn


class ForgeDetectHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, media: str, num_queries: int = 100) -> None:
        super().__init__()
        self.media = media
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, input_dim)
        self.classifier = nn.Linear(input_dim, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 4),
        )

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        feature = features[-1]
        if self.media == "image":
            pooled = feature.mean(dim=(-1, -2))
            queries = self.query_embed.weight.unsqueeze(0) + pooled.unsqueeze(1)
            return {
                "pred_logits": self.classifier(queries),
                "pred_boxes": self.box_head(queries).sigmoid(),
            }
        pooled = feature.mean(dim=(-1, -2)).transpose(1, 2)
        queries = self.query_embed.weight.view(1, 1, self.num_queries, -1) + pooled.unsqueeze(2)
        return {
            "pred_logits": self.classifier(queries),
            "pred_boxes": self.box_head(queries).sigmoid(),
        }
