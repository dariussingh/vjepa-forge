from __future__ import annotations

import torch
from torch import nn


class TemporalFeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(input_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, hidden_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, feature_volume: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(feature_volume)))


class TemporalDETRHead(nn.Module):
    def __init__(self, input_dim: int, num_queries: int, num_classes: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or input_dim
        self.adapter = TemporalFeatureAdapter(input_dim, hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.query_embed = nn.Embedding(num_queries, hidden)
        self.classifier = nn.Linear(hidden, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
        )

    def forward(self, feature_volume: torch.Tensor) -> dict[str, torch.Tensor]:
        adapted = self.adapter(feature_volume)
        batch_size, hidden_dim, num_frames, height, width = adapted.shape
        query = self.query_embed.weight.unsqueeze(0).expand(batch_size * num_frames, -1, -1)
        memory = adapted.permute(0, 2, 3, 4, 1).reshape(batch_size * num_frames, height * width, hidden_dim)
        decoded = self.decoder(query, memory).reshape(batch_size, num_frames, -1, hidden_dim)
        return {
            "pred_logits": self.classifier(decoded),
            "pred_boxes": self.box_head(decoded).sigmoid(),
        }
