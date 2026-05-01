from __future__ import annotations

import torch
from torch import nn

from vjepa_forge.heads.segmentation import InstanceSegmentationHead, SemanticSegmentationHead


class ForgeSegmentHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, media: str) -> None:
        super().__init__()
        self.media = media
        self.image_head = SemanticSegmentationHead(input_dim, num_classes)
        self.video_head = InstanceSegmentationHead(input_dim, num_queries=16, num_classes=num_classes)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        feature = features[-1]
        if self.media == "image":
            return self.image_head(feature)
        batch, channels, time, height, width = feature.shape
        flattened = feature.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        masks = self.video_head(flattened)
        return {
            "pred_logits": masks["pred_logits"].reshape(batch, time, *masks["pred_logits"].shape[1:]),
            "pred_masks": masks["pred_masks"].reshape(batch, time, *masks["pred_masks"].shape[1:]),
        }
