from __future__ import annotations

import torch
from torch import nn

from vjepa_forge.heads.classification import ImageClassificationHead, VideoClassificationHead


class ForgeClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, media: str) -> None:
        super().__init__()
        self.media = media
        if media == "image":
            self.impl = ImageClassificationHead(input_dim, num_classes)
        else:
            self.impl = VideoClassificationHead(input_dim, num_classes)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.impl(features[-1])
