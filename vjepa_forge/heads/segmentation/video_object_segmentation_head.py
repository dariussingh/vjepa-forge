from __future__ import annotations

import torch
from torch import nn


class VideoObjectSegmentationHead(nn.Module):
    def __init__(self, input_dim: int, num_objects: int = 4) -> None:
        super().__init__()
        self.memory_proj = nn.Linear(input_dim, input_dim)
        self.mask_proj = nn.Conv3d(input_dim, num_objects, kernel_size=1)

    def forward(self, reference_tokens: torch.Tensor, current_features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.memory_proj(reference_tokens.mean(dim=1))
        conditioned = current_features + memory.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return {"pred_masks": self.mask_proj(conditioned), "memory_tokens": memory}
