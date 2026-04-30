from __future__ import annotations

import torch
from torch import nn


class ResidualScorer(nn.Module):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((predicted - target) ** 2).mean(dim=-1)
