from __future__ import annotations

import torch


class VJEPAImageTokenizer:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image batch [B, C, H, W], got {tuple(image.shape)}")
        return image
