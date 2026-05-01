from __future__ import annotations

import torch


class VJEPAVideoTokenizer:
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video batch [B, T, C, H, W], got {tuple(video.shape)}")
        return video.permute(0, 2, 1, 3, 4).contiguous()
