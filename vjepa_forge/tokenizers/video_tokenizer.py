from __future__ import annotations

import torch


class VideoTokenizer:
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected BCTHW video tensor, got {tuple(video.shape)}")
        return video
