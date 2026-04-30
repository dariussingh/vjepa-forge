from __future__ import annotations

import torch

from .image_tokenizer import ImageTokenizer
from .video_tokenizer import VideoTokenizer


class MultiModalTokenizer:
    def __init__(self) -> None:
        self.image = ImageTokenizer()
        self.video = VideoTokenizer()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.image(x)
        if x.ndim == 5:
            return self.video(x)
        raise ValueError(f"Expected 4D or 5D tensor, got {tuple(x.shape)}")
