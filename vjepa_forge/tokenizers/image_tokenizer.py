from __future__ import annotations

import torch


class ImageTokenizer:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected BCHW image tensor, got {tuple(image.shape)}")
        return image.unsqueeze(2)
