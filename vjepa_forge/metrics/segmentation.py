from __future__ import annotations

import torch


def mean_iou_stub(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())
