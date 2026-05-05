from __future__ import annotations

"""Shared criterion helpers for Forge-native task losses."""

from typing import Any

import torch
import torch.nn.functional as F


def _class_weights(value: Any, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    weights = torch.as_tensor(value, dtype=torch.float32, device=device)
    return weights if weights.numel() else None


def label_smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smoothing: float = 0.0,
    class_weight: Any = None,
) -> torch.Tensor:
    if smoothing <= 0.0:
        return F.cross_entropy(logits, targets, weight=_class_weights(class_weight, device=logits.device))
    num_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    true_dist = torch.full_like(log_probs, float(smoothing) / max(num_classes - 1, 1))
    true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - float(smoothing))
    loss = -(true_dist * log_probs).sum(dim=-1)
    weights = _class_weights(class_weight, device=logits.device)
    if weights is not None:
        loss = loss * weights[targets]
    return loss.mean()


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()

