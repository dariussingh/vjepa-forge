from __future__ import annotations

"""Forge-native downstream classification criteria adapted from common ViT fine-tuning practice."""

from typing import Any

import torch

from .common import label_smoothed_cross_entropy, soft_target_cross_entropy


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, loss_cfg: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = {} if loss_cfg is None else dict(loss_cfg)
    if labels.dtype.is_floating_point and labels.ndim == logits.ndim:
        loss = soft_target_cross_entropy(logits, labels)
        return loss, {"loss_ce": float(loss.detach().cpu().item())}
    loss = label_smoothed_cross_entropy(
        logits,
        labels.long(),
        smoothing=float(cfg.get("label_smoothing", 0.0)),
        class_weight=cfg.get("class_weight"),
    )
    return loss, {"loss_ce": float(loss.detach().cpu().item())}
