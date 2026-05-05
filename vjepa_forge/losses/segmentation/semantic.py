from __future__ import annotations

"""Forge-native semantic segmentation losses adapted from standard dense prediction training."""

from typing import Any

import torch
import torch.nn.functional as F

from .common import build_semantic_targets


def semantic_segmentation_loss(
    outputs: torch.Tensor | dict[str, torch.Tensor],
    labels: dict[str, Any],
    *,
    num_classes: int,
    lambda_dice: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(outputs, dict):
        logits = outputs["pred_logits"]
        targets = build_semantic_targets(labels["segments"], num_classes=num_classes, output_size=int(logits.shape[-1]), video_frames=int(logits.shape[1])).to(logits.device)
        flat_logits = logits.reshape(-1, logits.shape[2], logits.shape[3], logits.shape[4])
    elif outputs.ndim == 5:
        targets = build_semantic_targets(labels["segments"], num_classes=num_classes, output_size=int(outputs.shape[-1]), video_frames=int(outputs.shape[2])).to(outputs.device)
        flat_logits = outputs.permute(0, 2, 1, 3, 4).reshape(-1, outputs.shape[1], outputs.shape[3], outputs.shape[4])
    else:
        flat_logits = outputs
        targets = build_semantic_targets(labels["segments"], num_classes=num_classes, output_size=int(outputs.shape[-1])).to(outputs.device)
    ce = F.cross_entropy(flat_logits, targets)
    probs = flat_logits.softmax(dim=1)
    target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = 2.0 * (probs * target_one_hot).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice = 1.0 - ((intersection + 1.0) / (denominator + 1.0)).mean()
    total = ce + float(lambda_dice) * dice
    return total, {
        "loss_ce": float(ce.detach().cpu().item()),
        "loss_dice": float(dice.detach().cpu().item()),
    }
