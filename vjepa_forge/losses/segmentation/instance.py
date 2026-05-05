from __future__ import annotations

"""Forge-native instance segmentation losses adapted from DETR/RF-DETR mask supervision."""

from typing import Any

import torch
import torch.nn.functional as F

from .common import build_instance_targets, dice_loss, match_instances, sigmoid_ce_mask_loss


def instance_segmentation_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, Any],
    *,
    num_classes: int,
    lambda_mask: float = 5.0,
    lambda_dice: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if outputs["pred_logits"].ndim == 4:
        flat_logits = outputs["pred_logits"].reshape(-1, outputs["pred_logits"].shape[-2], outputs["pred_logits"].shape[-1])
        flat_masks = outputs["pred_masks"].reshape(-1, outputs["pred_masks"].shape[-3], outputs["pred_masks"].shape[-2], outputs["pred_masks"].shape[-1])
        targets = build_instance_targets(
            [{"segments": [ann for ann in item["segments"] if int(ann["frame_idx"]) == frame_idx]} for item in labels["segments"] for frame_idx in range(outputs["pred_masks"].shape[1])],
            output_size=int(flat_masks.shape[-1]),
        )
    else:
        flat_logits = outputs["pred_logits"]
        flat_masks = outputs["pred_masks"]
        targets = build_instance_targets(labels["segments"], output_size=int(flat_masks.shape[-1]))
    cls_loss = flat_logits.sum() * 0.0
    mask_bce = flat_masks.sum() * 0.0
    mask_dice = flat_masks.sum() * 0.0
    for pred_logits, pred_masks, target in zip(flat_logits, flat_masks, targets, strict=True):
        src_idx, tgt_idx = match_instances(pred_logits, pred_masks, {key: value.to(pred_logits.device) for key, value in target.items()})
        target_classes = torch.full((pred_logits.shape[0],), int(num_classes), dtype=torch.int64, device=pred_logits.device)
        if src_idx.numel():
            target_classes[src_idx] = target["labels"].to(pred_logits.device)[tgt_idx]
        cls_loss = cls_loss + F.cross_entropy(pred_logits, target_classes)
        if src_idx.numel():
            selected_masks = pred_masks[src_idx]
            selected_targets = target["masks"].to(pred_masks.device)[tgt_idx]
            mask_bce = mask_bce + sigmoid_ce_mask_loss(selected_masks, selected_targets).mean()
            mask_dice = mask_dice + dice_loss(selected_masks, selected_targets).mean()
    total = cls_loss + float(lambda_mask) * mask_bce + float(lambda_dice) * mask_dice
    return total, {
        "loss_ce": float(cls_loss.detach().cpu().item()),
        "loss_mask": float(mask_bce.detach().cpu().item()),
        "loss_dice": float(mask_dice.detach().cpu().item()),
    }
