from __future__ import annotations

"""Implements Forge-native segmentation metrics for semantic and instance outputs."""

import torch


def mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Computes mean IoU for Forge-native semantic segmentation outputs."""
    pred = logits.argmax(dim=1)
    ious: list[float] = []
    for class_id in range(int(num_classes)):
        pred_mask = pred == class_id
        target_mask = target == class_id
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        intersection = (pred_mask & target_mask).sum().item()
        ious.append(intersection / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def instance_mask_iou(pred_masks: list[torch.Tensor], target_masks: list[torch.Tensor]) -> float:
    """Computes a simple matched mask IoU summary for Forge-native instance segmentation."""
    if not pred_masks and not target_masks:
        return 1.0
    if not pred_masks or not target_masks:
        return 0.0
    values: list[float] = []
    used: set[int] = set()
    for pred in pred_masks:
        best_iou = 0.0
        best_idx = -1
        pred_bool = pred.bool()
        for idx, target in enumerate(target_masks):
            if idx in used:
                continue
            target_bool = target.bool()
            union = (pred_bool | target_bool).sum().item()
            if union == 0:
                continue
            inter = (pred_bool & target_bool).sum().item()
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            used.add(best_idx)
        values.append(best_iou)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
