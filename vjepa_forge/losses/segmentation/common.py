from __future__ import annotations

"""Forge-native segmentation target builders and common mask losses."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


def dice_loss(pred_mask_logits: torch.Tensor, target_masks: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = pred_mask_logits.sigmoid().flatten(1)
    targets = target_masks.float().flatten(1)
    numerator = 2.0 * (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    return 1.0 - ((numerator + eps) / (denominator + eps))


def sigmoid_ce_mask_loss(pred_mask_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred_mask_logits, target_masks.float(), reduction="none").mean(dim=(1, 2))


def rasterize_polygon(normalized_polygon: list[float], *, output_size: int) -> torch.Tensor:
    mask = Image.new("L", (output_size, output_size), 0)
    if len(normalized_polygon) >= 6 and len(normalized_polygon) % 2 == 0:
        points = []
        for idx in range(0, len(normalized_polygon), 2):
            x = max(0.0, min(1.0, float(normalized_polygon[idx])))
            y = max(0.0, min(1.0, float(normalized_polygon[idx + 1])))
            points.append((x * (output_size - 1), y * (output_size - 1)))
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    return torch.from_numpy(np.array(mask, dtype="float32"))


def build_instance_targets(label_items: list[dict[str, Any]], *, output_size: int) -> list[dict[str, torch.Tensor]]:
    prepared: list[dict[str, torch.Tensor]] = []
    for item in label_items:
        annotations = item["segments"]
        labels = torch.tensor([int(ann["class_id"]) for ann in annotations], dtype=torch.int64) if annotations else torch.empty(0, dtype=torch.int64)
        masks = torch.stack([rasterize_polygon(list(ann["polygon"]), output_size=output_size) for ann in annotations], dim=0) if annotations else torch.empty(0, output_size, output_size, dtype=torch.float32)
        prepared.append({"labels": labels, "masks": masks})
    return prepared


def build_semantic_targets(label_items: list[dict[str, Any]], *, num_classes: int, output_size: int, video_frames: int | None = None) -> torch.Tensor:
    targets: list[torch.Tensor] = []
    for item in label_items:
        annotations = item["segments"]
        if video_frames is None:
            mask = torch.zeros(output_size, output_size, dtype=torch.long)
            for ann in annotations:
                poly_mask = rasterize_polygon(list(ann["polygon"]), output_size=output_size) > 0
                mask[poly_mask] = int(ann["class_id"]) % max(int(num_classes), 1)
            targets.append(mask)
            continue
        grouped: dict[int, list[dict[str, Any]]] = {}
        for ann in annotations:
            grouped.setdefault(int(ann["frame_idx"]), []).append(ann)
        for frame_idx in range(video_frames):
            mask = torch.zeros(output_size, output_size, dtype=torch.long)
            for ann in grouped.get(frame_idx, []):
                poly_mask = rasterize_polygon(list(ann["polygon"]), output_size=output_size) > 0
                mask[poly_mask] = int(ann["class_id"]) % max(int(num_classes), 1)
            targets.append(mask)
    return torch.stack(targets, dim=0) if targets else torch.empty(0, output_size, output_size, dtype=torch.long)


def match_instances(pred_logits: torch.Tensor, pred_masks: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if target["labels"].numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
        return empty, empty
    class_cost = -pred_logits.softmax(dim=-1)[:, target["labels"]]
    pred_probs = pred_masks.sigmoid().flatten(1)
    target_masks = target["masks"].to(pred_masks.device).flatten(1)
    bce_cost = torch.cdist(pred_probs, target_masks, p=1)
    inter = pred_probs @ target_masks.t()
    union = pred_probs.sum(dim=1, keepdim=True) + target_masks.sum(dim=1).unsqueeze(0)
    dice_cost = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0))
    cost = class_cost + bce_cost + dice_cost
    src_idx, tgt_idx = linear_sum_assignment(cost.detach().cpu())
    return (
        torch.as_tensor(src_idx, dtype=torch.int64, device=pred_logits.device),
        torch.as_tensor(tgt_idx, dtype=torch.int64, device=pred_logits.device),
    )
