from __future__ import annotations

"""Reimplements RF-DETR and DETR-style box utilities for Forge native dense-task training."""

import torch
from torchvision.ops import box_iou as _tv_box_iou
from torchvision.ops import generalized_box_iou as _tv_generalized_box_iou
from torchvision.ops import nms as _tv_nms


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Reimplements DETR/RF-DETR box conversion helpers for Forge-native box math."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Reimplements DETR/RF-DETR box conversion helpers for Forge-native box math."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0), dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Reuses torchvision box IoU inside Forge-native dense-task metrics and assignment."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    return _tv_box_iou(boxes1, boxes2)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Reuses torchvision generalized IoU for Forge-native DETR-style losses."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    return _tv_generalized_box_iou(boxes1, boxes2)


def clip_boxes_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Clamps normalized xyxy boxes into the unit square for Forge detection decode."""
    clipped = boxes.clone()
    clipped[..., 0::2] = clipped[..., 0::2].clamp(0.0, 1.0)
    clipped[..., 1::2] = clipped[..., 1::2].clamp(0.0, 1.0)
    return clipped


def batched_nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Reuses torchvision NMS in Forge-native prediction postprocess."""
    if boxes.numel() == 0 or scores.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    return _tv_nms(boxes, scores, float(iou_threshold))
