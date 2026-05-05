from __future__ import annotations

"""Forge-native Ultralytics-style detection losses adapted for V-JEPA feature pyramids."""

from typing import Any

import torch
import torch.nn.functional as F

from vjepa_forge.heads.detection.box_ops import box_cxcywh_to_xyxy, box_iou


def _distribution_focal_loss(raw_dist: torch.Tensor, target_ltrb: torch.Tensor, reg_max: int) -> torch.Tensor:
    pred = raw_dist.view(-1, 4, reg_max + 1)
    target = target_ltrb.clamp_(0.0, float(reg_max) - 1.0e-3)
    left = target.floor().long()
    right = (left + 1).clamp(max=reg_max)
    weight_right = target - left.float()
    weight_left = 1.0 - weight_right
    losses: list[torch.Tensor] = []
    for side in range(4):
        side_logits = pred[:, side]
        l = F.cross_entropy(side_logits, left[:, side], reduction="none") * weight_left[:, side]
        r = F.cross_entropy(side_logits, right[:, side], reduction="none") * weight_right[:, side]
        losses.append(l + r)
    return torch.stack(losses, dim=1).mean()


def _bbox_to_ltrb(anchor_xy: torch.Tensor, box_xyxy: torch.Tensor, scale_xy: torch.Tensor) -> torch.Tensor:
    left = (anchor_xy[:, 0] - box_xyxy[:, 0]) * scale_xy[:, 0]
    top = (anchor_xy[:, 1] - box_xyxy[:, 1]) * scale_xy[:, 1]
    right = (box_xyxy[:, 2] - anchor_xy[:, 0]) * scale_xy[:, 0]
    bottom = (box_xyxy[:, 3] - anchor_xy[:, 1]) * scale_xy[:, 1]
    return torch.stack((left, top, right, bottom), dim=-1)


def compute_ultralytics_detection_loss(
    outputs: dict[str, Any],
    prepared_targets: list[dict[str, torch.Tensor]],
    *,
    reg_max: int,
    num_classes: int,
    assign_single,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    if pred_logits.ndim == 4:
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-2], pred_logits.shape[-1])
        pred_boxes = pred_boxes.reshape(-1, pred_boxes.shape[-2], pred_boxes.shape[-1])
    flat_raw_dists = torch.cat(
        [level.view(level.shape[0], 4, reg_max + 1, level.shape[2], level.shape[3]).permute(0, 3, 4, 1, 2).reshape(level.shape[0], -1, 4 * (reg_max + 1)) for level in outputs["level_dists"]],
        dim=1,
    )
    if outputs["pred_logits"].ndim == 4:
        batch, time = outputs["pred_logits"].shape[:2]
        flat_raw_dists = flat_raw_dists.view(batch, time, flat_raw_dists.shape[1], flat_raw_dists.shape[2]).reshape(-1, flat_raw_dists.shape[1], flat_raw_dists.shape[2])
    flat_anchors = torch.cat([anchor.expand(pred_boxes.shape[0], -1, -1) for anchor in outputs["anchors"]], dim=1)
    if outputs["pred_logits"].ndim == 4:
        batch, time = outputs["pred_logits"].shape[:2]
        flat_anchors = flat_anchors.view(batch, 1, flat_anchors.shape[1], flat_anchors.shape[2]).expand(batch, time, -1, -1).reshape(-1, flat_anchors.shape[1], flat_anchors.shape[2])
    cls_loss = pred_logits.sum() * 0.0
    box_loss = pred_boxes.sum() * 0.0
    dfl_loss = pred_boxes.sum() * 0.0
    positive_count = 0
    for batch_idx, (batch_logits, batch_boxes, target) in enumerate(zip(pred_logits, pred_boxes, prepared_targets, strict=True)):
        assignment = assign_single(batch_boxes, batch_logits, target)
        cls_targets = torch.zeros_like(batch_logits)
        positive_mask = assignment["positive_mask"]
        if positive_mask.any():
            cls_targets[positive_mask, assignment["labels"][positive_mask]] = 1.0
        cls_loss = cls_loss + F.binary_cross_entropy_with_logits(batch_logits, cls_targets, reduction="mean")
        if not positive_mask.any():
            continue
        positive_count += int(positive_mask.sum().item())
        pos_pred_boxes = batch_boxes[positive_mask]
        pos_tgt_boxes = assignment["boxes"][positive_mask]
        pred_xyxy = box_cxcywh_to_xyxy(pos_pred_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(pos_tgt_boxes)
        ious = torch.diag(box_iou(pred_xyxy, tgt_xyxy))
        box_loss = box_loss + (1.0 - ious).mean()
        raw_dist = flat_raw_dists[batch_idx][positive_mask]
        anchor_xy = flat_anchors[batch_idx][positive_mask]
        scale_xy = anchor_xy.new_ones((anchor_xy.shape[0], 2))
        target_ltrb = _bbox_to_ltrb(anchor_xy, tgt_xyxy, scale_xy)
        dfl_loss = dfl_loss + _distribution_focal_loss(raw_dist, target_ltrb, reg_max)
    normalizer = max(positive_count, 1)
    cls_loss = cls_loss / max(len(prepared_targets), 1)
    box_loss = box_loss / normalizer
    dfl_loss = dfl_loss / normalizer
    total = cls_loss + 7.5 * box_loss + 1.5 * dfl_loss
    return total, {
        "loss_cls": float(cls_loss.detach().cpu().item()),
        "loss_box": float(box_loss.detach().cpu().item()),
        "loss_dfl": float(dfl_loss.detach().cpu().item()),
    }
