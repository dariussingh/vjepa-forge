from __future__ import annotations

"""Forge-native RF-DETR detection losses and matching adapted for V-JEPA dense features."""

from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from vjepa_forge.heads.detection.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class ForgeHungarianMatcher(nn.Module):
    def __init__(self, class_cost: float = 2.0, bbox_cost: float = 5.0, giou_cost: float = 2.0) -> None:
        super().__init__()
        self.class_cost = float(class_cost)
        self.bbox_cost = float(bbox_cost)
        self.giou_cost = float(giou_cost)

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"].softmax(-1)
        pred_boxes = outputs["pred_boxes"]
        matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_idx, target in enumerate(targets):
            tgt_ids = target["labels"]
            tgt_boxes = target["boxes"]
            if tgt_ids.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_boxes.device)
                matches.append((empty, empty))
                continue
            out_prob = pred_logits[batch_idx]
            out_bbox = pred_boxes[batch_idx]
            class_cost = -out_prob[:, tgt_ids]
            bbox_cost = torch.cdist(out_bbox, tgt_boxes, p=1)
            giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))
            cost = self.class_cost * class_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
            src_idx, tgt_idx = linear_sum_assignment(cost.detach().cpu())
            matches.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.int64, device=pred_boxes.device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=pred_boxes.device),
                )
            )
        return matches


class ForgeSetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: ForgeHungarianMatcher, weight_dict: dict[str, float], eos_coef: float = 0.1) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = float(eos_coef)
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], indices: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel():
                target_classes[batch_idx, src_idx] = targets[batch_idx]["labels"][tgt_idx]
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

    def loss_boxes(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], indices: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        src_boxes: list[torch.Tensor] = []
        tgt_boxes: list[torch.Tensor] = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel():
                src_boxes.append(outputs["pred_boxes"][batch_idx, src_idx])
                tgt_boxes.append(targets[batch_idx]["boxes"][tgt_idx])
        if not src_boxes:
            zero = outputs["pred_boxes"].sum() * 0.0
            return zero, zero
        src = torch.cat(src_boxes, dim=0)
        tgt = torch.cat(tgt_boxes, dim=0)
        normalizer = max(src.shape[0], 1)
        loss_bbox = F.l1_loss(src, tgt, reduction="none").sum() / normalizer
        loss_giou = (1.0 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt)))).sum() / normalizer
        return loss_bbox, loss_giou

    def _single_output_loss(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], suffix: str = "") -> dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)
        return {f"loss_ce{suffix}": loss_ce, f"loss_bbox{suffix}": loss_bbox, f"loss_giou{suffix}": loss_giou}

    def forward(self, outputs: dict[str, Any], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        losses = self._single_output_loss({"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}, targets)
        for idx, aux in enumerate(outputs.get("aux_outputs", [])):
            losses.update(self._single_output_loss(aux, targets, suffix=f"_{idx}"))
        return losses


def compute_rf_detr_loss(
    criterion: ForgeSetCriterion,
    outputs: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = criterion(outputs, targets)
    total = outputs["pred_logits"].sum() * 0.0
    for name, value in losses.items():
        base_name = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name
        total = total + criterion.weight_dict.get(base_name, 1.0) * value
    return total, {name: float(value.detach().cpu().item()) for name, value in losses.items()}
