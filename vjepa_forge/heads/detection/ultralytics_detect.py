from __future__ import annotations

"""Reimplements the required Ultralytics-style detection head and losses for Forge V-JEPA dense tasks."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from vjepa_forge.backbones.vjepa21 import VJEPAFeaturePyramidAdapter
from .box_ops import box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh, clip_boxes_xyxy


@dataclass(frozen=True)
class ModelConfig:
    nc: int
    class_names: list[str] | None
    imgsz: int
    hidden_dim: int
    reg_max: int
    num_classes: int
    backbone: dict[str, Any]


def build_model_config(config: dict[str, Any], data: dict[str, Any], *, imgsz: int | None = None) -> ModelConfig:
    """Builds a Forge-native Ultralytics-style detection config from existing Forge model/data settings."""
    model_cfg = dict(config["model"])
    detector_cfg = dict(model_cfg.get("detector", {}))
    nc = int(model_cfg.get("num_classes", data.get("nc", 1)))
    class_names = model_cfg.get("class_names") or data.get("names")
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names)]
    return ModelConfig(
        nc=nc,
        class_names=class_names if isinstance(class_names, list) else None,
        imgsz=int(model_cfg.get("image_size", imgsz or 384)),
        hidden_dim=int(detector_cfg.get("hidden_dim", model_cfg.get("head", {}).get("hidden_dim", 256))),
        reg_max=int(detector_cfg.get("reg_max", 16)),
        num_classes=nc,
        backbone=dict(model_cfg.get("backbone", {})),
    )


class ConvNormAct(nn.Module):
    """Reimplements the small conv blocks commonly used in Ultralytics-style dense heads for Forge."""

    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__()
        padding = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, padding=padding, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DistributionFocalProjector(nn.Module):
    """Reimplements Ultralytics-style DFL projection for Forge-native box decoding."""

    def __init__(self, reg_max: int) -> None:
        super().__init__()
        self.reg_max = int(reg_max)
        self.register_buffer("bins", torch.arange(self.reg_max + 1, dtype=torch.float32), persistent=False)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        batch, count, _, bins = distances.shape
        probs = distances.softmax(dim=-1)
        values = (probs * self.bins.view(1, 1, 1, bins)).sum(dim=-1)
        return values.view(batch, count, 4)


class UltralyticsLikeDetectHead(nn.Module):
    """Reimplements the required Ultralytics-style detection head for Forge using V-JEPA feature pyramids."""

    expects_image_input = False

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        media: str,
        *,
        hidden_dim: int = 256,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        self.media = media
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.reg_max = int(reg_max)
        self.adapter = VJEPAFeaturePyramidAdapter(input_dim, out_channels=self.hidden_dim)
        self.cls_towers = nn.ModuleList(
            [
                nn.Sequential(ConvNormAct(self.hidden_dim, self.hidden_dim), ConvNormAct(self.hidden_dim, self.hidden_dim), nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1))
                for _ in range(3)
            ]
        )
        self.reg_towers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(self.hidden_dim, self.hidden_dim),
                    ConvNormAct(self.hidden_dim, self.hidden_dim),
                    nn.Conv2d(self.hidden_dim, 4 * (self.reg_max + 1), kernel_size=1),
                )
                for _ in range(3)
            ]
        )
        self.dfl = DistributionFocalProjector(self.reg_max)

    def _flatten_video_features(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], tuple[int, int] | None]:
        if self.media != "video":
            return features, None
        flattened: list[torch.Tensor] = []
        batch = int(features[0].shape[0])
        time = int(features[0].shape[2])
        for feature in features:
            flattened.append(feature.permute(0, 2, 1, 3, 4).reshape(batch * time, feature.shape[1], feature.shape[3], feature.shape[4]))
        return flattened, (batch, time)

    def _anchor_grid(self, height: int, width: int, *, device: torch.device) -> torch.Tensor:
        ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height)
        xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1).view(-1, 2)

    def _decode_level_boxes(self, raw_dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = raw_dist.shape
        dist = raw_dist.view(batch, 4, self.reg_max + 1, height, width).permute(0, 3, 4, 1, 2).reshape(batch, height * width, 4, self.reg_max + 1)
        projected = self.dfl(dist)
        anchors = self._anchor_grid(height, width, device=raw_dist.device).view(1, height * width, 2)
        scale = projected.new_tensor([float(width), float(height), float(width), float(height)]).view(1, 1, 4)
        xyxy = projected.new_empty(batch, height * width, 4)
        xyxy[..., 0] = anchors[..., 0] - projected[..., 0] / scale[..., 0]
        xyxy[..., 1] = anchors[..., 1] - projected[..., 1] / scale[..., 1]
        xyxy[..., 2] = anchors[..., 0] + projected[..., 2] / scale[..., 2]
        xyxy[..., 3] = anchors[..., 1] + projected[..., 3] / scale[..., 3]
        xyxy = clip_boxes_xyxy(xyxy)
        return box_xyxy_to_cxcywh(xyxy), anchors

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features_in, video_shape = self._flatten_video_features(features)
        pyramid = self.adapter(features_in)
        level_logits: list[torch.Tensor] = []
        level_dists: list[torch.Tensor] = []
        decoded_boxes: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        for feature, cls_head, reg_head in zip(pyramid, self.cls_towers, self.reg_towers, strict=True):
            logits = cls_head(feature)
            dists = reg_head(feature)
            boxes, level_anchors = self._decode_level_boxes(dists)
            level_logits.append(logits)
            level_dists.append(dists)
            decoded_boxes.append(boxes)
            anchors.append(level_anchors)
        pred_logits = torch.cat([logits.flatten(2).transpose(1, 2) for logits in level_logits], dim=1)
        pred_boxes = torch.cat(decoded_boxes, dim=1)
        if video_shape is not None:
            batch, time = video_shape
            pred_logits = pred_logits.view(batch, time, pred_logits.shape[1], pred_logits.shape[2])
            pred_boxes = pred_boxes.view(batch, time, pred_boxes.shape[1], pred_boxes.shape[2])
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "level_logits": level_logits,
            "level_dists": level_dists,
            "anchors": anchors,
        }

    def _prepare_targets(self, labels: dict[str, Any], device: torch.device) -> list[dict[str, torch.Tensor]]:
        prepared: list[dict[str, torch.Tensor]] = []
        for item in labels["detections"]:
            detections = item["detections"]
            if detections and isinstance(detections[0], dict) and "frame_idx" in detections[0]:
                raise ValueError("Video detection targets must be flattened per frame before loss computation")
            classes = torch.tensor([int(ann["class_id"]) for ann in detections], dtype=torch.int64, device=device) if detections else torch.empty(0, dtype=torch.int64, device=device)
            boxes = torch.tensor([ann["box"] for ann in detections], dtype=torch.float32, device=device) if detections else torch.empty(0, 4, dtype=torch.float32, device=device)
            prepared.append({"labels": classes, "boxes": boxes})
        return prepared

    def _assign_single(self, pred_boxes: torch.Tensor, pred_logits: torch.Tensor, target: dict[str, torch.Tensor], topk: int = 10) -> dict[str, torch.Tensor]:
        num_preds = pred_boxes.shape[0]
        device = pred_boxes.device
        assigned_labels = torch.full((num_preds,), -1, dtype=torch.int64, device=device)
        assigned_boxes = torch.zeros((num_preds, 4), dtype=torch.float32, device=device)
        positive_mask = torch.zeros((num_preds,), dtype=torch.bool, device=device)
        if target["labels"].numel() == 0:
            return {"labels": assigned_labels, "boxes": assigned_boxes, "positive_mask": positive_mask}
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target["boxes"])
        ious = box_iou(pred_xyxy, tgt_xyxy)
        centers = pred_boxes[:, :2]
        tgt_xyxy_exp = tgt_xyxy.unsqueeze(0)
        inside = (
            (centers[:, 0:1] >= tgt_xyxy_exp[..., 0])
            & (centers[:, 0:1] <= tgt_xyxy_exp[..., 2])
            & (centers[:, 1:2] >= tgt_xyxy_exp[..., 1])
            & (centers[:, 1:2] <= tgt_xyxy_exp[..., 3])
        )
        scores = pred_logits.sigmoid()
        cls_scores = scores[:, target["labels"]]
        quality = ious * cls_scores
        for tgt_idx in range(target["labels"].numel()):
            valid = inside[:, tgt_idx]
            if not valid.any():
                valid = torch.ones_like(valid)
            candidate_quality = quality[:, tgt_idx].masked_fill(~valid, float("-inf"))
            k = min(int(topk), int(candidate_quality.numel()))
            top_idx = torch.topk(candidate_quality, k=k).indices
            top_idx = top_idx[candidate_quality[top_idx].isfinite()]
            if top_idx.numel() == 0:
                continue
            best_idx = top_idx[torch.argmax(candidate_quality[top_idx])]
            positive_mask[best_idx] = True
            assigned_labels[best_idx] = target["labels"][tgt_idx]
            assigned_boxes[best_idx] = target["boxes"][tgt_idx]
        return {"labels": assigned_labels, "boxes": assigned_boxes, "positive_mask": positive_mask}

    def compute_loss(self, outputs: dict[str, Any], labels: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        if outputs["pred_logits"].ndim == 4:
            flat_logits = outputs["pred_logits"].reshape(-1, outputs["pred_logits"].shape[-2], outputs["pred_logits"].shape[-1])
            flat_boxes = outputs["pred_boxes"].reshape(-1, outputs["pred_boxes"].shape[-2], outputs["pred_boxes"].shape[-1])
            flat_targets: list[dict[str, Any]] = []
            for item in labels["detections"]:
                frames = {}
                for det in item["detections"]:
                    frames.setdefault(int(det["frame_idx"]), []).append(det)
                max_frames = outputs["pred_logits"].shape[1]
                for frame_idx in range(max_frames):
                    flat_targets.append({"detections": frames.get(frame_idx, [])})
            prepared = self._prepare_targets({"detections": flat_targets}, flat_logits.device)
        else:
            flat_logits = outputs["pred_logits"]
            flat_boxes = outputs["pred_boxes"]
            prepared = self._prepare_targets(labels, flat_logits.device)

        cls_loss = flat_logits.sum() * 0.0
        box_loss = flat_boxes.sum() * 0.0
        dfl_loss = flat_boxes.sum() * 0.0
        for pred_logits, pred_boxes, target in zip(flat_logits, flat_boxes, prepared, strict=True):
            assignment = self._assign_single(pred_boxes, pred_logits, target)
            cls_targets = torch.zeros_like(pred_logits)
            positive_mask = assignment["positive_mask"]
            if positive_mask.any():
                cls_targets[positive_mask, assignment["labels"][positive_mask]] = 1.0
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(pred_logits, cls_targets, reduction="mean")
            if positive_mask.any():
                pos_pred_boxes = pred_boxes[positive_mask]
                pos_tgt_boxes = assignment["boxes"][positive_mask]
                box_loss = box_loss + F.l1_loss(pos_pred_boxes, pos_tgt_boxes, reduction="mean")
                pred_xyxy = box_cxcywh_to_xyxy(pos_pred_boxes)
                tgt_xyxy = box_cxcywh_to_xyxy(pos_tgt_boxes)
                lt = (pred_xyxy[..., :2] - tgt_xyxy[..., :2]).abs()
                rb = (pred_xyxy[..., 2:] - tgt_xyxy[..., 2:]).abs()
                dfl_loss = dfl_loss + (lt.mean() + rb.mean()) * 0.5
        total = cls_loss + 5.0 * box_loss + 1.5 * dfl_loss
        stats = {
            "loss_cls": float(cls_loss.detach().cpu().item()),
            "loss_box": float(box_loss.detach().cpu().item()),
            "loss_dfl": float(dfl_loss.detach().cpu().item()),
        }
        return total, stats

    def decode_predictions(
        self,
        outputs: dict[str, Any],
        *,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> list[dict[str, torch.Tensor]]:
        from .box_ops import batched_nms_xyxy

        if outputs["pred_logits"].ndim == 4:
            logits = outputs["pred_logits"].reshape(-1, outputs["pred_logits"].shape[-2], outputs["pred_logits"].shape[-1])
            boxes = outputs["pred_boxes"].reshape(-1, outputs["pred_boxes"].shape[-2], outputs["pred_boxes"].shape[-1])
        else:
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
        decoded: list[dict[str, torch.Tensor]] = []
        for pred_logits, pred_boxes in zip(logits, boxes, strict=True):
            probs = pred_logits.sigmoid()
            scores, labels = probs.max(dim=-1)
            keep = scores >= float(score_threshold)
            if not keep.any():
                decoded.append(
                    {
                        "scores": scores.new_empty(0),
                        "labels": labels.new_empty(0),
                        "boxes": pred_boxes.new_empty((0, 4)),
                    }
                )
                continue
            kept_scores = scores[keep]
            kept_labels = labels[keep]
            kept_boxes = box_cxcywh_to_xyxy(pred_boxes[keep])
            final_indices: list[torch.Tensor] = []
            for class_id in kept_labels.unique():
                class_mask = kept_labels == class_id
                class_keep = batched_nms_xyxy(kept_boxes[class_mask], kept_scores[class_mask], iou_threshold)
                source_idx = torch.nonzero(class_mask, as_tuple=False).squeeze(1)[class_keep]
                final_indices.append(source_idx)
            keep_idx = torch.cat(final_indices, dim=0) if final_indices else kept_scores.new_empty(0, dtype=torch.int64)
            if keep_idx.numel() > max_detections:
                keep_idx = keep_idx[torch.argsort(kept_scores[keep_idx], descending=True)[:max_detections]]
            decoded.append(
                {
                    "scores": kept_scores[keep_idx],
                    "labels": kept_labels[keep_idx],
                    "boxes": kept_boxes[keep_idx],
                }
            )
        return decoded


class VJEPADetectionModel(UltralyticsLikeDetectHead):
    """Keeps the historical public symbol while routing to the native Forge implementation."""


def create_vjepa_detection_model(model_cfg: ModelConfig):
    """Keeps the historical factory symbol while returning the native Forge implementation."""
    return VJEPADetectionModel(
        input_dim=int(model_cfg.hidden_dim),
        num_classes=model_cfg.num_classes,
        media="image",
        hidden_dim=model_cfg.hidden_dim,
        reg_max=model_cfg.reg_max,
    )
