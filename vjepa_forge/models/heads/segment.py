from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from vjepa_forge.heads.segmentation import InstanceSegmentationHead, SemanticSegmentationHead, VideoSemanticSegmentationHead
from vjepa_forge.heads.segmentation.losses import build_instance_targets, build_semantic_targets, dice_loss, match_instances, sigmoid_ce_mask_loss


class ForgeSegmentHead(nn.Module):
    """Forge-facing segmentation head adapter that selects the configured segmentation strategy."""

    def __init__(self, input_dim: int, num_classes: int, media: str, model_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        model_cfg = {} if model_cfg is None else dict(model_cfg)
        segmenter_cfg = dict(model_cfg.get("segmenter", {}))
        self.strategy = str(segmenter_cfg.get("type", model_cfg.get("strategy", "ultralytics"))).lower()
        self.media = media
        self.num_classes = int(num_classes)
        self.num_queries = int(segmenter_cfg.get("num_queries", 16))
        if self.strategy == "ultralytics":
            self.image_head = SemanticSegmentationHead(input_dim, self.num_classes)
            self.video_head = VideoSemanticSegmentationHead(input_dim, self.num_classes)
        elif self.strategy == "rf_detr":
            self.image_head = InstanceSegmentationHead(input_dim, num_queries=self.num_queries, num_classes=self.num_classes)
            self.video_head = InstanceSegmentationHead(input_dim, num_queries=self.num_queries, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported segmenter strategy: {self.strategy}")

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        feature = features[-1]
        if self.strategy == "ultralytics":
            if self.media == "image":
                return self.image_head(feature)
            logits = self.video_head(feature).permute(0, 2, 1, 3, 4).contiguous()
            return {
                "pred_logits": logits,
                "pred_masks": logits.softmax(dim=2),
            }
        if self.media == "image":
            return self.image_head(feature)
        batch, channels, time, height, width = feature.shape
        flattened = feature.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        masks = self.video_head(flattened)
        return {
            "pred_logits": masks["pred_logits"].reshape(batch, time, *masks["pred_logits"].shape[1:]),
            "pred_masks": masks["pred_masks"].reshape(batch, time, *masks["pred_masks"].shape[1:]),
        }

    def compute_segmentation_loss(self, outputs: torch.Tensor | dict[str, torch.Tensor], labels: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        if self.strategy == "ultralytics":
            if isinstance(outputs, dict):
                logits = outputs["pred_logits"]
                targets = build_semantic_targets(labels["segments"], num_classes=self.num_classes, output_size=int(logits.shape[-1]), video_frames=int(logits.shape[1])).to(logits.device)
                flat_logits = logits.reshape(-1, logits.shape[2], logits.shape[3], logits.shape[4])
                loss = F.cross_entropy(flat_logits, targets)
                return loss, {"loss_ce": float(loss.detach().cpu().item())}
            if outputs.ndim == 5:
                targets = build_semantic_targets(labels["segments"], num_classes=self.num_classes, output_size=int(outputs.shape[-1]), video_frames=int(outputs.shape[2])).to(outputs.device)
                logits = outputs.permute(0, 2, 1, 3, 4).reshape(-1, outputs.shape[1], outputs.shape[3], outputs.shape[4])
                targets = targets.to(outputs.device)
            else:
                targets = build_semantic_targets(labels["segments"], num_classes=self.num_classes, output_size=int(outputs.shape[-1])).to(outputs.device)
                logits = outputs
            loss = F.cross_entropy(logits, targets)
            return loss, {"loss_ce": float(loss.detach().cpu().item())}

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
            src_idx, tgt_idx = match_instances(pred_logits, pred_masks, {k: v.to(pred_logits.device) for k, v in target.items()})
            target_classes = torch.full((pred_logits.shape[0],), self.num_classes, dtype=torch.int64, device=pred_logits.device)
            if src_idx.numel():
                target_classes[src_idx] = target["labels"].to(pred_logits.device)[tgt_idx]
            cls_loss = cls_loss + F.cross_entropy(pred_logits, target_classes)
            if src_idx.numel():
                selected_masks = pred_masks[src_idx]
                selected_targets = target["masks"].to(pred_masks.device)[tgt_idx]
                mask_bce = mask_bce + sigmoid_ce_mask_loss(selected_masks, selected_targets).mean()
                mask_dice = mask_dice + dice_loss(selected_masks, selected_targets).mean()
        total = cls_loss + 5.0 * mask_bce + 5.0 * mask_dice
        return total, {
            "loss_ce": float(cls_loss.detach().cpu().item()),
            "loss_mask": float(mask_bce.detach().cpu().item()),
            "loss_dice": float(mask_dice.detach().cpu().item()),
        }

    def decode_predictions(self, outputs: torch.Tensor | dict[str, torch.Tensor], *, threshold: float = 0.5):
        if self.strategy == "ultralytics":
            logits = outputs["pred_logits"] if isinstance(outputs, dict) else outputs
            if logits.ndim == 5:
                probs = logits.softmax(dim=2)
                return [{"semantic_masks": probs.argmax(dim=2)}]
            return [{"semantic_masks": logits.softmax(dim=1).argmax(dim=1)}]
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        if pred_logits.ndim == 4:
            pred_logits = pred_logits.reshape(-1, pred_logits.shape[-2], pred_logits.shape[-1])
            pred_masks = pred_masks.reshape(-1, pred_masks.shape[-3], pred_masks.shape[-2], pred_masks.shape[-1])
        decoded = []
        for logits, masks in zip(pred_logits, pred_masks, strict=True):
            scores, labels = logits.softmax(dim=-1)[..., :-1].max(dim=-1)
            keep = scores > float(threshold)
            decoded.append(
                {
                    "scores": scores[keep],
                    "labels": labels[keep],
                    "masks": masks[keep].sigmoid() > threshold,
                }
            )
        return decoded
