from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vjepa_forge.heads.segmentation import InstanceSegmentationHead, SemanticSegmentationHead, VideoSemanticSegmentationHead
from vjepa_forge.losses.segmentation import instance_segmentation_loss, semantic_segmentation_loss


class ForgeSegmentHead(nn.Module):
    """Forge-facing segmentation head adapter that selects the configured segmentation strategy."""

    def __init__(self, input_dim: int, num_classes: int, media: str, model_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        model_cfg = {} if model_cfg is None else dict(model_cfg)
        segmenter_cfg = dict(model_cfg.get("segmenter", {}))
        self._loss_cfg = dict(model_cfg.get("loss", {}))
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
            return semantic_segmentation_loss(outputs, labels, num_classes=self.num_classes, lambda_dice=float(self.model_cfg_loss().get("lambda_dice", 1.0)))
        return instance_segmentation_loss(
            outputs,
            labels,
            num_classes=self.num_classes,
            lambda_mask=float(self.model_cfg_loss().get("lambda_mask", 5.0)),
            lambda_dice=float(self.model_cfg_loss().get("lambda_dice", 5.0)),
        )

    def model_cfg_loss(self) -> dict[str, Any]:
        return dict(getattr(self, "_loss_cfg", {}))

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
