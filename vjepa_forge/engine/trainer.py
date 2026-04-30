from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vjepa_forge.backbones import VJEPAImageBackbone, VJEPAVideoBackbone
from vjepa_forge.data import (
    ImageNetVIDDataset,
    RandomDetectionDataset,
    RandomImageDataset,
    RandomSegmentationDataset,
    RandomVideoDataset,
    RandomVideoDetectionDataset,
    collate_detection_batch,
)
from vjepa_forge.engine.checkpointing import save_checkpoint
from vjepa_forge.heads.classification import ImageClassificationHead, VideoClassificationHead
from vjepa_forge.heads.detection import FrozenVJEPARFDETR, TemporalDETRHead, build_rf_detr_config
from vjepa_forge.heads.detection.rf_detr import HungarianMatcher, SetCriterion, prepare_targets
from vjepa_forge.heads.segmentation import InstanceSegmentationHead, SemanticSegmentationHead
from vjepa_forge.metrics.classification import top1_accuracy
from vjepa_forge.metrics.segmentation import mean_iou_stub


@dataclass
class TrainResult:
    checkpoint_path: str
    steps: int


class ClassificationModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_size = int(config["data"].get("image_size", 384))
        num_classes = int(config["model"].get("num_classes", 10))
        backbone_cfg = dict(config["model"]["backbone"])
        if config.get("input_type", "image") == "video":
            num_frames = int(config["data"].get("num_frames", 8))
            self.backbone = VJEPAVideoBackbone(
                name=backbone_cfg.get("name", "vit_base"),
                checkpoint=backbone_cfg.get("checkpoint"),
                imgsz=image_size,
                num_frames=num_frames,
            )
            self.head = VideoClassificationHead(self.backbone.embed_dim, num_classes)
        else:
            self.backbone = VJEPAImageBackbone(
                name=backbone_cfg.get("name", "vit_base"),
                checkpoint=backbone_cfg.get("checkpoint"),
                imgsz=image_size,
            )
            self.head = ImageClassificationHead(self.backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]
        return self.head(features)


class SemanticSegmentationModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_size = int(config["data"].get("image_size", 384))
        num_classes = int(config["model"].get("num_classes", 8))
        backbone_cfg = dict(config["model"]["backbone"])
        self.backbone = VJEPAImageBackbone(
            name=backbone_cfg.get("name", "vit_base"),
            checkpoint=backbone_cfg.get("checkpoint"),
            imgsz=image_size,
        )
        self.head = SemanticSegmentationHead(self.backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)[-1])


class InstanceSegmentationModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_size = int(config["data"].get("image_size", 384))
        num_classes = int(config["model"].get("num_classes", 8))
        backbone_cfg = dict(config["model"]["backbone"])
        self.backbone = VJEPAImageBackbone(
            name=backbone_cfg.get("name", "vit_base"),
            checkpoint=backbone_cfg.get("checkpoint"),
            imgsz=image_size,
        )
        self.head = InstanceSegmentationHead(self.backbone.embed_dim, num_queries=16, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.backbone(x)[-1])


class TemporalDetectionModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_size = int(config["data"].get("image_size", 384))
        num_frames = int(config["data"].get("num_frames", 8))
        backbone_cfg = dict(config["model"]["backbone"])
        detector_cfg = dict(config.get("detector", {}))
        self.backbone = VJEPAVideoBackbone(
            name=backbone_cfg.get("name", "vit_base"),
            checkpoint=backbone_cfg.get("checkpoint"),
            imgsz=image_size,
            num_frames=num_frames,
            tubelet_size=int(backbone_cfg.get("tubelet_size", 1)),
        )
        self.head = TemporalDETRHead(
            self.backbone.embed_dim,
            num_queries=int(detector_cfg.get("num_queries", 100)),
            num_classes=int(config["model"].get("num_classes", 10)),
            hidden_dim=int(detector_cfg.get("hidden_dim", self.backbone.embed_dim)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.backbone(x)[-1])


def build_model(config: dict[str, Any]) -> nn.Module:
    task = config["task"]
    if task == "classification":
        return ClassificationModel(config)
    if task == "detection":
        if config.get("input_type", "image") == "video":
            return TemporalDetectionModel(config)
        data = {"nc": int(config["model"].get("num_classes", 10)), "names": list(range(int(config["model"].get("num_classes", 10))))}
        return FrozenVJEPARFDETR(build_rf_detr_config(config, data, imgsz=int(config["data"].get("image_size", 384))))
    if task == "segmentation":
        seg_type = config.get("segmentation_type", "semantic")
        if seg_type == "instance":
            return InstanceSegmentationModel(config)
        return SemanticSegmentationModel(config)
    raise ValueError(f"Unsupported task for unified trainer: {task}")


def build_dataset(config: dict[str, Any], *, split: str = "train"):
    task = config["task"]
    image_size = int(config["data"].get("image_size", 384))
    num_classes = int(config["model"].get("num_classes", 10))
    if task == "classification":
        if config.get("input_type", "image") == "video":
            return RandomVideoDataset(image_size=image_size, num_frames=int(config["data"].get("num_frames", 8)), num_classes=num_classes)
        return RandomImageDataset(image_size=image_size, num_classes=num_classes)
    if task == "detection":
        if config.get("input_type", "image") == "video":
            if str(config["data"].get("name", "random_video_detection")).lower() == "imagenet_vid":
                return ImageNetVIDDataset(
                    config["data"]["root"],
                    split=split,
                    image_size=image_size,
                    clip_length=int(config["data"].get("num_frames", 8)),
                    clip_stride=int(config["data"].get("clip_stride", 1)),
                    max_clips=config["data"].get("max_clips"),
                )
            return RandomVideoDetectionDataset(
                image_size=image_size,
                num_frames=int(config["data"].get("num_frames", 8)),
                num_classes=num_classes,
            )
        return RandomDetectionDataset(image_size=image_size, num_classes=num_classes)
    if task == "segmentation":
        return RandomSegmentationDataset(image_size=image_size, num_classes=num_classes)
    raise ValueError(f"Unsupported dataset task: {task}")


def build_dataloader(config: dict[str, Any], *, split: str = "train") -> DataLoader:
    dataset = build_dataset(config, split=split)
    batch_size = int(config["data"].get("batch_size", 2))
    if config["task"] == "detection":
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_detection_batch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_detection_criterion(config: dict[str, Any]) -> SetCriterion:
    return SetCriterion(
        num_classes=int(config["model"].get("num_classes", 10)),
        matcher=HungarianMatcher(),
        weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
    )


def _prepare_video_targets(targets: list[dict[str, Any]], image_size: int) -> list[dict[str, torch.Tensor]]:
    frames: list[dict[str, torch.Tensor]] = []
    for sample in targets:
        for frame in sample["frames"]:
            frames.append({"boxes": frame["boxes"], "labels": frame["labels"]})
    return prepare_targets(frames, image_size)


def compute_detection_loss(
    config: dict[str, Any],
    criterion: SetCriterion,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, float]]:
    image_size = int(config["data"].get("image_size", 384))
    if config.get("input_type", "image") == "video":
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        batch_size, num_frames, num_queries, channels = pred_logits.shape
        flat_outputs = {
            "pred_logits": pred_logits.reshape(batch_size * num_frames, num_queries, channels),
            "pred_boxes": pred_boxes.reshape(batch_size * num_frames, num_queries, 4),
        }
        prepared_targets = _prepare_video_targets(targets, image_size)
    else:
        flat_outputs = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"], "aux_outputs": outputs.get("aux_outputs", [])}
        prepared_targets = prepare_targets(targets, image_size)
    target_device = flat_outputs["pred_boxes"].device
    prepared_targets = [
        {
            "boxes": target["boxes"].to(target_device),
            "labels": target["labels"].to(target_device),
        }
        for target in prepared_targets
    ]
    losses = criterion(flat_outputs, prepared_targets)
    total = sum(losses[key] * _metric_weight(key, criterion.weight_dict) for key in losses)
    metrics = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
    return total, metrics


def _metric_weight(loss_name: str, weight_dict: dict[str, float]) -> float:
    if loss_name.startswith("loss_ce"):
        return weight_dict["loss_ce"]
    if loss_name.startswith("loss_bbox"):
        return weight_dict["loss_bbox"]
    if loss_name.startswith("loss_giou"):
        return weight_dict["loss_giou"]
    return 1.0


def _compute_loss(
    config: dict[str, Any],
    model: nn.Module,
    batch: Any,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    if config["task"] == "classification":
        inputs, labels = batch
        logits = model(inputs.to(device))
        loss = criterion(logits, labels.to(device))
        return loss, {"top1": top1_accuracy(logits.detach().cpu(), labels)}
    if config["task"] == "segmentation":
        inputs, labels = batch
        logits = model(inputs.to(device))
        loss = criterion(logits, labels.to(device))
        return loss, {"miou": mean_iou_stub(logits.detach().cpu(), labels)}
    if config["task"] == "detection":
        inputs, targets = batch
        outputs = model(inputs.to(device))
        loss, metrics = compute_detection_loss(config, criterion, outputs, targets)
        return loss, metrics
    raise ValueError(f"Unsupported task: {config['task']}")


def train(config: dict[str, Any]) -> TrainResult:
    model = build_model(config)
    loader = build_dataloader(config, split="train")
    device = torch.device(config["train"].get("device", "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("lr", 1.0e-4)))
    criterion: nn.Module
    if config["task"] == "detection":
        criterion = build_detection_criterion(config)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.train()
    steps = 0
    for _ in range(int(config["train"].get("epochs", 1))):
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = _compute_loss(config, model, batch, device, criterion)
            loss.backward()
            optimizer.step()
            steps += 1
    checkpoint_path = str((Path(config["_config_path"]).parent / "last.pt").resolve())
    save_checkpoint(model, checkpoint_path)
    return TrainResult(checkpoint_path=checkpoint_path, steps=steps)


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    model = build_model(config)
    loader = build_dataloader(config, split="val")
    device = torch.device(config["train"].get("device", "cpu"))
    model.to(device)
    if config["task"] == "detection":
        criterion: nn.Module = build_detection_criterion(config)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.eval()
    count = 0
    total_loss = 0.0
    metrics_sum: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            loss, metrics = _compute_loss(config, model, batch, device, criterion)
            total_loss += float(loss.detach().cpu().item())
            count += 1
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
    summary = {"task": config["task"], "batches": count, "loss": total_loss / max(count, 1)}
    for key, value in metrics_sum.items():
        summary[key] = value / max(count, 1)
    return summary
