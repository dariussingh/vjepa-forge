from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vjepa_forge.backbones import VJEPAImageBackbone, VJEPAVideoBackbone
from vjepa_forge.data import RandomDetectionDataset, RandomImageDataset, RandomSegmentationDataset, RandomVideoDataset
from vjepa_forge.engine.checkpointing import save_checkpoint
from vjepa_forge.heads.classification import ImageClassificationHead, VideoClassificationHead
from vjepa_forge.heads.detection import FrozenVJEPARFDETR, build_rf_detr_config
from vjepa_forge.heads.segmentation import InstanceSegmentationHead, SemanticSegmentationHead


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


def build_model(config: dict[str, Any]) -> nn.Module:
    task = config["task"]
    if task == "classification":
        return ClassificationModel(config)
    if task == "detection":
        data = {"nc": int(config["model"].get("num_classes", 10)), "names": list(range(int(config["model"].get("num_classes", 10))))}
        return FrozenVJEPARFDETR(build_rf_detr_config(config, data, imgsz=int(config["data"].get("image_size", 384))))
    if task == "segmentation":
        seg_type = config.get("segmentation_type", "semantic")
        if seg_type == "instance":
            return InstanceSegmentationModel(config)
        return SemanticSegmentationModel(config)
    raise ValueError(f"Unsupported task for unified trainer: {task}")


def build_dataset(config: dict[str, Any]):
    task = config["task"]
    image_size = int(config["data"].get("image_size", 384))
    num_classes = int(config["model"].get("num_classes", 10))
    if task == "classification":
        if config.get("input_type", "image") == "video":
            return RandomVideoDataset(image_size=image_size, num_frames=int(config["data"].get("num_frames", 8)), num_classes=num_classes)
        return RandomImageDataset(image_size=image_size, num_classes=num_classes)
    if task == "detection":
        return RandomDetectionDataset(image_size=image_size, num_classes=num_classes)
    if task == "segmentation":
        return RandomSegmentationDataset(image_size=image_size, num_classes=num_classes)
    raise ValueError(f"Unsupported dataset task: {task}")


def train(config: dict[str, Any]) -> TrainResult:
    model = build_model(config)
    dataset = build_dataset(config)
    loader = DataLoader(dataset, batch_size=int(config["data"].get("batch_size", 2)), shuffle=False)
    device = torch.device(config["train"].get("device", "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("lr", 1.0e-4)))
    criterion = nn.CrossEntropyLoss()
    model.train()
    steps = 0
    for _ in range(int(config["train"].get("epochs", 1))):
        for batch in loader:
            optimizer.zero_grad()
            if config["task"] == "classification":
                inputs, labels = batch
                logits = model(inputs.to(device))
                loss = criterion(logits, labels.to(device))
            elif config["task"] == "segmentation":
                inputs, labels = batch
                logits = model(inputs.to(device))
                loss = criterion(logits, labels.to(device))
            elif config["task"] == "detection":
                inputs, _targets = batch
                outputs = model(inputs.to(device))
                loss = outputs["pred_logits"].mean() + outputs["pred_boxes"].mean()
            else:
                raise ValueError(f"Unsupported task: {config['task']}")
            loss.backward()
            optimizer.step()
            steps += 1
    checkpoint_path = str((Path(config["_recipe_path"]).parent / "last.pt").resolve())
    save_checkpoint(model, checkpoint_path)
    return TrainResult(checkpoint_path=checkpoint_path, steps=steps)
