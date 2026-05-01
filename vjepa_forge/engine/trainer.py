from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vjepa_forge.data import AnomalyLoader, ClassifyLoader, DetectLoader, ForgeBatch, ForgeDataset, SegmentLoader


@dataclass
class TrainResult:
    loss: float
    steps: int


class BaseTrainer:
    def __init__(self, model, *, data: str, epochs: int = 1, batch_size: int = 2, num_workers: int = 0, device: str = "cpu") -> None:
        self.model = model
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)

    def build_dataset(self, split: str = "train") -> ForgeDataset:
        return ForgeDataset(self.data, split=split)

    def build_loader(self, split: str = "train"):
        dataset = self.build_dataset(split=split)
        clip_len = int(self.model.data_cfg.get("clip_len", self.model.data_cfg.get("num_frames", 8)))
        clip_stride = int(self.model.data_cfg.get("clip_stride", 1))
        if dataset.task == "classify":
            collator = ClassifyLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride)
        elif dataset.task == "detect":
            collator = DetectLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride)
        elif dataset.task == "segment":
            collator = SegmentLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride)
        else:
            collator = AnomalyLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collator.collate)

    def compute_loss(self, batch: ForgeBatch, outputs) -> torch.Tensor:
        if batch.task == "classify":
            return F.cross_entropy(outputs, batch.labels["class_ids"].to(outputs.device))
        if batch.task == "detect":
            return outputs["pred_boxes"].mean() + outputs["pred_logits"].float().mean()
        if batch.task == "segment":
            if isinstance(outputs, dict):
                return outputs["pred_masks"].mean() + outputs["pred_logits"].float().mean()
            return outputs.mean()
        targets = batch.labels["targets"].to(outputs.device)
        return F.binary_cross_entropy_with_logits(outputs, targets)

    def run(self) -> TrainResult:
        self.model.to(self.device)
        self.model.train(True)
        loader = self.build_loader(split="train")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        last_loss = 0.0
        steps = 0
        for _ in range(self.epochs):
            for batch in loader:
                batch.x = batch.x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch)
                loss = self.compute_loss(batch, outputs)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                steps += 1
        return TrainResult(loss=last_loss, steps=steps)
