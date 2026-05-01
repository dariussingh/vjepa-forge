from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vjepa_forge.data import AnomalyLoader, ClassifyLoader, DetectLoader, ForgeBatch, ForgeDataset, SegmentLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class TrainResult:
    loss: float
    steps: int


class BaseTrainer:
    def __init__(self, model, *, data: str, epochs: int = 1, batch_size: int = 2, num_workers: int = 0, device: str = "cpu", split: str | None = None) -> None:
        self.model = model
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.split = split if split is not None else getattr(self, "split", None)

    def build_dataset(self, split: str = "train") -> ForgeDataset:
        return ForgeDataset(self.data, split=split)

    def build_loader(self, split: str = "train"):
        dataset = self.build_dataset(split=split)
        clip_len = int(self.model.data_cfg.get("clip_len", self.model.data_cfg.get("num_frames", 8)))
        clip_stride = int(self.model.data_cfg.get("clip_stride", 1))
        image_size = int(self.model.data_cfg.get("image_size", 384))
        reader_cache_size = int(self.model.data_cfg.get("reader_cache_size", 4))
        video_backend = str(self.model.data_cfg.get("video_backend", "auto"))
        if dataset.task == "classify":
            collator = ClassifyLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend)
        elif dataset.task == "detect":
            collator = DetectLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend)
        elif dataset.task == "segment":
            collator = SegmentLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend)
        else:
            collator = AnomalyLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend)
        worker_count = int(self.num_workers)
        if dataset.media == "video" and video_backend == "dali":
            worker_count = 0
        if worker_count <= 0 and dataset.media == "video":
            if video_backend != "dali":
                worker_count = max(1, min(8, os.cpu_count() or 1))
        pin_memory = bool(self.model.data_cfg.get("pin_memory", torch.cuda.is_available() and video_backend != "dali"))
        persistent_workers = bool(self.model.data_cfg.get("persistent_workers", dataset.media == "video" and worker_count > 0))
        prefetch_factor = self.model.data_cfg.get("prefetch_factor", 2 if dataset.media == "video" and worker_count > 0 else None)
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": worker_count,
            "collate_fn": collator.collate,
            "pin_memory": pin_memory,
        }
        if worker_count > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)

    def progress(self, iterable, *, desc: str, total: int | None = None):
        if tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)

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
        for epoch_idx in range(self.epochs):
            progress = self.progress(loader, desc=f"train {epoch_idx + 1}/{self.epochs}", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch)
                loss = self.compute_loss(batch, outputs)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                steps += 1
                if tqdm is not None:
                    progress.set_postfix(loss=f"{last_loss:.4f}")
        return TrainResult(loss=last_loss, steps=steps)
