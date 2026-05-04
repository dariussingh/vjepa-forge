from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vjepa_forge.data import AnomalyLoader, ClassifyLoader, DetectLoader, ForgeBatch, ForgeDataset, SegmentLoader
from vjepa_forge.engine.checkpointing import checkpoint_paths, checkpoint_payload, load_checkpoint, resolve_resume_path, resolve_run_dir, results_csv_rows, save_checkpoint, write_results_csv

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class TrainResult:
    loss: float
    steps: int
    epochs: int = 0
    last_checkpoint: str | None = None
    best_checkpoint: str | None = None
    run_dir: str | None = None


class BaseTrainer:
    def __init__(
        self,
        model,
        *,
        data: str,
        epochs: int = 1,
        batch_size: int = 2,
        num_workers: int = 0,
        device: str = "cpu",
        split: str | None = None,
        save: bool = True,
        save_period: int = 0,
        resume: bool | str = False,
        project: str | None = None,
        name: str | None = None,
        exist_ok: bool = False,
    ) -> None:
        self.model = model
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.split = split if split is not None else getattr(self, "split", None)
        self.save = bool(save)
        self.save_period = int(save_period)
        self.resume = resume
        self.project = project
        self.name = name
        self.exist_ok = bool(exist_ok)
        self.run_dir = resolve_run_dir(
            task=str(getattr(self.model, "task", "train")),
            data=self.data,
            project=self.project,
            name=self.name,
            exist_ok=self.exist_ok,
            resume=self.resume,
        )
        self.paths = checkpoint_paths(self.run_dir)

    def build_dataset(self, split: str = "train") -> ForgeDataset:
        return ForgeDataset(self.data, split=split)

    def build_loader(self, split: str = "train"):
        dataset = self.build_dataset(split=split)
        clip_len = int(self.model.data_cfg.get("clip_len", self.model.data_cfg.get("num_frames", 8)))
        clip_stride = int(self.model.data_cfg.get("clip_stride", 1))
        image_size = int(self.model.data_cfg.get("image_size", 384))
        reader_cache_size = int(self.model.data_cfg.get("reader_cache_size", 4))
        video_backend = str(self.model.data_cfg.get("video_backend", "auto"))
        image_backend = str(self.model.data_cfg.get("image_backend", "auto"))
        if dataset.task == "classify":
            collator = ClassifyLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend, image_backend=image_backend)
        elif dataset.task == "detect":
            collator = DetectLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend, image_backend=image_backend)
        elif dataset.task == "segment":
            collator = SegmentLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend, image_backend=image_backend)
        else:
            collator = AnomalyLoader(dataset.media, clip_len=clip_len, clip_stride=clip_stride, image_size=image_size, reader_cache_size=reader_cache_size, video_backend=video_backend, image_backend=image_backend)
        worker_count = int(self.num_workers)
        if dataset.media == "video" and video_backend == "dali":
            worker_count = 0
        if dataset.media == "image" and image_backend == "dali":
            worker_count = 0
        if worker_count <= 0 and dataset.media == "video":
            if video_backend != "dali":
                worker_count = max(1, min(8, os.cpu_count() or 1))
        use_gpu_backend = (dataset.media == "video" and video_backend == "dali") or (dataset.media == "image" and image_backend == "dali")
        pin_memory = bool(self.model.data_cfg.get("pin_memory", torch.cuda.is_available() and not use_gpu_backend))
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

    def build_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.model.data_cfg.get("lr", 1.0e-4))
        weight_decay = float(self.model.data_cfg.get("weight_decay", 1.0e-4))
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def validate_epoch(self) -> float | None:
        try:
            loader = self.build_loader(split="val")
        except Exception:
            return None
        self.model.eval()
        total = 0.0
        batches = 0
        with torch.no_grad():
            progress = self.progress(loader, desc="val", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                outputs = self.model(batch)
                total += float(self.compute_loss(batch, outputs).detach().cpu().item())
                batches += 1
                if batches > 0 and progress is not loader:
                    progress.set_postfix(loss=f"{(total / batches):.4f}")
        return total / max(batches, 1)

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "task": getattr(self.model, "task", None),
            "media": getattr(self.model, "media", None),
            "model": getattr(self.model, "model_cfg", {}),
            "data": getattr(self.model, "data_cfg", {}),
            "train": {
                "data": self.data,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "device": str(self.device),
                "save": self.save,
                "save_period": self.save_period,
                "resume": self.resume if isinstance(self.resume, str) else bool(self.resume),
                "project": self.project,
                "name": self.name,
                "exist_ok": self.exist_ok,
            },
        }

    def _load_resume_state(self, optimizer: torch.optim.Optimizer, scheduler) -> tuple[int, int, float, list[dict[str, Any]]]:
        resume_path = resolve_resume_path(self.resume, run_dir=self.run_dir)
        if resume_path is None:
            return 1, 0, float("inf"), []
        checkpoint = load_checkpoint(resume_path)
        task = str(checkpoint.get("task", getattr(self.model, "task", "")))
        media = str(checkpoint.get("media", getattr(self.model, "media", "")))
        if task != str(getattr(self.model, "task", "")) or media != str(getattr(self.model, "media", "")):
            raise ValueError(f"Checkpoint {resume_path} is incompatible with task={self.model.task} media={self.model.media}")
        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        rows = []
        for row in results_csv_rows(self.paths.results_csv):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                elif key in {"train_loss", "lr", "best_fitness"}:
                    parsed[key] = float(value)
                elif key == "val_loss":
                    parsed[key] = "" if value == "" else float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
        return int(checkpoint["epoch"]) + 1, int(checkpoint.get("global_step", 0)), float(checkpoint.get("best_fitness", float("inf"))), rows

    def _save_epoch_checkpoint(
        self,
        *,
        epoch: int,
        global_step: int,
        best_fitness: float,
        train_loss: float,
        val_loss: float | None,
        optimizer: torch.optim.Optimizer,
        scheduler,
        checkpoint_kind: str,
        target_path: Path,
    ) -> None:
        payload = checkpoint_payload(
            model_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            epoch=epoch,
            global_step=global_step,
            best_fitness=best_fitness,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
            config=self.checkpoint_config(),
            task=str(getattr(self.model, "task", "unknown")),
            media=str(getattr(self.model, "media", "unknown")),
            checkpoint_kind=checkpoint_kind,
        )
        save_checkpoint(payload, target_path)

    def run(self) -> TrainResult:
        self.model.to(self.device)
        loader = self.build_loader(split="train")
        optimizer = self.build_optimizer()
        scheduler = self.build_scheduler(optimizer)
        start_epoch, steps, best_fitness, rows = self._load_resume_state(optimizer, scheduler)
        last_loss = 0.0
        completed_epochs = 0
        for epoch_idx in range(start_epoch, self.epochs + 1):
            self.model.train(True)
            epoch_losses: list[float] = []
            progress = self.progress(loader, desc=f"train {epoch_idx}/{self.epochs}", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch)
                loss = self.compute_loss(batch, outputs)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                epoch_losses.append(last_loss)
                steps += 1
                if tqdm is not None:
                    progress.set_postfix(loss=f"{last_loss:.4f}")
            scheduler.step()
            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            val_loss = self.validate_epoch()
            fitness = float(val_loss if val_loss is not None else train_loss)
            previous_best = best_fitness
            best_fitness = min(best_fitness, fitness)
            rows.append(
                {
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "val_loss": "" if val_loss is None else val_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "best_fitness": best_fitness,
                }
            )
            write_results_csv(self.paths.results_csv, rows)
            if self.save:
                self._save_epoch_checkpoint(
                    epoch=epoch_idx,
                    global_step=steps,
                    best_fitness=best_fitness,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_kind="last",
                    target_path=self.paths.last,
                )
                if fitness <= previous_best:
                    self._save_epoch_checkpoint(
                        epoch=epoch_idx,
                        global_step=steps,
                        best_fitness=best_fitness,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_kind="best",
                        target_path=self.paths.best,
                    )
                if self.save_period > 0 and epoch_idx % self.save_period == 0:
                    self._save_epoch_checkpoint(
                        epoch=epoch_idx,
                        global_step=steps,
                        best_fitness=best_fitness,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_kind="epoch",
                        target_path=self.paths.weights_dir / f"epoch_{epoch_idx:03d}.pt",
                    )
            completed_epochs = epoch_idx
        return TrainResult(
            loss=last_loss,
            steps=steps,
            epochs=completed_epochs,
            last_checkpoint=str(self.paths.last) if self.save else None,
            best_checkpoint=str(self.paths.best) if self.save else None,
            run_dir=str(self.run_dir),
        )
