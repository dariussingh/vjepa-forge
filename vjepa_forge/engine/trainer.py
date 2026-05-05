from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vjepa_forge.data import AnomalyLoader, ClassifyLoader, DetectLoader, ForgeBatch, ForgeDataset, SegmentLoader
from vjepa_forge.engine.checkpointing import checkpoint_paths, checkpoint_payload, load_checkpoint, resolve_resume_path, resolve_run_dir, results_csv_rows, save_checkpoint, write_results_csv
from vjepa_forge.engine.optimization import (
    EarlyStoppingState,
    apply_stage_freeze,
    build_optimizer,
    build_scheduler,
    build_train_settings,
    extract_monitor_value,
    is_improvement,
    normalize_stages,
    update_early_stopping,
)
from vjepa_forge.engine.runtime import broadcast_object, distributed_sampler, setup_runtime
from vjepa_forge.losses.classification import classification_loss

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)

_CORE_RESULTS_COLUMNS = ("epoch", "train_loss", "val_loss", "lr", "best_fitness")


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
        self.runtime = setup_runtime(device=device, data_cfg=getattr(self.model, "data_cfg", {}))
        self.device = self.runtime.device
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
        sampler = None
        if split == "train" and self.runtime.distributed:
            sampler = distributed_sampler(dataset, shuffle=True)
        elif split != "train" and self.runtime.distributed and self.runtime.config.ddp_eval:
            sampler = distributed_sampler(dataset, shuffle=False)
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        else:
            loader_kwargs["shuffle"] = split == "train"
        if worker_count > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)

    def progress(self, iterable, *, desc: str, total: int | None = None):
        if tqdm is None or not self.runtime.is_primary:
            return iterable
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)

    def move_batch_to_device(self, batch: ForgeBatch) -> ForgeBatch:
        batch.x = self.runtime.move_tensor(batch.x)
        return batch

    def forward_pass(self, model, batch: ForgeBatch):
        with self.runtime.autocast_context():
            return model(batch)

    def compute_loss(self, batch: ForgeBatch, outputs) -> torch.Tensor:
        if batch.task == "classify":
            loss, stats = classification_loss(outputs, batch.labels["class_ids"].to(outputs.device), self.model.model_cfg.get("loss"))
            self._last_loss_stats = stats
            return loss
        if batch.task == "detect":
            raise NotImplementedError("Detection training/validation loss is not implemented in the generic Forge trainer")
        if batch.task == "segment":
            raise NotImplementedError("Segmentation training/validation loss is not implemented in the generic Forge trainer")
        targets = batch.labels["targets"].to(outputs.device)
        return F.binary_cross_entropy_with_logits(outputs, targets)

    def _validator_kwargs(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": str(self.device),
            "split": "val",
            "save": self.save,
            "save_period": self.save_period,
            "resume": self.resume,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
        }

    def _parse_results_value(self, key: str, value: str) -> Any:
        if key == "epoch":
            return int(value)
        if value == "":
            return ""
        try:
            return float(value)
        except ValueError:
            return value

    def _ordered_metric_items(self, metrics: dict[str, Any] | None) -> list[tuple[str, Any]]:
        if not metrics:
            return []
        return sorted(metrics.items(), key=lambda item: item[0])

    def _results_row(
        self,
        *,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        lr: float,
        best_fitness: float,
        val_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": "" if val_loss is None else val_loss,
            "lr": lr,
            "best_fitness": best_fitness,
        }
        for key, value in self._ordered_metric_items(val_metrics):
            row[key] = value
        return row

    def _normalized_results_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metric_keys = sorted({key for row in rows for key in row.keys() if key not in _CORE_RESULTS_COLUMNS})
        ordered_keys = [*list(_CORE_RESULTS_COLUMNS), *metric_keys]
        normalized: list[dict[str, Any]] = []
        for row in rows:
            normalized.append({key: row.get(key, "") for key in ordered_keys})
        return normalized

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _emit_epoch_summary(self, *, epoch: int, train_loss: float, val_loss: float | None, val_metrics: dict[str, Any] | None) -> None:
        if not self.runtime.is_primary:
            return
        parts = [f"epoch={epoch}", f"train_loss={train_loss:.4f}"]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        for key, value in self._ordered_metric_items(val_metrics):
            parts.append(f"{key}={self._format_metric_value(value)}")
        summary = " ".join(parts)
        if tqdm is not None:
            tqdm.write(summary)
        else:
            print(summary)

    def validate_epoch(self):
        from vjepa_forge.tasks import TASK_REGISTRY
        from vjepa_forge.engine.validator import ValidationResult

        if self.runtime.distributed and not self.runtime.config.ddp_eval:
            payload = None
            if self.runtime.is_primary:
                try:
                    validator = TASK_REGISTRY[str(getattr(self.model, "task", "classify"))]["val"](self.model, **self._validator_kwargs())
                except (FileNotFoundError, KeyError) as exc:
                    logger.warning("Skipping validation split due to missing validation data: %s", exc)
                else:
                    try:
                        result = validator.run()
                    except (FileNotFoundError, KeyError) as exc:
                        logger.warning("Skipping validation split due to missing validation data: %s", exc)
                    else:
                        payload = {
                            "loss": result.loss,
                            "batches": result.batches,
                            "metrics": result.metrics,
                            "split": result.split,
                        }
            payload = broadcast_object(payload, src=0)
            if payload is None:
                return None
            return ValidationResult(**payload)

        try:
            validator = TASK_REGISTRY[str(getattr(self.model, "task", "classify"))]["val"](self.model, **self._validator_kwargs())
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("Skipping validation split due to missing validation data: %s", exc)
            return None
        try:
            return validator.run()
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("Skipping validation split due to missing validation data: %s", exc)
            return None

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
                parsed[key] = self._parse_results_value(key, value)
            rows.append(parsed)
        return int(checkpoint["epoch"]) + 1, int(checkpoint.get("global_step", 0)), float(checkpoint.get("best_fitness", float("inf"))), rows

    def _save_epoch_checkpoint(
        self,
        *,
        epoch: int,
        global_step: int,
        best_fitness: float,
        stage_name: str,
        train_loss: float,
        val_loss: float | None,
        val_metrics: dict[str, Any] | None,
        optimizer: torch.optim.Optimizer,
        scheduler,
        checkpoint_kind: str,
        target_path: Path,
    ) -> None:
        if not self.runtime.is_primary:
            return
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "stage": stage_name,
        }
        if val_metrics:
            metrics.update(val_metrics)
        payload = checkpoint_payload(
            model_state=self.model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            epoch=epoch,
            global_step=global_step,
            best_fitness=best_fitness,
            metrics=metrics,
            config=self.checkpoint_config(),
            task=str(getattr(self.model, "task", "unknown")),
            media=str(getattr(self.model, "media", "unknown")),
            checkpoint_kind=checkpoint_kind,
            extras={
                "stage_name": stage_name,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            },
        )
        save_checkpoint(payload, target_path)

    def run(self) -> TrainResult:
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = self.model.to(self.device)
        train_model = self.runtime.prepare_module(self.model, training=True)
        loader = self.build_loader(split="train")
        train_settings = build_train_settings(self.model.data_cfg, epochs=self.epochs, batch_size=self.batch_size)
        stages = normalize_stages(task=str(getattr(self.model, "task", "classify")), model=self.model, train_cfg=train_settings, default_epochs=self.epochs, batch_size=self.batch_size)
        apply_stage_freeze(self.model, stages[0])
        optimizer = build_optimizer(self.model, stages[0], batch_size=self.batch_size)
        scheduler = build_scheduler(optimizer, stages[0], steps_per_epoch=len(loader))
        start_epoch, steps, best_fitness, rows = self._load_resume_state(optimizer, scheduler)
        last_loss = 0.0
        completed_epochs = 0
        total_epochs = sum(stage.epochs for stage in stages)
        running_epoch = 0
        global_best_state = None if not hasattr(self.model, "state_dict") else {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        for stage in stages:
            apply_stage_freeze(self.model, stage)
            optimizer = build_optimizer(self.model, stage, batch_size=self.batch_size)
            scheduler = build_scheduler(optimizer, stage, steps_per_epoch=len(loader))
            stage_stop = EarlyStoppingState(best=None, bad_epochs=0, stopped=False)
            for _local_epoch in range(stage.epochs):
                running_epoch += 1
                if running_epoch < start_epoch:
                    continue
                sampler = getattr(loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(running_epoch)
                self.model.train(True)
                train_model.train(True)
                epoch_losses: list[float] = []
                progress = self.progress(loader, desc=f"{stage.name} {running_epoch}/{total_epochs}", total=len(loader))
                stage_global_step = scheduler.current_step
                for batch in progress:
                    batch = self.move_batch_to_device(batch)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.forward_pass(train_model, batch)
                    with self.runtime.autocast_context():
                        loss = self.compute_loss(batch, outputs)
                    if self.runtime.scaler is not None:
                        self.runtime.scaler.scale(loss).backward()
                        self.runtime.scaler.step(optimizer)
                        self.runtime.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    current_lr = scheduler.step(stage_global_step)
                    stage_global_step += 1
                    last_loss = float(loss.detach().cpu().item())
                    epoch_losses.append(last_loss)
                    steps += 1
                    if tqdm is not None:
                        progress.set_postfix(loss=f"{last_loss:.4f}", lr=f"{current_lr:.2e}")
                train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
                val_result = self.validate_epoch()
                val_loss = None if val_result is None else float(val_result.loss)
                val_metrics = None if val_result is None else val_result.metrics
                monitor_value = train_loss if val_result is None else extract_monitor_value(val_result, stage.monitor)
                previous_best = best_fitness
                fitness = monitor_value if stage.monitor.mode == "min" else -monitor_value
                best_fitness = min(best_fitness, fitness)
                row = self._results_row(
                    epoch=running_epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=float(max(group["lr"] for group in optimizer.param_groups)),
                    best_fitness=best_fitness,
                    val_metrics=val_metrics,
                )
                row["stage"] = stage.name
                row["monitor_metric"] = stage.monitor.metric
                row["monitor_value"] = monitor_value
                for group in optimizer.param_groups:
                    row[f"lr_{group.get('group_name', 'group')}"] = float(group["lr"])
                rows.append(row)
                rows = self._normalized_results_rows(rows)
                self._emit_epoch_summary(epoch=running_epoch, train_loss=train_loss, val_loss=val_loss, val_metrics=(val_metrics or {}) | {"monitor": monitor_value})
                if self.runtime.is_primary:
                    write_results_csv(self.paths.results_csv, rows)
                improved = is_improvement(monitor_value, stage_stop.best, stage.monitor, min_delta=stage.early_stopping.min_delta)
                stage_stop = update_early_stopping(stage_stop, value=monitor_value, stage=stage, completed_stage_epochs=_local_epoch + 1)
                if improved:
                    global_best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                if self.save:
                    self._save_epoch_checkpoint(
                        epoch=running_epoch,
                        global_step=steps,
                        best_fitness=best_fitness,
                        stage_name=stage.name,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_metrics=(val_metrics or {}) | {"monitor_value": monitor_value},
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_kind="last",
                        target_path=self.paths.last,
                    )
                    if best_fitness <= previous_best:
                        self._save_epoch_checkpoint(
                            epoch=running_epoch,
                            global_step=steps,
                            best_fitness=best_fitness,
                            stage_name=stage.name,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            val_metrics=(val_metrics or {}) | {"monitor_value": monitor_value},
                            optimizer=optimizer,
                            scheduler=scheduler,
                            checkpoint_kind="best",
                            target_path=self.paths.best,
                        )
                    if self.save_period > 0 and running_epoch % self.save_period == 0:
                        self._save_epoch_checkpoint(
                            epoch=running_epoch,
                            global_step=steps,
                            best_fitness=best_fitness,
                            stage_name=stage.name,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            val_metrics=(val_metrics or {}) | {"monitor_value": monitor_value},
                            optimizer=optimizer,
                            scheduler=scheduler,
                            checkpoint_kind="epoch",
                            target_path=self.paths.weights_dir / f"epoch_{running_epoch:03d}.pt",
                        )
                completed_epochs = running_epoch
                if stage_stop.stopped:
                    break
            if stage_stop.stopped and stage.early_stopping.scope == "every_stage":
                break
        if global_best_state is not None and stages[-1].early_stopping.restore_best:
            self.model.load_state_dict(global_best_state, strict=False)
        return TrainResult(
            loss=last_loss,
            steps=steps,
            epochs=completed_epochs,
            last_checkpoint=str(self.paths.last) if self.save else None,
            best_checkpoint=str(self.paths.best) if self.save else None,
            run_dir=str(self.run_dir),
        )
