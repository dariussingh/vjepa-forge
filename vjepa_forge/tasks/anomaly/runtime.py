from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vjepa_forge.data.forge.dataset import ForgeDataset
from vjepa_forge.data.video import get_video_frame_count, read_video_clip, read_video_clips, read_video_frames_uint8
from vjepa_forge.engine.checkpointing import checkpoint_paths, checkpoint_payload, load_checkpoint, resolve_resume_path, resolve_run_dir, results_csv_rows, save_checkpoint, write_results_csv
from vjepa_forge.engine.optimization import build_scheduler, build_train_settings, normalize_stages, resolve_autoscaled_lr
from vjepa_forge.engine.runtime import setup_runtime
from vjepa_forge.heads.anomaly.modeling import ExtractedFeatures, build_feature_extractor, build_predictor
from vjepa_forge.losses.anomaly import anomaly_future_prediction_loss

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class VideoClipRecord:
    name: str
    media_path: str
    frame_count: int
    frame_labels: tuple[int, ...] | None


@dataclass(frozen=True)
class WindowRecord:
    video_name: str
    past_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    future_labels: tuple[int, ...] | None


@dataclass
class AnomalyTrainResult:
    best_val_loss: float
    best_checkpoint: str
    last_checkpoint: str
    run_dir: str


@dataclass
class AnomalyValidationResult:
    split: str
    metrics: dict[str, Any]
    report_path: str


@dataclass
class AnomalyPredictResult:
    split: str | None
    metrics: dict[str, Any]
    report_path: str
    rendered_outputs: list[str] | None = None


@dataclass
class AnomalyExportResult:
    output_path: str
    checkpoint_path: str


class ForgeAnomalyWindowDataset(Dataset):
    def __init__(self, videos: list[VideoClipRecord], windows: list[WindowRecord], image_size: int, video_backend: str = "auto") -> None:
        self.video_lookup = {video.name: video for video in videos}
        self.windows = windows
        self.image_size = image_size
        self.reader_cache_size = 4
        self.video_backend = video_backend

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        record = self.video_lookup[window.video_name]
        sample: dict[str, Any] = {
            "video_name": record.name,
            "media_path": record.media_path,
            "clip_start": int(window.past_indices[0]),
            "clip_len": len(window.past_indices) + len(window.future_indices),
            "past_len": len(window.past_indices),
            "future_indices": torch.tensor(window.future_indices, dtype=torch.long),
        }
        if window.future_labels is not None:
            sample["future_labels"] = torch.tensor(window.future_labels, dtype=torch.long)
        return sample


class _WindowBatchSampler:
    def __init__(self, windows: list[WindowRecord], *, batch_size: int, shuffle: bool) -> None:
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        grouped: dict[str, list[int]] = {}
        for idx, window in enumerate(windows):
            grouped.setdefault(window.video_name, []).append(idx)
        self.groups = list(grouped.values())

    def __iter__(self):
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            random.shuffle(groups)
            for group in groups:
                if len(group) > self.batch_size:
                    chunks = [group[i : i + self.batch_size] for i in range(0, len(group), self.batch_size)]
                    random.shuffle(chunks)
                    group[:] = [idx for chunk in chunks for idx in chunk]
        batches: list[list[int]] = []
        for group in groups:
            for start in range(0, len(group), self.batch_size):
                batches.append(group[start : start + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


class _InferenceWrapper(nn.Module):
    def __init__(self, feature_extractor: nn.Module, predictor: nn.Module, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.model_cfg = model_cfg
        self.tubelet_size = feature_extractor.tubelet_size

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        past_feat = self.feature_extractor(past)
        future_feat = self.feature_extractor(future)
        return _predict_sample_scores(
            self.predictor,
            past_feat,
            future_feat,
            self.model_cfg,
            tubelet_size=self.tubelet_size,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_output_root(cfg: dict[str, Any]) -> Path:
    source = cfg.get("predict", {}).get("source") if cfg.get("action") == "predict" else None
    return resolve_run_dir(
        task="anomaly",
        data=source or cfg["dataset"]["dataset_yaml"],
        project=cfg["train"].get("project") or cfg.get("output", {}).get("root"),
        name=cfg["train"].get("name"),
        exist_ok=bool(cfg["train"].get("exist_ok", False) or cfg.get("action") != "train"),
        resume=cfg["train"].get("resume", False) if cfg.get("action") == "train" else True,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_cfg(config: dict[str, Any], *, action: str) -> dict[str, Any]:
    model_cfg = dict(config["model"])
    if "name" not in model_cfg and "name_full" in model_cfg:
        model_cfg["name"] = model_cfg["name_full"]
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    data_cfg = dict(config["data"])
    dataset_yaml = data_cfg.get("_path") or data_cfg.get("dataset_yaml") or data_cfg.get("path")
    source = config.get("predict", {}).get("source")
    if not dataset_yaml and not (action == "predict" and source):
        raise ValueError("Anomaly runtime requires data._path or data.dataset_yaml")
    output_root = config.get("output", {}).get("root")
    default_workers = max(1, min(8, os.cpu_count() or 1))
    return {
        "action": action,
        "dataset": {
            "dataset_yaml": None if dataset_yaml is None else str(dataset_yaml),
            "image_size": int(data_cfg.get("image_size", 384)),
            "past_frames": int(data_cfg.get("past_frames", data_cfg.get("num_frames", 8))),
            "future_frames": int(data_cfg.get("future_frames", data_cfg.get("num_frames", 8))),
            "stride": int(data_cfg.get("stride", 1)),
            "video_backend": str(data_cfg.get("video_backend", "auto")),
        },
        "model": {
            "name": str(model_cfg.get("name", "vjepa2_1_vit_base_384")),
            "checkpoint": str(backbone_cfg.get("checkpoint")),
            "checkpoint_key": str(backbone_cfg.get("checkpoint_key", "ema_encoder")),
            "predictor_type": str(model_cfg.get("predictor_type", "vit_patch")),
            "hidden_dim": int(model_cfg.get("hidden_dim", 1024)),
            "dropout": float(model_cfg.get("dropout", 0.1)),
            "predictor_embed_dim": int(model_cfg.get("predictor_embed_dim", 768)),
            "predictor_depth": int(model_cfg.get("predictor_depth", 12)),
            "predictor_num_heads": int(model_cfg.get("predictor_num_heads", 12)),
            "predictor_use_rope": bool(model_cfg.get("predictor_use_rope", True)),
            "token_aggregation": str(model_cfg.get("token_aggregation", "topk_mean")),
            "token_topk_fraction": float(model_cfg.get("token_topk_fraction", 0.1)),
        },
        "train": {
            "batch_size": int(config["train"].get("batch_size", 1)),
            "epochs": int(config["train"].get("epochs", 10)),
            "save": bool(config["train"].get("save", True)),
            "save_period": int(config["train"].get("save_period", 0)),
            "resume": config["train"].get("resume", False),
            "project": config["train"].get("project"),
            "name": config["train"].get("name"),
            "exist_ok": bool(config["train"].get("exist_ok", False)),
            "lr_mode": str(config["train"].get("lr_mode", "manual")),
            "lr": float(config["train"].get("lr", 1.0e-4)),
            "reference_batch_size": int(config["train"].get("reference_batch_size", config["train"].get("batch_size", 1))),
            "reference_lr": float(config["train"].get("reference_lr", config["train"].get("lr", 1.0e-4))),
            "lr_scale_rule": str(config["train"].get("lr_scale_rule", "sqrt")),
            "weight_decay": float(config["train"].get("weight_decay", 1.0e-4)),
            "num_workers": int(config["train"].get("num_workers", default_workers)),
            "prefetch_factor": int(config["train"].get("prefetch_factor", 2)),
            "persistent_workers": bool(config["train"].get("persistent_workers", True)),
            "pin_memory": bool(config["train"].get("pin_memory", torch.cuda.is_available())),
            "reader_cache_size": int(config["train"].get("reader_cache_size", 4)),
            "device": str(config["train"].get("device", "cpu")),
            "seed": int(config["train"].get("seed", 7)),
            "save_latest_every_epoch": bool(config["train"].get("save_latest_every_epoch", True)),
            "save_epoch_checkpoints": bool(config["train"].get("save_epoch_checkpoints", False)),
            "scheduler": dict(config["train"].get("scheduler", {})),
            "early_stopping": dict(config["train"].get("early_stopping", {})),
        },
        "eval": {
            "batch_size": int(config["val"].get("batch_size", 1)),
            "num_workers": int(config["val"].get("num_workers", default_workers)),
            "prefetch_factor": int(config["val"].get("prefetch_factor", 2)),
            "persistent_workers": bool(config["val"].get("persistent_workers", True)),
            "pin_memory": bool(config["val"].get("pin_memory", torch.cuda.is_available())),
            "reader_cache_size": int(config["val"].get("reader_cache_size", 4)),
            "threshold_std_multiplier": float(config["val"].get("threshold_std_multiplier", 3.0)),
            "smoothing_window": int(config["val"].get("smoothing_window", 9)),
            "checkpoint_target": str(config["val"].get("checkpoint_target", "best")),
            "checkpoint_path": config["val"].get("checkpoint_path"),
            "split": str(config["val"].get("split", "val" if action == "val" else "test")),
        },
        "predict": {
            "batch_size": int(config.get("predict", {}).get("batch_size", config["val"].get("batch_size", 1))),
            "num_workers": int(config.get("predict", {}).get("num_workers", config["val"].get("num_workers", default_workers))),
            "split": str(config.get("predict", {}).get("split", "test")),
            "threshold": config.get("predict", {}).get("threshold"),
            "visualize": bool(config.get("predict", {}).get("visualize", False)),
            "source": source,
            "output_dir": config.get("predict", {}).get("output_dir"),
        },
        "export": {
            "format": str(config["export"].get("format", "onnx")),
            "output_path": str(config["export"].get("output_path", "anomaly.onnx")),
            "opset": int(config["export"].get("opset", 17)),
            "dynamic_axes": bool(config["export"].get("dynamic_axes", True)),
            "checkpoint_target": str(config["export"].get("checkpoint_target", config["val"].get("checkpoint_target", "best"))),
            "checkpoint_path": config["export"].get("checkpoint_path"),
        },
        "output": {"root": output_root},
        "distributed": dict(config.get("distributed", {})),
    }


def _build_video_records(dataset_yaml: str | Path, split: str) -> list[VideoClipRecord]:
    dataset = ForgeDataset(dataset_yaml, split=split)
    records: list[VideoClipRecord] = []
    for record in dataset.records:
        labels = [0] * get_video_frame_count(
            record.media_path,
            reader_cache_size=4,
            video_backend="decord",
        )
        for annotation in record.annotations:
            if annotation.op != "ano":
                continue
            payload = annotation.payload
            if payload.get("status") != "abnormal":
                continue
            start = max(0, int(payload.get("start_frame", 0)))
            end = min(len(labels) - 1, int(payload.get("end_frame", -1)))
            for idx in range(start, end + 1):
                labels[idx] = 1
        records.append(
            VideoClipRecord(
                name=Path(record.media_path).stem,
                media_path=record.media_path,
                frame_count=len(labels),
                frame_labels=tuple(labels),
            )
        )
    return records


def _build_source_record(source: str | Path, *, video_backend: str) -> VideoClipRecord:
    source_path = Path(source)
    frame_count = get_video_frame_count(source_path, reader_cache_size=4, video_backend=video_backend)
    return VideoClipRecord(
        name=source_path.stem,
        media_path=str(source_path),
        frame_count=frame_count,
        frame_labels=None,
    )


def _build_window_records(videos: list[VideoClipRecord], past_frames: int, future_frames: int, stride: int) -> list[WindowRecord]:
    if past_frames != future_frames:
        raise ValueError("Active anomaly runtime requires past_frames == future_frames")
    total = past_frames + future_frames
    windows: list[WindowRecord] = []
    for record in videos:
        if record.frame_count < total:
            continue
        for start in range(0, record.frame_count - total + 1, stride):
            past = tuple(range(start, start + past_frames))
            future = tuple(range(start + past_frames, start + total))
            future_labels = None if record.frame_labels is None else tuple(record.frame_labels[idx] for idx in future)
            windows.append(
                WindowRecord(
                    video_name=record.name,
                    past_indices=past,
                    future_indices=future,
                    future_labels=future_labels,
                )
            )
    if not windows:
        raise RuntimeError("No anomaly windows could be built from the Forge dataset")
    return windows


def _collate_window_batch(batch: list[dict[str, Any]], *, image_size: int, reader_cache_size: int, video_backend: str) -> dict[str, Any]:
    if not batch:
        return {}
    decode_start = time.perf_counter()

    # Group by video path so each video is decoded once.
    # _WindowBatchSampler already co-locates windows from the same video,
    # so typically all batch items share one path and only the minimal
    # contiguous frame range needs to be read (O(B+T) instead of O(B*T)).
    groups: dict[str, list[int]] = {}
    for idx, sample in enumerate(batch):
        groups.setdefault(sample["media_path"], []).append(idx)

    decoded_clips: list[torch.Tensor] = [torch.empty(0)] * len(batch)
    for path, indices in groups.items():
        min_start = min(int(batch[i]["clip_start"]) for i in indices)
        max_end = max(int(batch[i]["clip_start"]) + int(batch[i]["clip_len"]) for i in indices)
        full = read_video_clip(
            path,
            clip_start=min_start,
            clip_len=max_end - min_start,
            stride=1,
            image_size=image_size,
            reader_cache_size=reader_cache_size,
            video_backend=video_backend,
        )
        for i in indices:
            s = int(batch[i]["clip_start"]) - min_start
            decoded_clips[i] = full[s : s + int(batch[i]["clip_len"])]

    past_lens = [int(sample["past_len"]) for sample in batch]
    future_len = max(0, int(batch[0]["clip_len"]) - past_lens[0])
    past = torch.stack([decoded_clips[i][:pl].permute(1, 0, 2, 3).contiguous() for i, pl in enumerate(past_lens)], dim=0)
    future = torch.stack([decoded_clips[i][pl : pl + future_len].permute(1, 0, 2, 3).contiguous() for i, pl in enumerate(past_lens)], dim=0)
    collated: dict[str, Any] = {
        "past": past,
        "future": future,
        "video_name": [sample["video_name"] for sample in batch],
        "future_indices": torch.stack([sample["future_indices"] for sample in batch], dim=0),
        "decode_time": float(time.perf_counter() - decode_start),
    }
    if "future_labels" in batch[0]:
        collated["future_labels"] = torch.stack([sample["future_labels"] for sample in batch], dim=0)
    return collated


def _loader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    collate_fn,
    batch_sampler=None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "collate_fn": collate_fn,
        "pin_memory": bool(pin_memory),
    }
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = int(batch_size)
        kwargs["shuffle"] = False
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _make_loaders(cfg: dict[str, Any], include_test: bool = True) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    train_videos = _build_video_records(dataset_cfg["dataset_yaml"], split="train")
    val_split = cfg["eval"].get("split", "val")
    val_videos = _build_video_records(dataset_cfg["dataset_yaml"], split=val_split)
    test_videos = _build_video_records(dataset_cfg["dataset_yaml"], split="test") if include_test else []
    common = {
        "past_frames": dataset_cfg["past_frames"],
        "future_frames": dataset_cfg["future_frames"],
        "stride": dataset_cfg["stride"],
    }
    train_windows = _build_window_records(train_videos, **common)
    val_windows = _build_window_records(val_videos, **common)
    test_windows = _build_window_records(test_videos, **common) if include_test else []
    image_size = dataset_cfg["image_size"]
    video_backend = str(dataset_cfg.get("video_backend", "auto"))
    train_ds = ForgeAnomalyWindowDataset(train_videos, train_windows, image_size, video_backend=video_backend)
    val_ds = ForgeAnomalyWindowDataset(val_videos, val_windows, image_size, video_backend=video_backend)
    test_ds = ForgeAnomalyWindowDataset(test_videos, test_windows, image_size, video_backend=video_backend) if include_test else None
    train_ds.reader_cache_size = int(cfg["train"]["reader_cache_size"])
    val_ds.reader_cache_size = int(cfg["eval"]["reader_cache_size"])
    if test_ds is not None:
        test_ds.reader_cache_size = int(cfg["eval"]["reader_cache_size"])
    train_num_workers = int(cfg["train"]["num_workers"])
    eval_num_workers = int(cfg["eval"]["num_workers"])
    train_collate = partial(
        _collate_window_batch,
        image_size=image_size,
        reader_cache_size=int(cfg["train"]["reader_cache_size"]),
        video_backend=video_backend,
    )
    eval_collate = partial(
        _collate_window_batch,
        image_size=image_size,
        reader_cache_size=int(cfg["eval"]["reader_cache_size"]),
        video_backend=video_backend,
    )
    train_loader_kwargs = _loader_kwargs(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=train_num_workers,
        pin_memory=bool(cfg["train"]["pin_memory"] and video_backend != "dali"),
        persistent_workers=bool(cfg["train"]["persistent_workers"]),
        prefetch_factor=int(cfg["train"]["prefetch_factor"]),
        collate_fn=train_collate,
        batch_sampler=_WindowBatchSampler(train_windows, batch_size=int(cfg["train"]["batch_size"]), shuffle=True),
    )
    eval_loader_kwargs = _loader_kwargs(
        batch_size=int(cfg["eval"]["batch_size"]),
        num_workers=eval_num_workers,
        pin_memory=bool(cfg["eval"]["pin_memory"] and video_backend != "dali"),
        persistent_workers=bool(cfg["eval"]["persistent_workers"]),
        prefetch_factor=int(cfg["eval"]["prefetch_factor"]),
        collate_fn=eval_collate,
        batch_sampler=_WindowBatchSampler(val_windows, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False),
    )
    loaders: dict[str, Any] = {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "train_loader": DataLoader(train_ds, **train_loader_kwargs),
        "val_loader": DataLoader(val_ds, **eval_loader_kwargs),
    }
    if test_ds is not None:
        test_loader_kwargs = _loader_kwargs(
            batch_size=int(cfg["eval"]["batch_size"]),
            num_workers=eval_num_workers,
            pin_memory=bool(cfg["eval"]["pin_memory"] and video_backend != "dali"),
            persistent_workers=bool(cfg["eval"]["persistent_workers"]),
            prefetch_factor=int(cfg["eval"]["prefetch_factor"]),
            collate_fn=eval_collate,
            batch_sampler=_WindowBatchSampler(test_windows, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False),
        )
        loaders["test_loader"] = DataLoader(test_ds, **test_loader_kwargs)
    return loaders


def _build_eval_loader(
    videos: list[VideoClipRecord],
    cfg: dict[str, Any],
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    dataset_cfg = cfg["dataset"]
    windows = _build_window_records(
        videos,
        past_frames=dataset_cfg["past_frames"],
        future_frames=dataset_cfg["future_frames"],
        stride=dataset_cfg["stride"],
    )
    video_backend = str(dataset_cfg.get("video_backend", "auto"))
    ds = ForgeAnomalyWindowDataset(videos, windows, dataset_cfg["image_size"], video_backend=video_backend)
    ds.reader_cache_size = int(cfg["eval"]["reader_cache_size"])
    worker_count = int(cfg["eval"]["num_workers"] if num_workers is None else num_workers)
    loader_kwargs = _loader_kwargs(
        batch_size=int(cfg["eval"]["batch_size"] if batch_size is None else batch_size),
        num_workers=worker_count,
        pin_memory=bool(cfg["eval"]["pin_memory"] and video_backend != "dali"),
        persistent_workers=bool(cfg["eval"]["persistent_workers"]),
        prefetch_factor=int(cfg["eval"]["prefetch_factor"]),
        collate_fn=partial(
            _collate_window_batch,
            image_size=dataset_cfg["image_size"],
            reader_cache_size=int(cfg["eval"]["reader_cache_size"]),
            video_backend=video_backend,
        ),
        batch_sampler=_WindowBatchSampler(windows, batch_size=int(cfg["eval"]["batch_size"] if batch_size is None else batch_size), shuffle=False),
    )
    return DataLoader(ds, **loader_kwargs)


def _extract_pair_features(feature_extractor: nn.Module, batch: dict[str, Any], runtime) -> tuple[ExtractedFeatures, ExtractedFeatures]:
    past = runtime.move_tensor(batch["past"])
    future = runtime.move_tensor(batch["future"])
    with torch.no_grad():
        with runtime.autocast_context():
            past_feat = feature_extractor(past)
            future_feat = feature_extractor(future)
    return past_feat, future_feat


def _mse_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=1)


def _score_tokens(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, *, tubelet_size: int, aggregation: str, topk_fraction: float) -> torch.Tensor:
    token_errors = ((pred_tokens - target_tokens) ** 2).mean(dim=-1)
    if aggregation == "mean":
        temporal_scores = token_errors.mean(dim=-1)
    elif aggregation == "max":
        temporal_scores = token_errors.max(dim=-1).values
    elif aggregation == "topk_mean":
        k = max(1, int(round(token_errors.size(-1) * topk_fraction)))
        temporal_scores = torch.topk(token_errors, k=k, dim=-1).values.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported token aggregation: {aggregation}")
    return temporal_scores.repeat_interleave(tubelet_size, dim=1)


def _predict_sample_scores(predictor: nn.Module, past_features: ExtractedFeatures, future_features: ExtractedFeatures, model_cfg: dict[str, Any], *, tubelet_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    predictor_type = model_cfg["predictor_type"]
    if predictor_type == "global_mlp":
        pred_future = predictor(past_features.pooled)
        predictor_scores = _mse_score(pred_future, future_features.pooled)
        frozen_scores = _mse_score(past_features.pooled, future_features.pooled)
        frame_count = future_features.tokens.size(1) * tubelet_size
        return predictor_scores.unsqueeze(1).repeat(1, frame_count), frozen_scores.unsqueeze(1).repeat(1, frame_count)
    if predictor_type == "vit_patch":
        pred_tokens = predictor(past_features.tokens)
        predictor_scores = _score_tokens(
            pred_tokens,
            future_features.tokens,
            tubelet_size=tubelet_size,
            aggregation=str(model_cfg.get("token_aggregation", "topk_mean")),
            topk_fraction=float(model_cfg.get("token_topk_fraction", 0.1)),
        )
        past_reference = past_features.tokens.mean(dim=1, keepdim=True).expand_as(future_features.tokens)
        frozen_scores = _score_tokens(
            past_reference,
            future_features.tokens,
            tubelet_size=tubelet_size,
            aggregation=str(model_cfg.get("token_aggregation", "topk_mean")),
            topk_fraction=float(model_cfg.get("token_topk_fraction", 0.1)),
        )
        return predictor_scores, frozen_scores
    raise ValueError(f"Unsupported predictor_type: {predictor_type}")


def _normal_stats(scores: np.ndarray) -> dict[str, float]:
    return {"mean": float(scores.mean()) if len(scores) else 0.0, "std": float(scores.std()) if len(scores) else 0.0}


def _timing_metrics(*, decode_times: list[float], model_times: list[float]) -> dict[str, float]:
    avg_decode = float(np.mean(decode_times)) if decode_times else 0.0
    avg_model = float(np.mean(model_times)) if model_times else 0.0
    return {
        "avg_decode_time": avg_decode,
        "avg_model_time": avg_model,
        "avg_step_time": avg_decode + avg_model,
    }


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(scores) == 0:
        return scores.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(scores.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _aggregate_scores(loader: DataLoader, predictor: nn.Module, feature_extractor: nn.Module, runtime, desc: str, model_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    by_video: dict[str, dict[str, Any]] = {}
    predictor.eval()
    decode_times: list[float] = []
    model_times: list[float] = []
    for batch in _progress(loader, desc=desc, total=len(loader)):
        decode_times.append(float(batch.get("decode_time", 0.0)))
        model_start = time.perf_counter()
        past_feat, future_feat = _extract_pair_features(feature_extractor, batch, runtime)
        with runtime.autocast_context():
            predictor_scores_t, frozen_scores_t = _predict_sample_scores(
                predictor,
                past_feat,
                future_feat,
                model_cfg,
                tubelet_size=feature_extractor.tubelet_size,
            )
        model_times.append(float(time.perf_counter() - model_start))
        predictor_scores = predictor_scores_t.detach().cpu().numpy()
        frozen_scores = frozen_scores_t.detach().cpu().numpy()
        future_indices = batch["future_indices"].numpy()
        labels = batch.get("future_labels")
        labels_np = labels.numpy() if labels is not None else None
        video_names = batch["video_name"]
        for i, video_name in enumerate(video_names):
            state = by_video.setdefault(video_name, {"predictor_sum": {}, "predictor_count": {}, "frozen_sum": {}, "frozen_count": {}, "labels": {}, "has_labels": False})
            for local_idx, frame_idx in enumerate(future_indices[i]):
                idx = int(frame_idx)
                state["predictor_sum"][idx] = state["predictor_sum"].get(idx, 0.0) + float(predictor_scores[i, local_idx])
                state["predictor_count"][idx] = state["predictor_count"].get(idx, 0) + 1
                state["frozen_sum"][idx] = state["frozen_sum"].get(idx, 0.0) + float(frozen_scores[i, local_idx])
                state["frozen_count"][idx] = state["frozen_count"].get(idx, 0) + 1
            if labels_np is not None:
                state["has_labels"] = True
                for frame_idx, label in zip(future_indices[i], labels_np[i]):
                    state["labels"][int(frame_idx)] = int(label)
    summary = _finalize_video_summary(by_video)
    return summary, _timing_metrics(decode_times=decode_times, model_times=model_times)


def _finalize_video_summary(by_video: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"videos": {}}
    for video_name, state in by_video.items():
        frame_ids = sorted(state["predictor_sum"].keys())
        predictor_series = np.asarray([state["predictor_sum"][idx] / state["predictor_count"][idx] for idx in frame_ids], dtype=np.float32)
        frozen_series = np.asarray([state["frozen_sum"][idx] / state["frozen_count"][idx] for idx in frame_ids], dtype=np.float32)
        labels = None
        if state["has_labels"]:
            labels = np.asarray([state["labels"].get(idx, 0) for idx in frame_ids], dtype=np.int64)
        summary["videos"][video_name] = {
            "frame_ids": frame_ids,
            "predictor_scores": predictor_series.tolist(),
            "frozen_scores": frozen_series.tolist(),
            "labels": None if labels is None else labels.tolist(),
        }
    return summary


def _flatten_metric_arrays(video_summary: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for video in video_summary["videos"].values():
        if video["labels"] is None:
            continue
        scores.extend(video[key])
        labels.extend(video["labels"])
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def _build_smoothed_summary(video_summary: dict[str, Any], smoothing_window: int) -> dict[str, Any]:
    smoothed: dict[str, Any] = {"videos": {}}
    for video_name, payload in video_summary["videos"].items():
        predictor_scores = np.asarray(payload["predictor_scores"], dtype=np.float32)
        frozen_scores = np.asarray(payload["frozen_scores"], dtype=np.float32)
        smoothed["videos"][video_name] = {
            "frame_ids": payload["frame_ids"],
            "predictor_scores": _smooth_scores(predictor_scores, smoothing_window).tolist(),
            "frozen_scores": _smooth_scores(frozen_scores, smoothing_window).tolist(),
            "labels": payload["labels"],
        }
    return smoothed


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _clip_score_rows(video_summary: dict[str, Any], key: str, *, reduction: str = "max") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_name, payload in video_summary["videos"].items():
        scores = np.asarray(payload[key], dtype=np.float32)
        labels = None if payload["labels"] is None else np.asarray(payload["labels"], dtype=np.int64)
        if reduction == "max":
            clip_score = float(scores.max()) if scores.size else float("nan")
        elif reduction == "mean":
            clip_score = float(scores.mean()) if scores.size else float("nan")
        else:
            raise ValueError(f"Unsupported clip score reduction: {reduction}")
        clip_label = None if labels is None else int(np.any(labels != 0))
        rows.append(
            {
                "video_name": video_name,
                "clip_label": clip_label,
                "clip_score": clip_score,
            }
        )
    return rows


def _clip_level_metrics(video_summary: dict[str, Any], key: str, *, threshold: float, reduction: str = "max") -> dict[str, Any]:
    rows = _clip_score_rows(video_summary, key, reduction=reduction)
    labeled_rows = [row for row in rows if row["clip_label"] is not None]
    labels = np.asarray([row["clip_label"] for row in labeled_rows], dtype=np.int64)
    scores = np.asarray([row["clip_score"] for row in labeled_rows], dtype=np.float32)
    predictions = (scores > float(threshold)).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, len(rows))
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0.0 else 0.0
    return {
        "reduction": reduction,
        "auc": _roc_auc_score(labels, scores),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "counts": {
            "total": int(len(labeled_rows)),
            "normal": int(np.sum(labels == 0)),
            "anomaly": int(np.sum(labels == 1)),
        },
        "clips": rows,
    }


def _threshold_clip_predictions(video_summary: dict[str, Any], key: str, *, threshold: float, reduction: str = "max") -> dict[str, Any]:
    rows = _clip_score_rows(video_summary, key, reduction=reduction)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        label = row["clip_label"]
        clip_score = float(row["clip_score"])
        predicted_label = int(clip_score > float(threshold))
        enriched.append(
            {
                "video_name": row["video_name"],
                "clip_score": clip_score,
                "clip_label": label,
                "predicted_label": predicted_label,
                "threshold": float(threshold),
            }
        )
    return {
        "clip_score_reduction": reduction,
        "threshold": float(threshold),
        "clips": enriched,
    }


def _predict_output_root(cfg: dict[str, Any], *, split: str | None, source: str | None) -> Path:
    output_root = _make_output_root(cfg)
    base = output_root / "predict"
    if source:
        return base / Path(source).stem
    return base / str(split or "custom")


def _render_timeline_strip(
    image: np.ndarray,
    *,
    scores: np.ndarray,
    current_index: int,
    threshold: float,
    labels: np.ndarray | None,
) -> np.ndarray:
    height, width = image.shape[:2]
    strip_h = max(28, height // 10)
    y0 = height - strip_h - 8
    x0 = 8
    x1 = width - 8
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y0 + strip_h), (15, 15, 15), -1)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)
    if labels is not None and labels.size:
        for idx, value in enumerate(labels):
            if int(value) <= 0:
                continue
            px0 = int(x0 + idx * (x1 - x0) / max(labels.size, 1))
            px1 = int(x0 + (idx + 1) * (x1 - x0) / max(labels.size, 1))
            cv2.rectangle(image, (px0, y0), (px1, y0 + strip_h), (40, 60, 180), -1)
    if scores.size:
        vmax = max(float(scores.max()), threshold, 1e-6)
        points = []
        for idx, score in enumerate(scores):
            px = int(x0 + idx * (x1 - x0) / max(scores.size - 1, 1))
            py = int(y0 + strip_h - 4 - (float(score) / vmax) * max(strip_h - 8, 1))
            points.append((px, py))
        if len(points) > 1:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (255, 220, 0), 2)
        tx_y = int(y0 + strip_h - 4 - (float(threshold) / vmax) * max(strip_h - 8, 1))
        cv2.line(image, (x0, tx_y), (x1, tx_y), (0, 0, 255), 1)
        cx = int(x0 + current_index * (x1 - x0) / max(scores.size - 1, 1))
        cv2.line(image, (cx, y0), (cx, y0 + strip_h), (255, 255, 255), 1)
    return image


def _render_prediction_video(
    *,
    source_path: str,
    output_path: Path,
    predictor_scores: list[float],
    threshold: float,
    labels: list[int] | None,
) -> None:
    frames = read_video_frames_uint8(source_path)
    if frames.size == 0:
        raise RuntimeError(f"No frames available for visualization: {source_path}")
    frame_count = min(len(frames), len(predictor_scores))
    frames = frames[:frame_count]
    scores = np.asarray(predictor_scores[:frame_count], dtype=np.float32)
    label_arr = None if labels is None else np.asarray(labels[:frame_count], dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for idx in range(frame_count):
            rgb = frames[idx].copy()
            score = float(scores[idx])
            predicted = score > float(threshold)
            if label_arr is not None and int(label_arr[idx]) > 0:
                overlay = rgb.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (255, 96, 96), -1)
                rgb = cv2.addWeighted(overlay, 0.12, rgb, 0.88, 0.0)
            status = "ANOMALY" if predicted else "NORMAL"
            color = (0, 0, 255) if predicted else (0, 200, 0)
            cv2.putText(rgb, f"score={score:.4f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb, f"threshold={float(threshold):.4f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb, status, (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            rgb = _render_timeline_strip(rgb, scores=scores, current_index=idx, threshold=float(threshold), labels=label_arr)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _checkpoint_payload(predictor: nn.Module, cfg: dict[str, Any], *, epoch: int, train_loss: float, val_loss: float, best_val_loss: float, effective_lr: float, checkpoint_kind: str) -> dict[str, Any]:
    return checkpoint_payload(
        model_state=predictor.state_dict(),
        optimizer_state=None,
        scheduler_state=None,
        epoch=epoch,
        global_step=0,
        best_fitness=best_val_loss,
        metrics={
            "train_loss": train_loss,
            "val_loss": val_loss,
            "effective_lr": effective_lr,
        },
        config=cfg,
        task="anomaly",
        media="video",
        checkpoint_kind=checkpoint_kind,
        component="predictor",
        extras={"effective_lr": effective_lr},
    )


def _predictor_state_dict_from_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    state_dict = checkpoint.get("model_state") or checkpoint.get("predictor_state") or checkpoint.get("extras", {}).get("predictor_state")
    if state_dict is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain predictor weights")
    return state_dict


def _resolve_effective_lr(train_cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    lr_mode = str(train_cfg.get("lr_mode", "manual"))
    batch_size = int(train_cfg["batch_size"])
    if lr_mode == "manual":
        effective_lr = float(train_cfg["lr"])
    elif lr_mode == "autoscale":
        reference_batch_size = int(train_cfg["reference_batch_size"])
        reference_lr = float(train_cfg["reference_lr"])
        ratio = batch_size / float(reference_batch_size)
        scale_rule = str(train_cfg.get("lr_scale_rule", "sqrt"))
        if scale_rule == "sqrt":
            effective_lr = reference_lr * math.sqrt(ratio)
        elif scale_rule == "linear":
            effective_lr = reference_lr * ratio
        else:
            raise ValueError(f"Unsupported lr_scale_rule: {scale_rule}")
    else:
        raise ValueError(f"Unsupported lr_mode: {lr_mode}")
    return float(effective_lr), {"lr_mode": lr_mode, "effective_lr": float(effective_lr)}


def _resolve_checkpoint_path(cfg: dict[str, Any], section: str) -> Path:
    explicit_path = cfg[section].get("checkpoint_path")
    output_root = _make_output_root(cfg)
    checkpoint_dir = checkpoint_paths(output_root).weights_dir
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        return path
    target = str(cfg[section].get("checkpoint_target", "best"))
    if target == "best":
        return checkpoint_dir / "best.pt"
    if target in {"latest", "last"}:
        return checkpoint_dir / "last.pt"
    raise ValueError(f"Unsupported checkpoint target: {target}")


def _thresholds_from_smoothed_summary(smoothed: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    calibration_labels, calibration_scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
    _, calibration_scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
    pred_reference = calibration_scores_pred[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_pred
    frozen_reference = calibration_scores_frozen[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_frozen
    pred_stats = _normal_stats(pred_reference)
    frozen_stats = _normal_stats(frozen_reference)
    multiplier = cfg["eval"]["threshold_std_multiplier"]
    predictor_threshold = float(pred_stats["mean"] + multiplier * pred_stats["std"])
    frozen_threshold = float(frozen_stats["mean"] + multiplier * frozen_stats["std"])
    return predictor_threshold, frozen_threshold, calibration_labels, calibration_scores_pred, calibration_scores_frozen


def train_from_runtime_config(config: dict[str, Any]) -> AnomalyTrainResult:
    cfg = _build_cfg(config, action="train")
    if int(cfg["train"]["num_workers"]) > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    _seed_everything(cfg["train"]["seed"])
    loaders = _make_loaders(cfg, include_test=False)
    output_root = _make_output_root(cfg)
    paths = checkpoint_paths(output_root)
    reports = output_root / "reports"
    paths.weights_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor), training=True)
    train_settings = build_train_settings(cfg["train"], epochs=int(cfg["train"]["epochs"]), batch_size=int(cfg["train"]["batch_size"]))
    backbone_ref = getattr(feature_extractor, "encoder", feature_extractor)
    optimization_host = type("AnomalyOptimizationHost", (), {"backbone": backbone_ref, "head": predictor})()
    stages = normalize_stages(task="anomaly", model=optimization_host, train_cfg=train_settings, default_epochs=int(cfg["train"]["epochs"]), batch_size=int(cfg["train"]["batch_size"]))
    stage0 = stages[0]
    effective_lr = resolve_autoscaled_lr(stage0.optimizer, batch_size=int(cfg["train"]["batch_size"]))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(predictor.parameters()),
                "lr": effective_lr,
                "initial_lr": effective_lr,
                "weight_decay": float(stage0.optimizer.get("weight_decay", cfg["train"]["weight_decay"])),
                "group_name": "predictor",
            }
        ],
        betas=tuple(float(value) for value in stage0.optimizer.get("betas", (0.9, 0.999))),
        eps=float(stage0.optimizer.get("eps", 1.0e-8)),
    )
    scheduler = build_scheduler(optimizer, stage0, steps_per_epoch=len(loaders["train_loader"]))
    train_rows = [
        {key: (int(value) if key == "epoch" else float(value) if key in {"train_loss", "val_loss", "lr", "best_fitness"} else value) for key, value in row.items()}
        for row in results_csv_rows(paths.results_csv)
    ]
    best_val = math.inf
    start_epoch = 1
    global_step = 0
    resume_path = resolve_resume_path(cfg["train"].get("resume", False), run_dir=output_root)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, resume_path))
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val = float(checkpoint.get("best_fitness", checkpoint.get("best_val_loss", math.inf)))
    best_path = paths.best
    last_path = paths.last
    epoch_timing_rows: list[dict[str, float]] = []
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        predictor.train()
        train_losses: list[float] = []
        train_decode_times: list[float] = []
        train_model_times: list[float] = []
        train_bar = _progress(loaders["train_loader"], desc=f"train {epoch}/{cfg['train']['epochs']}", total=len(loaders["train_loader"]))
        for batch in train_bar:
            train_decode_times.append(float(batch.get("decode_time", 0.0)))
            model_start = time.perf_counter()
            past_feat, future_feat = _extract_pair_features(feature_extractor, batch, runtime)
            with runtime.autocast_context():
                loss, _ = anomaly_future_prediction_loss(predictor, past_feat, future_feat, cfg["model"])
            optimizer.zero_grad(set_to_none=True)
            if runtime.scaler is not None:
                runtime.scaler.scale(loss).backward()
                runtime.scaler.step(optimizer)
                runtime.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step(global_step)
            train_model_times.append(float(time.perf_counter() - model_start))
            train_losses.append(float(loss.item()))
            global_step += 1
            if tqdm is not None:
                train_bar.set_postfix(loss=f"{train_losses[-1]:.5f}")
        predictor.eval()
        val_losses: list[float] = []
        val_decode_times: list[float] = []
        val_model_times: list[float] = []
        val_by_video: dict[str, dict[str, Any]] = {}
        with runtime.inference_context():
            val_bar = _progress(loaders["val_loader"], desc=f"val {epoch}/{cfg['train']['epochs']}", total=len(loaders["val_loader"]))
            for batch in val_bar:
                val_decode_times.append(float(batch.get("decode_time", 0.0)))
                model_start = time.perf_counter()
                past_feat, future_feat = _extract_pair_features(feature_extractor, batch, runtime)
                with runtime.autocast_context():
                    predictor_scores_t, frozen_scores_t = _predict_sample_scores(
                        predictor,
                        past_feat,
                        future_feat,
                        cfg["model"],
                        tubelet_size=feature_extractor.tubelet_size,
                    )
                    val_loss_value = float(anomaly_future_prediction_loss(predictor, past_feat, future_feat, cfg["model"])[0].item())
                val_model_times.append(float(time.perf_counter() - model_start))
                val_losses.append(val_loss_value)
                predictor_scores = predictor_scores_t.detach().cpu().numpy()
                frozen_scores = frozen_scores_t.detach().cpu().numpy()
                future_indices = batch["future_indices"].numpy()
                labels = batch.get("future_labels")
                labels_np = labels.numpy() if labels is not None else None
                for i, video_name in enumerate(batch["video_name"]):
                    state = val_by_video.setdefault(video_name, {"predictor_sum": {}, "predictor_count": {}, "frozen_sum": {}, "frozen_count": {}, "labels": {}, "has_labels": False})
                    for local_idx, frame_idx in enumerate(future_indices[i]):
                        idx = int(frame_idx)
                        state["predictor_sum"][idx] = state["predictor_sum"].get(idx, 0.0) + float(predictor_scores[i, local_idx])
                        state["predictor_count"][idx] = state["predictor_count"].get(idx, 0) + 1
                        state["frozen_sum"][idx] = state["frozen_sum"].get(idx, 0.0) + float(frozen_scores[i, local_idx])
                        state["frozen_count"][idx] = state["frozen_count"].get(idx, 0) + 1
                    if labels_np is not None:
                        state["has_labels"] = True
                        for frame_idx, label_value in zip(future_indices[i], labels_np[i]):
                            state["labels"][int(frame_idx)] = int(label_value)
                if tqdm is not None:
                    val_bar.set_postfix(loss=f"{val_loss_value:.5f}")
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_timings = _timing_metrics(decode_times=train_decode_times, model_times=train_model_times)
        val_timings = _timing_metrics(decode_times=val_decode_times, model_times=val_model_times)
        val_cfg = dict(cfg.get("eval", {}))
        val_summary = _finalize_video_summary(val_by_video)
        val_smoothed = _build_smoothed_summary(val_summary, int(val_cfg.get("smoothing_window", 1)))
        labels_raw, scores_pred_raw = _flatten_metric_arrays(val_summary, "predictor_scores")
        _, scores_frozen_raw = _flatten_metric_arrays(val_summary, "frozen_scores")
        labels_smooth, scores_pred_smooth = _flatten_metric_arrays(val_smoothed, "predictor_scores")
        _, scores_frozen_smooth = _flatten_metric_arrays(val_smoothed, "frozen_scores")
        predictor_threshold, frozen_threshold, calibration_labels, calibration_scores_pred, calibration_scores_frozen = _thresholds_from_smoothed_summary(val_smoothed, {"eval": val_cfg})
        calibration_normals_pred = calibration_scores_pred[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_pred
        calibration_normals_frozen = calibration_scores_frozen[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_frozen
        predictor_clip = _clip_level_metrics(val_smoothed, "predictor_scores", threshold=predictor_threshold, reduction="max")
        frozen_clip = _clip_level_metrics(val_smoothed, "frozen_scores", threshold=frozen_threshold, reduction="max")
        val_metrics = {
            "predictor_frame_auc_raw": _roc_auc_score(labels_raw, scores_pred_raw),
            "frozen_diff_frame_auc_raw": _roc_auc_score(labels_raw, scores_frozen_raw),
            "predictor_frame_auc": _roc_auc_score(labels_smooth, scores_pred_smooth),
            "frozen_diff_frame_auc": _roc_auc_score(labels_smooth, scores_frozen_smooth),
            "predictor_threshold": predictor_threshold,
            "frozen_threshold": frozen_threshold,
            "predictor_val_false_positive_rate": float(np.mean(calibration_normals_pred > predictor_threshold)) if len(calibration_normals_pred) else 0.0,
            "frozen_val_false_positive_rate": float(np.mean(calibration_normals_frozen > frozen_threshold)) if len(calibration_normals_frozen) else 0.0,
            "predictor_clip_auc": predictor_clip["auc"],
            "predictor_clip_accuracy": predictor_clip["accuracy"],
            "predictor_clip_precision": predictor_clip["precision"],
            "predictor_clip_recall": predictor_clip["recall"],
            "predictor_clip_specificity": predictor_clip["specificity"],
            "predictor_clip_f1": predictor_clip["f1"],
            "frozen_diff_clip_auc": frozen_clip["auc"],
            "frozen_diff_clip_accuracy": frozen_clip["accuracy"],
            "frozen_diff_clip_precision": frozen_clip["precision"],
            "frozen_diff_clip_recall": frozen_clip["recall"],
            "frozen_diff_clip_specificity": frozen_clip["specificity"],
            "frozen_diff_clip_f1": frozen_clip["f1"],
        }
        previous_best = best_val
        best_val = min(best_val, val_loss)
        train_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "best_fitness": best_val,
                "avg_train_decode_time": train_timings["avg_decode_time"],
                "avg_train_model_time": train_timings["avg_model_time"],
                "avg_val_decode_time": val_timings["avg_decode_time"],
                "avg_val_model_time": val_timings["avg_model_time"],
                **val_metrics,
            }
        )
        epoch_timing_rows.append({"epoch": epoch, **train_timings, **{f"val_{key}": value for key, value in val_timings.items()}})
        metric_summary = " ".join(
            [
                f"epoch={epoch}",
                f"train_loss={train_loss:.6f}",
                f"val_loss={val_loss:.6f}",
                f"predictor_frame_auc={val_metrics['predictor_frame_auc']:.4f}",
                f"predictor_clip_auc={val_metrics['predictor_clip_auc']:.4f}",
                f"predictor_clip_f1={val_metrics['predictor_clip_f1']:.4f}",
                f"frozen_diff_frame_auc={val_metrics['frozen_diff_frame_auc']:.4f}",
            ]
        )
        if tqdm is not None:
            tqdm.write(metric_summary)
        else:
            print(metric_summary)
        write_results_csv(paths.results_csv, train_rows)
        if cfg["train"]["save"]:
            latest_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val, effective_lr=effective_lr, checkpoint_kind="last")
            latest_payload["optimizer_state"] = optimizer.state_dict()
            latest_payload["scheduler_state"] = scheduler.state_dict()
            latest_payload["global_step"] = global_step
            latest_payload["best_fitness"] = best_val
            save_checkpoint(latest_payload, last_path)
            if val_loss <= previous_best:
                best_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val, effective_lr=effective_lr, checkpoint_kind="best")
                best_payload["optimizer_state"] = optimizer.state_dict()
                best_payload["scheduler_state"] = scheduler.state_dict()
                best_payload["global_step"] = global_step
                best_payload["best_fitness"] = best_val
                save_checkpoint(best_payload, best_path)
            if int(cfg["train"].get("save_period", 0)) > 0 and epoch % int(cfg["train"]["save_period"]) == 0:
                epoch_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val, effective_lr=effective_lr, checkpoint_kind="epoch")
                epoch_payload["optimizer_state"] = optimizer.state_dict()
                epoch_payload["scheduler_state"] = scheduler.state_dict()
                epoch_payload["global_step"] = global_step
                epoch_payload["best_fitness"] = best_val
                save_checkpoint(epoch_payload, paths.weights_dir / f"epoch_{epoch:03d}.pt")
    _write_csv(reports / "train_log.csv", train_rows)
    summary = {
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "effective_lr": effective_lr,
        "run_dir": str(output_root),
        "timings": epoch_timing_rows,
    }
    _write_json(reports / "train_summary.json", summary)
    return AnomalyTrainResult(best_val_loss=best_val, best_checkpoint=str(best_path), last_checkpoint=str(last_path), run_dir=str(output_root))


def _run_eval(config: dict[str, Any], *, split: str) -> tuple[dict[str, Any], Path]:
    cfg = _build_cfg(config, action="val")
    cfg["eval"]["split"] = split
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    loaders = _make_loaders(cfg, include_test=True)
    output_root = _make_output_root(cfg)
    reports = output_root / "reports"
    plots = output_root / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor).eval(), training=False)
    checkpoint_path = _resolve_checkpoint_path(cfg, "eval")
    checkpoint = load_checkpoint(checkpoint_path)
    predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, checkpoint_path))
    predictor.eval()
    summary, timings = _aggregate_scores(
        loaders["val_loader"] if split == "val" else loaders["test_loader"],
        predictor,
        feature_extractor,
        runtime,
        desc=f"score {split}",
        model_cfg=cfg["model"],
    )
    smoothed = _build_smoothed_summary(summary, int(cfg["eval"].get("smoothing_window", 1)))
    calibration_smoothed = smoothed
    if split != "val":
        calibration_summary, _ = _aggregate_scores(
            loaders["val_loader"],
            predictor,
            feature_extractor,
            runtime,
            desc="score val (threshold calibration)",
            model_cfg=cfg["model"],
        )
        calibration_smoothed = _build_smoothed_summary(calibration_summary, int(cfg["eval"].get("smoothing_window", 1)))
    labels_raw, scores_pred_raw = _flatten_metric_arrays(summary, "predictor_scores")
    _, scores_frozen_raw = _flatten_metric_arrays(summary, "frozen_scores")
    labels, scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
    _, scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
    predictor_threshold, frozen_threshold, calibration_labels, calibration_scores_pred, calibration_scores_frozen = _thresholds_from_smoothed_summary(calibration_smoothed, cfg)
    calibration_normals_pred = calibration_scores_pred[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_pred
    calibration_normals_frozen = calibration_scores_frozen[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_frozen
    predictor_clip = _clip_level_metrics(smoothed, "predictor_scores", threshold=predictor_threshold, reduction="max")
    frozen_clip = _clip_level_metrics(smoothed, "frozen_scores", threshold=frozen_threshold, reduction="max")
    metrics = {
        "split": split,
        "predictor_type": cfg["model"]["predictor_type"],
        "predictor_frame_auc_raw": _roc_auc_score(labels_raw, scores_pred_raw),
        "frozen_diff_frame_auc_raw": _roc_auc_score(labels_raw, scores_frozen_raw),
        "predictor_frame_auc": _roc_auc_score(labels, scores_pred),
        "frozen_diff_frame_auc": _roc_auc_score(labels, scores_frozen),
        "predictor_threshold": predictor_threshold,
        "frozen_threshold": frozen_threshold,
        "predictor_val_false_positive_rate": float(np.mean(calibration_normals_pred > predictor_threshold)) if len(calibration_normals_pred) else 0.0,
        "frozen_val_false_positive_rate": float(np.mean(calibration_normals_frozen > frozen_threshold)) if len(calibration_normals_frozen) else 0.0,
        "clip_score_reduction": "max",
        "predictor_clip_auc": predictor_clip["auc"],
        "frozen_diff_clip_auc": frozen_clip["auc"],
        "predictor_clip_accuracy": predictor_clip["accuracy"],
        "predictor_clip_precision": predictor_clip["precision"],
        "predictor_clip_recall": predictor_clip["recall"],
        "predictor_clip_specificity": predictor_clip["specificity"],
        "predictor_clip_f1": predictor_clip["f1"],
        "predictor_clip_confusion_matrix": predictor_clip["confusion_matrix"],
        "predictor_clip_counts": predictor_clip["counts"],
        "frozen_diff_clip_accuracy": frozen_clip["accuracy"],
        "frozen_diff_clip_precision": frozen_clip["precision"],
        "frozen_diff_clip_recall": frozen_clip["recall"],
        "frozen_diff_clip_specificity": frozen_clip["specificity"],
        "frozen_diff_clip_f1": frozen_clip["f1"],
        "frozen_diff_clip_confusion_matrix": frozen_clip["confusion_matrix"],
        "frozen_diff_clip_counts": frozen_clip["counts"],
        "smoothing_window": int(cfg["eval"].get("smoothing_window", 1)),
        "checkpoint_path": str(checkpoint_path),
        **timings,
    }
    report_path = reports / f"{split}_metrics.json"
    _write_json(report_path, metrics)
    _write_json(reports / f"{split}_scores.json", summary)
    _write_json(reports / f"{split}_scores_smoothed.json", smoothed)
    _write_json(
        reports / f"{split}_clip_scores.json",
        {
            "split": split,
            "predictor_threshold": predictor_threshold,
            "frozen_threshold": frozen_threshold,
            "clip_score_reduction": "max",
            "predictor_clips": predictor_clip["clips"],
            "frozen_clips": frozen_clip["clips"],
        },
    )
    return metrics, report_path


def _run_predict(config: dict[str, Any]) -> tuple[dict[str, Any], Path, list[str]]:
    cfg = _build_cfg(config, action="predict")
    source = cfg["predict"].get("source")
    split = None if source else str(cfg["predict"].get("split", "test"))
    if split not in {None, "val", "test"}:
        split = "test"
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_root = _make_output_root(cfg)
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    predict_out = _predict_output_root(cfg, split=split, source=source)
    print(f"Output will be saved to: {predict_out}")
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor).eval(), training=False)
    checkpoint_path = _resolve_checkpoint_path(cfg, "eval")
    checkpoint = load_checkpoint(checkpoint_path)
    predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, checkpoint_path))
    predictor.eval()
    if source:
        videos = [_build_source_record(source, video_backend=str(cfg["dataset"].get("video_backend", "auto")))]
        loader = _build_eval_loader(
            videos,
            cfg,
            batch_size=int(cfg["predict"]["batch_size"]),
            num_workers=int(cfg["predict"]["num_workers"]),
        )
        desc = f"predict:{Path(source).name}"
    else:
        loaders = _make_loaders(cfg, include_test=True)
        videos = loaders["val_videos"] if split == "val" else loaders["test_videos"]
        loader = loaders["val_loader"] if split == "val" else loaders["test_loader"]
        desc = f"predict:{split}"
    summary, timings = _aggregate_scores(loader, predictor, feature_extractor, runtime, desc=desc, model_cfg=cfg["model"])
    smoothed = _build_smoothed_summary(summary, int(cfg["eval"].get("smoothing_window", 1)))
    report_stem = "source" if source else str(split)
    _write_json(reports / f"{report_stem}_predict_scores.json", summary)
    _write_json(reports / f"{report_stem}_predict_scores_smoothed.json", smoothed)
    threshold = cfg["predict"].get("threshold")
    metrics: dict[str, Any] = {
        "split": split,
        "source": source,
        "predictor_type": cfg["model"]["predictor_type"],
        "smoothing_window": int(cfg["eval"].get("smoothing_window", 1)),
        "checkpoint_path": str(checkpoint_path),
        "threshold": None if threshold is None else float(threshold),
        **timings,
    }
    if not source:
        labels_raw, scores_pred_raw = _flatten_metric_arrays(summary, "predictor_scores")
        _, scores_frozen_raw = _flatten_metric_arrays(summary, "frozen_scores")
        labels, scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
        _, scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
        metrics.update(
            {
                "predictor_frame_auc_raw": _roc_auc_score(labels_raw, scores_pred_raw),
                "frozen_diff_frame_auc_raw": _roc_auc_score(labels_raw, scores_frozen_raw),
                "predictor_frame_auc": _roc_auc_score(labels, scores_pred),
                "frozen_diff_frame_auc": _roc_auc_score(labels, scores_frozen),
            }
        )
    if threshold is not None:
        threshold_value = float(threshold)
        predictor_clip = _threshold_clip_predictions(smoothed, "predictor_scores", threshold=threshold_value, reduction="max")
        metrics.update(
            {
                "clip_score_reduction": "max",
                "predictor_thresholded": predictor_clip,
            }
        )
    report_payload = {"summary": smoothed, "metrics": metrics}
    report_path = reports / f"{report_stem}_predict_summary.json"
    _write_json(report_path, report_payload)
    rendered_outputs: list[str] = []
    if bool(cfg["predict"].get("visualize", False)):
        if threshold is None:
            raise ValueError("predict.visualize=true requires predict.threshold=<float>")
        output_dir = cfg["predict"].get("output_dir")
        render_root = (_repo_root() / output_dir).resolve() if output_dir else _predict_output_root(cfg, split=split, source=source)
        for video in videos:
            payload = smoothed["videos"].get(video.name)
            if payload is None:
                continue
            render_path = render_root / f"{video.name}.mp4"
            _render_prediction_video(
                source_path=video.media_path,
                output_path=render_path,
                predictor_scores=payload["predictor_scores"],
                threshold=float(threshold),
                labels=payload.get("labels"),
            )
            rendered_outputs.append(str(render_path))
        metrics["rendered_outputs"] = rendered_outputs
        _write_json(reports / f"{report_stem}_predict_visualization.json", metrics)
    return metrics, report_path, rendered_outputs


def validate_from_runtime_config(config: dict[str, Any]) -> AnomalyValidationResult:
    split = str(config["val"].get("split", "val"))
    if split not in {"val", "test"}:
        split = "val"
    metrics, report_path = _run_eval(config, split=split)
    return AnomalyValidationResult(split=split, metrics=metrics, report_path=str(report_path))


def predict_from_runtime_config(config: dict[str, Any]) -> AnomalyPredictResult:
    split = config["predict"].get("split")
    metrics, report_path, rendered_outputs = _run_predict(config)
    if split not in {"val", "test"}:
        split = None
    return AnomalyPredictResult(split=split, metrics=metrics, report_path=str(report_path), rendered_outputs=rendered_outputs or None)


def export_from_runtime_config(config: dict[str, Any]) -> AnomalyExportResult:
    cfg = _build_cfg(config, action="export")
    if str(cfg["export"]["format"]).lower() != "onnx":
        raise ValueError("Active forge anomaly export currently supports format=onnx only")
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor).eval(), training=False)
    checkpoint_path = _resolve_checkpoint_path(cfg, "export")
    checkpoint = load_checkpoint(checkpoint_path)
    predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, checkpoint_path))
    predictor.eval()
    wrapper = runtime.prepare_module(_InferenceWrapper(feature_extractor, predictor, cfg["model"]).eval(), training=False)
    output_path = Path(cfg["export"]["output_path"])
    if not output_path.is_absolute():
        output_path = (_repo_root() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_past = torch.randn(1, 3, cfg["dataset"]["past_frames"], cfg["dataset"]["image_size"], cfg["dataset"]["image_size"], device=device)
    sample_future = torch.randn(1, 3, cfg["dataset"]["future_frames"], cfg["dataset"]["image_size"], cfg["dataset"]["image_size"], device=device)
    torch.onnx.export(
        wrapper,
        (sample_past, sample_future),
        str(output_path),
        input_names=["past", "future"],
        output_names=["predictor_scores", "frozen_scores"],
        dynamic_axes={
            "past": {0: "batch"},
            "future": {0: "batch"},
            "predictor_scores": {0: "batch"},
            "frozen_scores": {0: "batch"},
        } if cfg["export"]["dynamic_axes"] else None,
        opset_version=int(cfg["export"]["opset"]),
    )
    return AnomalyExportResult(output_path=str(output_path), checkpoint_path=str(checkpoint_path))
