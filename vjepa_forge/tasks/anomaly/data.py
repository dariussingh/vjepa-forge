from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vjepa_forge.data.forge.dataset import ForgeDataset
from vjepa_forge.data.video import get_video_frame_count
from vjepa_forge.engine.checkpointing import resolve_run_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _subsample_videos(videos: list[VideoClipRecord], fraction: float, seed: int) -> list[VideoClipRecord]:
    """Deterministically sample `fraction` of videos using the given seed."""
    if fraction >= 1.0 or not videos:
        return videos
    k = max(1, int(round(len(videos) * fraction)))
    return random.Random(seed).sample(videos, k)


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


def _predict_output_root(cfg: dict[str, Any], *, split: str | None, source: str | None) -> Path:
    output_root = _make_output_root(cfg)
    base = output_root / "predict"
    if source:
        return base / Path(source).stem
    return base / str(split or "custom")
