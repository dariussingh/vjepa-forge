from __future__ import annotations

import time
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vjepa_forge.data.video import read_video_clip
from vjepa_forge.tasks.anomaly.data import (
    ForgeAnomalyWindowDataset,
    VideoClipRecord,
    WindowRecord,
    _WindowBatchSampler,
    _build_video_records,
    _build_window_records,
    _subsample_videos,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


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


def _dali_active(video_backend: str) -> bool:
    if video_backend == "dali":
        return True
    if video_backend == "auto":
        try:
            import nvidia.dali  # noqa: F401
            return True
        except Exception:
            return False
    return False


def _make_loaders(cfg: dict[str, Any], include_test: bool = True, *, feature_extractor: nn.Module | None = None, device: torch.device | None = None, runtime=None) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    train_videos = _build_video_records(dataset_cfg["dataset_yaml"], split="train")
    train_fraction = float(dataset_cfg.get("train_fraction", 1.0))
    train_seed = int(cfg["train"].get("seed", 0))
    train_videos = _subsample_videos(train_videos, train_fraction, train_seed)
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
    # DALI initializes CUDA in the main process; forked workers cannot re-initialize it.
    train_num_workers = 0 if _dali_active(video_backend) else int(cfg["train"]["num_workers"])
    eval_num_workers = 0 if _dali_active(video_backend) else int(cfg["eval"]["num_workers"])
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
    feature_extractor: nn.Module | None = None,
    device: torch.device | None = None,
    source: str | None = None,
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
    worker_count = 0 if _dali_active(video_backend) else int(cfg["eval"]["num_workers"] if num_workers is None else num_workers)
    resolved_batch_size = int(cfg["eval"]["batch_size"] if batch_size is None else batch_size)
    loader_kwargs = _loader_kwargs(
        batch_size=resolved_batch_size,
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
        batch_sampler=_WindowBatchSampler(windows, batch_size=resolved_batch_size, shuffle=False),
    )
    return DataLoader(ds, **loader_kwargs)
