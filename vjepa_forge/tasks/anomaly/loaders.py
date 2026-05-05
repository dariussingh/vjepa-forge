from __future__ import annotations

import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vjepa_forge.data.cache import CachedFeatureItem, FeatureCacheStore, cached_feature_item_key, default_feature_cache_root, manifest_cache_dir
from vjepa_forge.data.video import read_video_clip
from vjepa_forge.tasks.anomaly.data import (
    ForgeAnomalyWindowDataset,
    VideoClipRecord,
    WindowRecord,
    _WindowBatchSampler,
    _build_video_records,
    _build_window_records,
    _repo_root,
    _subsample_videos,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _feature_cache_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = str(cfg["dataset"].get("feature_cache", "false")).lower()
    dataset_yaml = cfg["dataset"].get("dataset_yaml")
    source = cfg.get("predict", {}).get("source")
    default_root_base = Path(source).expanduser().resolve().parent if source else (Path(dataset_yaml).expanduser().resolve().parent if dataset_yaml else _repo_root())
    return {
        "enabled": enabled,
        "root": (
            default_feature_cache_root(default_root_base)
            if not cfg["dataset"].get("feature_cache_root")
            else Path(str(cfg["dataset"]["feature_cache_root"])).expanduser().resolve()
        ),
        "build_on_miss": bool(cfg["dataset"].get("feature_cache_build_on_miss", True)),
        "readonly": bool(cfg["dataset"].get("feature_cache_readonly", False)),
        "shard_size": max(1, int(cfg["dataset"].get("feature_cache_shard_size", 64))),
        "dtype": str(cfg["dataset"].get("feature_cache_dtype", "fp16")),
    }


def _resolve_cache_dtype(dtype_str: str) -> torch.dtype | None:
    lowered = str(dtype_str).lower()
    if lowered == "fp16":
        return torch.float16
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None  # fp32 / keep as-is


def _anomaly_cache_spec(cfg: dict[str, Any], *, split: str, source: str | None = None) -> dict[str, Any]:
    return {
        "task": "anomaly",
        "media": "video",
        "split": split,
        "source": None if source is None else str(Path(source).expanduser().resolve()),
        "dataset_yaml": cfg["dataset"].get("dataset_yaml"),
        "image_size": int(cfg["dataset"]["image_size"]),
        "past_frames": int(cfg["dataset"]["past_frames"]),
        "future_frames": int(cfg["dataset"]["future_frames"]),
        "stride": int(cfg["dataset"]["stride"]),
        "video_backend": str(cfg["dataset"].get("video_backend", "auto")),
        "model_name": str(cfg["model"]["name"]),
        "checkpoint": str(cfg["model"]["checkpoint"]),
        "checkpoint_key": str(cfg["model"]["checkpoint_key"]),
        "predictor_type": str(cfg["model"]["predictor_type"]),
        # fraction and seed are part of the spec so different fractions get different cache dirs
        "train_fraction": float(cfg["dataset"].get("train_fraction", 1.0)) if split == "train" else 1.0,
        "train_seed": int(cfg["train"].get("seed", 0)),
    }


def _build_anomaly_feature_cache(
    *,
    store: FeatureCacheStore,
    spec: dict[str, Any],
    windows: list[WindowRecord],
    videos: list[VideoClipRecord],
    cfg: dict[str, Any],
    feature_extractor: nn.Module,
    device: torch.device,
    shard_size: int,
    batch_size: int | None = None,
    num_workers: int | None = None,
    amp_dtype: torch.dtype | None = None,
    cache_dtype: torch.dtype | None = torch.float16,
) -> None:
    """Build the anomaly feature cache using batched DataLoader for GPU efficiency.

    Reuses _WindowBatchSampler + _collate_window_batch so windows from the same
    video are decoded once (O(B+T) instead of O(B*T)) and the configured video
    backend (DALI or decord) is respected. Writes shards to disk incrementally
    via _StreamingCacheWriter to avoid accumulating the full dataset in RAM.
    """
    dataset_cfg = cfg["dataset"]
    image_size = int(dataset_cfg["image_size"])
    video_backend = str(dataset_cfg.get("video_backend", "auto"))
    reader_cache_size = int(cfg["eval"]["reader_cache_size"])

    resolved_batch_size = batch_size if batch_size is not None else int(cfg["train"]["batch_size"])

    # DALI requires num_workers=0 — mirror the trainer's existing policy
    def _has_dali_local() -> bool:
        try:
            import nvidia.dali  # noqa: F401
            return True
        except Exception:
            return False

    dali_active = video_backend == "dali" or (video_backend == "auto" and _has_dali_local())
    if dali_active:
        resolved_workers = 0
    elif num_workers is not None:
        resolved_workers = int(num_workers)
    else:
        resolved_workers = int(cfg["train"].get("num_workers", 0))

    if tqdm is not None:
        tqdm.write(
            f"building anomaly feature cache at {store.cache_dir} "
            f"(batch_size={resolved_batch_size}, workers={resolved_workers}, backend={'dali' if dali_active else video_backend})"
        )

    ds = ForgeAnomalyWindowDataset(videos, windows, image_size, video_backend=video_backend)
    ds.reader_cache_size = reader_cache_size
    collate = partial(
        _collate_window_batch,
        image_size=image_size,
        reader_cache_size=reader_cache_size,
        video_backend=video_backend,
    )
    batch_sampler = _WindowBatchSampler(windows, batch_size=resolved_batch_size, shuffle=False)
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "collate_fn": collate,
        "num_workers": resolved_workers,
        "pin_memory": (device.type == "cuda" and not dali_active),
    }
    if resolved_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kwargs)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (amp_dtype is not None and device.type == "cuda")
        else nullcontext()
    )

    iterator = loader if tqdm is None else tqdm(
        loader,
        desc=f"cache:anomaly:{spec['split']}",
        total=len(batch_sampler),
        dynamic_ncols=True,
    )

    height_patches = int(feature_extractor.grid_size)
    width_patches = int(feature_extractor.grid_size)
    temporal_tokens = int(feature_extractor.grid_depth)

    feature_extractor.eval()
    with store.open_streaming_write(spec=spec, shard_size=shard_size) as writer:
        with torch.no_grad():
            with autocast_ctx:
                for batch in iterator:
                    past_b = batch["past"].to(device, non_blocking=True)
                    future_b = batch["future"].to(device, non_blocking=True)
                    past_feat = feature_extractor(past_b)
                    future_feat = feature_extractor(future_b)
                    for b in range(past_b.shape[0]):
                        key = cached_feature_item_key(
                            media_path=batch["media_path"][b],
                            clip_start=int(batch["clip_start"][b]),
                            clip_len=int(batch["clip_len"][b]),
                            stride=1,
                        )

                        def _to_cache(t: torch.Tensor) -> torch.Tensor:
                            out = t.detach()
                            if cache_dtype is not None:
                                out = out.to(dtype=cache_dtype)
                            return out.cpu()

                        item = CachedFeatureItem(
                            mode="final",
                            media="video",
                            split_layer=-1,
                            token_state=None,
                            cached_outputs=[
                                _to_cache(past_feat.pooled[b]),
                                _to_cache(past_feat.tokens[b]),
                                _to_cache(future_feat.pooled[b]),
                                _to_cache(future_feat.tokens[b]),
                            ],
                            height_patches=height_patches,
                            width_patches=width_patches,
                            temporal_tokens=temporal_tokens,
                        )
                        writer.append(key, item)


def _resolve_anomaly_feature_cache_store(
    *,
    cfg: dict[str, Any],
    split: str,
    windows: list[WindowRecord],
    videos: list[VideoClipRecord],
    feature_extractor: nn.Module | None,
    device: torch.device | None,
    source: str | None = None,
    runtime=None,
) -> FeatureCacheStore | None:
    settings = _feature_cache_settings(cfg)
    if settings["enabled"] == "false":
        return None
    spec = _anomaly_cache_spec(cfg, split=split, source=source)
    store = FeatureCacheStore(manifest_cache_dir(settings["root"], spec))
    if not store.exists():
        if settings["readonly"] or not settings["build_on_miss"]:
            if settings["enabled"] == "true":
                raise FileNotFoundError(f"Anomaly feature cache missing: {store.cache_dir}")
            return None
        if feature_extractor is None or device is None:
            if settings["enabled"] == "true":
                raise RuntimeError("Cannot build anomaly feature cache without a feature extractor")
            return None
        _build_anomaly_feature_cache(
            store=store,
            spec=spec,
            windows=windows,
            videos=videos,
            cfg=cfg,
            feature_extractor=feature_extractor,
            device=device,
            shard_size=int(settings["shard_size"]),
            amp_dtype=runtime.amp_dtype if runtime is not None else None,
            cache_dtype=_resolve_cache_dtype(settings["dtype"]),
        )
    elif store.load_manifest().get("spec") != spec and settings["enabled"] == "true":
        raise ValueError(f"Anomaly feature cache spec mismatch: {store.cache_dir}")
    return store


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
        # extra keys used by the feature cache builder to construct cache item keys
        "media_path": [sample["media_path"] for sample in batch],
        "clip_start": torch.tensor([int(sample["clip_start"]) for sample in batch], dtype=torch.long),
        "clip_len": torch.tensor([int(sample["clip_len"]) for sample in batch], dtype=torch.long),
    }
    if "future_labels" in batch[0]:
        collated["future_labels"] = torch.stack([sample["future_labels"] for sample in batch], dim=0)
    return collated


def _collate_cached_window_batch(batch: list[dict[str, Any]], *, feature_cache: FeatureCacheStore) -> dict[str, Any]:
    if not batch:
        return {}
    past_pooled: list[torch.Tensor] = []
    past_tokens: list[torch.Tensor] = []
    future_pooled: list[torch.Tensor] = []
    future_tokens: list[torch.Tensor] = []
    for sample in batch:
        item = feature_cache.get(
            cached_feature_item_key(
                media_path=sample["media_path"],
                clip_start=int(sample["clip_start"]),
                clip_len=int(sample["clip_len"]),
                stride=1,
            )
        )
        past_pooled.append(item.cached_outputs[0])
        past_tokens.append(item.cached_outputs[1])
        future_pooled.append(item.cached_outputs[2])
        future_tokens.append(item.cached_outputs[3])
    collated: dict[str, Any] = {
        "past_pooled": torch.stack(past_pooled, dim=0),
        "past_tokens": torch.stack(past_tokens, dim=0),
        "future_pooled": torch.stack(future_pooled, dim=0),
        "future_tokens": torch.stack(future_tokens, dim=0),
        "video_name": [sample["video_name"] for sample in batch],
        "future_indices": torch.stack([sample["future_indices"] for sample in batch], dim=0),
        "decode_time": 0.0,
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
    train_num_workers = int(cfg["train"]["num_workers"])
    eval_num_workers = int(cfg["eval"]["num_workers"])
    train_cache = _resolve_anomaly_feature_cache_store(
        cfg=cfg,
        split="train",
        windows=train_windows,
        videos=train_videos,
        feature_extractor=feature_extractor,
        device=device,
        runtime=runtime,
    )
    val_cache = _resolve_anomaly_feature_cache_store(
        cfg=cfg,
        split=val_split,
        windows=val_windows,
        videos=val_videos,
        feature_extractor=feature_extractor,
        device=device,
        runtime=runtime,
    )
    test_cache = None
    if include_test:
        test_cache = _resolve_anomaly_feature_cache_store(
            cfg=cfg,
            split="test",
            windows=test_windows,
            videos=test_videos,
            feature_extractor=feature_extractor,
            device=device,
            runtime=runtime,
        )
    train_collate = partial(_collate_cached_window_batch, feature_cache=train_cache) if train_cache is not None else partial(
        _collate_window_batch,
        image_size=image_size,
        reader_cache_size=int(cfg["train"]["reader_cache_size"]),
        video_backend=video_backend,
    )
    eval_collate = partial(_collate_cached_window_batch, feature_cache=val_cache) if val_cache is not None else partial(
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
            collate_fn=partial(_collate_cached_window_batch, feature_cache=test_cache) if test_cache is not None else eval_collate,
            batch_sampler=_WindowBatchSampler(test_windows, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False),
        )
        loaders["test_loader"] = DataLoader(test_ds, **test_loader_kwargs)
    return loaders


def _make_loaders_compat(cfg: dict[str, Any], include_test: bool = True, *, feature_extractor: nn.Module | None = None, device: torch.device | None = None, runtime=None) -> dict[str, Any]:
    try:
        return _make_loaders(cfg, include_test=include_test, feature_extractor=feature_extractor, device=device, runtime=runtime)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return _make_loaders(cfg, include_test=include_test)


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
    worker_count = int(cfg["eval"]["num_workers"] if num_workers is None else num_workers)
    cache_store = _resolve_anomaly_feature_cache_store(
        cfg=cfg,
        split=str(cfg["eval"].get("split", "test")),
        windows=windows,
        videos=videos,
        feature_extractor=feature_extractor,
        device=device,
        source=source,
    )
    loader_kwargs = _loader_kwargs(
        batch_size=int(cfg["eval"]["batch_size"] if batch_size is None else batch_size),
        num_workers=worker_count,
        pin_memory=bool(cfg["eval"]["pin_memory"] and video_backend != "dali"),
        persistent_workers=bool(cfg["eval"]["persistent_workers"]),
        prefetch_factor=int(cfg["eval"]["prefetch_factor"]),
        collate_fn=partial(_collate_cached_window_batch, feature_cache=cache_store) if cache_store is not None else partial(
            _collate_window_batch,
            image_size=dataset_cfg["image_size"],
            reader_cache_size=int(cfg["eval"]["reader_cache_size"]),
            video_backend=video_backend,
        ),
        batch_sampler=_WindowBatchSampler(windows, batch_size=int(cfg["eval"]["batch_size"] if batch_size is None else batch_size), shuffle=False),
    )
    return DataLoader(ds, **loader_kwargs)
