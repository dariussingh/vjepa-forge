from __future__ import annotations

import functools
import os
import random as _random_module
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from vjepa_forge.data.cache import FeatureCacheStore, cached_feature_item_key, default_feature_cache_root, manifest_cache_dir, serialize_spec
from vjepa_forge.data.forge.dataset import ForgeDataset
from vjepa_forge.data.image import read_image
from vjepa_forge.data.video import read_video_clip

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass(frozen=True)
class FeatureCacheSettings:
    enabled: str
    root: Path
    build_on_miss: bool
    readonly: bool
    shard_size: int
    validate: bool


def resolve_feature_cache_settings(*, data_cfg: dict[str, Any], dataset_root: str | Path) -> FeatureCacheSettings:
    enabled = str(data_cfg.get("feature_cache", "false")).lower()
    root_ref = data_cfg.get("feature_cache_root")
    root = default_feature_cache_root(dataset_root) if root_ref in {None, ""} else Path(str(root_ref)).expanduser().resolve()
    return FeatureCacheSettings(
        enabled=enabled,
        root=root,
        build_on_miss=bool(data_cfg.get("feature_cache_build_on_miss", True)),
        readonly=bool(data_cfg.get("feature_cache_readonly", False)),
        shard_size=max(1, int(data_cfg.get("feature_cache_shard_size", 64))),
        validate=bool(data_cfg.get("feature_cache_validate", True)),
    )


def resolve_cache_split_layer(*, model, freeze_cfg: dict[str, Any] | None, data_cfg: dict[str, Any] | None = None) -> int | None:
    freeze_cfg = {} if freeze_cfg is None else dict(freeze_cfg)
    data_cfg = {} if data_cfg is None else dict(data_cfg)
    if not freeze_cfg:
        train_cfg = data_cfg.get("train")
        if isinstance(train_cfg, dict):
            stages = list(train_cfg.get("stages", []))
            if stages:
                freeze_cfg = dict(stages[0].get("freeze", {}))
    total_layers = int(model.backbone.get_num_layers())
    backbone_blocks = freeze_cfg.get("backbone_blocks")
    freeze_backbone = bool(freeze_cfg.get("backbone", False))
    if backbone_blocks is not None:
        split_layer = max(0, total_layers - int(backbone_blocks))
        return None if split_layer <= 0 else split_layer
    if freeze_backbone or bool(model.model_cfg.get("backbone", {}).get("freeze", False)):
        return total_layers
    return None


def build_generic_cache_spec(
    *,
    dataset: ForgeDataset,
    model,
    split: str,
    image_size: int,
    clip_len: int,
    clip_stride: int,
    image_backend: str,
    video_backend: str,
    split_layer: int,
) -> dict[str, Any]:
    return serialize_spec(
        {
            "dataset_yaml": str(dataset.yaml_path),
            "dataset_root": str(dataset.root),
            "split": split,
            "task": dataset.task,
            "media": dataset.media,
            "image_size": int(image_size),
            "clip_len": int(clip_len),
            "clip_stride": int(clip_stride),
            "image_backend": str(image_backend),
            "video_backend": str(video_backend),
            "model_name": str(model.model_cfg.get("name", "")),
            "backbone_name": str(model.model_cfg.get("backbone", {}).get("name", "")),
            "backbone_checkpoint": str(model.model_cfg.get("backbone", {}).get("checkpoint")),
            "checkpoint_key": str(model.model_cfg.get("backbone", {}).get("checkpoint_key", "ema_encoder")),
            "output_layers": list(getattr(model.backbone.image_backbone, "out_layers", [])),
            "split_layer": int(split_layer),
            "train_fraction": float(data_cfg.get("train_fraction", 1.0)) if split == "train" else 1.0,
        }
    )


def resolve_generic_cache_store(
    *,
    dataset: ForgeDataset,
    model,
    split: str,
    data_cfg: dict[str, Any],
    freeze_cfg: dict[str, Any] | None,
    runtime=None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> FeatureCacheStore | None:
    settings = resolve_feature_cache_settings(data_cfg=data_cfg, dataset_root=dataset.root)
    if settings.enabled == "false":
        return None
    clip_len = int(data_cfg.get("clip_len", data_cfg.get("num_frames", 8)))
    clip_stride = int(data_cfg.get("clip_stride", 1))
    image_size = int(data_cfg.get("image_size", 384))
    image_backend = str(data_cfg.get("image_backend", "auto"))
    video_backend = str(data_cfg.get("video_backend", "auto"))
    split_layer = resolve_cache_split_layer(model=model, freeze_cfg=freeze_cfg, data_cfg=data_cfg)
    if split_layer is None:
        return None
    spec = build_generic_cache_spec(
        dataset=dataset,
        model=model,
        split=split,
        image_size=image_size,
        clip_len=clip_len,
        clip_stride=clip_stride,
        image_backend=image_backend,
        video_backend=video_backend,
        split_layer=split_layer,
    )
    store = FeatureCacheStore(manifest_cache_dir(settings.root, spec))
    if not store.exists():
        if settings.readonly or not settings.build_on_miss:
            if settings.enabled == "true":
                raise FileNotFoundError(f"Feature cache missing for split={split}: {store.cache_dir}")
            return None
        resolved_batch_size = int(data_cfg.get("feature_cache_batch_size") or batch_size or data_cfg.get("batch_size", 32))
        resolved_workers = num_workers
        amp_dtype = runtime.amp_dtype if runtime is not None else None
        cache_dtype = _resolve_cache_dtype(str(data_cfg.get("feature_cache_dtype", "fp16")))
        train_fraction = float(data_cfg.get("train_fraction", 1.0)) if split == "train" else 1.0
        train_seed = int(data_cfg.get("train", {}).get("seed", 0)) if isinstance(data_cfg.get("train"), dict) else 0
        build_generic_feature_cache(
            store=store,
            spec=spec,
            dataset=dataset,
            model=model,
            split_layer=split_layer,
            image_size=image_size,
            clip_len=clip_len,
            clip_stride=clip_stride,
            image_backend=image_backend,
            video_backend=video_backend,
            shard_size=settings.shard_size,
            batch_size=resolved_batch_size,
            num_workers=resolved_workers,
            amp_dtype=amp_dtype,
            cache_dtype=cache_dtype,
            train_fraction=train_fraction,
            train_seed=train_seed,
        )
    elif settings.validate and not store.spec_matches(spec):
        if settings.enabled == "true":
            raise ValueError(f"Feature cache spec mismatch for split={split}: {store.cache_dir}")
        return None
    return store


def _resolve_cache_dtype(dtype_str: str) -> torch.dtype | None:
    lowered = str(dtype_str).lower()
    if lowered == "fp16":
        return torch.float16
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None  # fp32 — keep as-is


# ---------------------------------------------------------------------------
# DataLoader-based batched cache build helpers
# ---------------------------------------------------------------------------

class _ForgeRecordDataset(TorchDataset):
    """Thin torch Dataset adapter exposing ForgeDataset records for DataLoader workers."""

    def __init__(self, forge_dataset: ForgeDataset) -> None:
        self._records = forge_dataset.records
        self._media = forge_dataset.media

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"index": index, "media_path": str(self._records[index].media_path)}


def _cache_build_collate_fn(
    batch: list[dict[str, Any]],
    *,
    records,
    media: str,
    image_size: int,
    clip_len: int,
    clip_stride: int,
    image_backend: str,
    video_backend: str,
    reader_cache_size: int,
) -> dict[str, Any]:
    """Decode a batch of media items in worker processes; returns stacked tensors and keys."""
    tensors: list[torch.Tensor] = []
    keys: list[str] = []
    for item in batch:
        record = records[item["index"]]
        if media == "image":
            tensor = read_image(record.media_path, image_size=image_size, image_backend=image_backend, reader_cache_size=reader_cache_size)
            key = cached_feature_item_key(media_path=record.media_path)
        else:
            tensor = read_video_clip(
                record.media_path,
                clip_len=clip_len,
                stride=clip_stride,
                image_size=image_size,
                reader_cache_size=reader_cache_size,
                video_backend=video_backend,
            )
            key = cached_feature_item_key(media_path=record.media_path, clip_len=clip_len, stride=clip_stride)
        tensors.append(tensor)
        keys.append(key)
    return {"tensors": torch.stack(tensors, dim=0), "keys": keys}


def _dali_active(*, video_backend: str, image_backend: str, media: str) -> bool:
    """Return True if the configured backend resolves to DALI for the given media type."""
    def _has_dali() -> bool:
        try:
            import nvidia.dali  # noqa: F401
            return True
        except Exception:
            return False

    if media == "video":
        return video_backend == "dali" or (video_backend == "auto" and _has_dali())
    if media == "image":
        return image_backend == "dali" or (image_backend == "auto" and _has_dali())
    return False


def build_generic_feature_cache(
    *,
    store: FeatureCacheStore,
    spec: dict[str, Any],
    dataset: ForgeDataset,
    model,
    split_layer: int,
    image_size: int,
    clip_len: int,
    clip_stride: int,
    image_backend: str,
    video_backend: str,
    shard_size: int,
    batch_size: int = 32,
    num_workers: int | None = None,
    amp_dtype: torch.dtype | None = None,
    cache_dtype: torch.dtype | None = torch.float16,
    train_fraction: float = 1.0,
    train_seed: int = 0,
) -> None:
    model.backbone.eval()
    device = next(model.parameters()).device
    reader_cache_size = int(spec.get("reader_cache_size", 4))

    # DALI requires num_workers=0 (pipelines cannot be forked) — mirror trainer policy
    use_dali = _dali_active(video_backend=video_backend, image_backend=image_backend, media=dataset.media)
    if use_dali:
        resolved_workers = 0
    elif num_workers is not None:
        resolved_workers = int(num_workers)
    else:
        resolved_workers = max(2, min(8, os.cpu_count() or 1))

    if tqdm is not None:
        fraction_str = f", fraction={train_fraction:.2f}" if train_fraction < 1.0 else ""
        tqdm.write(
            f"building feature cache at {store.cache_dir} "
            f"(batch_size={batch_size}, workers={resolved_workers}, backend={'dali' if use_dali else video_backend if dataset.media == 'video' else image_backend}{fraction_str})"
        )

    torch_dataset = _ForgeRecordDataset(dataset)
    if train_fraction < 1.0 and torch_dataset._records:
        k = max(1, int(round(len(torch_dataset._records) * train_fraction)))
        torch_dataset._records = _random_module.Random(train_seed).sample(torch_dataset._records, k)
    collate_fn = functools.partial(
        _cache_build_collate_fn,
        records=dataset.records,
        media=dataset.media,
        image_size=image_size,
        clip_len=clip_len,
        clip_stride=clip_stride,
        image_backend=image_backend,
        video_backend=video_backend,
        reader_cache_size=reader_cache_size,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": resolved_workers,
        "collate_fn": collate_fn,
        "pin_memory": (device.type == "cuda" and not use_dali),
    }
    if resolved_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(torch_dataset, **loader_kwargs)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (amp_dtype is not None and device.type == "cuda")
        else nullcontext()
    )

    iterator = loader if tqdm is None else tqdm(
        loader,
        desc=f"cache:{dataset.task}:{dataset.split}",
        total=len(loader),
        dynamic_ncols=True,
    )

    with store.open_streaming_write(spec=spec, shard_size=shard_size) as writer:
        with torch.inference_mode():
            with autocast_ctx:
                for batch in iterator:
                    x = batch["tensors"].to(device, non_blocking=True)
                    if dataset.media == "video" and x.ndim == 5:
                        # read_video_clip returns [T, C, H, W]; stacked → [B, T, C, H, W]
                        # backbone expects [B, C, T, H, W]
                        x = x.permute(0, 2, 1, 3, 4).contiguous()
                    items = model.backbone.build_cache_items_batch(x, media=dataset.media, split_layer=split_layer)
                    for key, item in zip(batch["keys"], items):
                        if cache_dtype is not None:
                            item.cached_outputs = [t.to(dtype=cache_dtype) for t in item.cached_outputs]
                            if item.token_state is not None:
                                item.token_state = item.token_state.to(dtype=cache_dtype)
                        writer.append(key, item)
