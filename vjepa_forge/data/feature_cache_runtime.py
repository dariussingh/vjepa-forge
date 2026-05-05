from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vjepa_forge.data.cache import FeatureCacheStore, cached_feature_item_key, default_feature_cache_root, manifest_cache_dir, serialize_spec
from vjepa_forge.data.forge.dataset import ForgeDataset
from vjepa_forge.data.image import read_image
from vjepa_forge.data.video import read_video_clip


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
        }
    )


def resolve_generic_cache_store(
    *,
    dataset: ForgeDataset,
    model,
    split: str,
    data_cfg: dict[str, Any],
    freeze_cfg: dict[str, Any] | None,
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
        )
    elif settings.validate and not store.spec_matches(spec):
        if settings.enabled == "true":
            raise ValueError(f"Feature cache spec mismatch for split={split}: {store.cache_dir}")
        return None
    return store


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
) -> None:
    items: list[tuple[str, Any]] = []
    model.backbone.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for record in dataset.records:
            if dataset.media == "image":
                tensor = read_image(record.media_path, image_size=image_size, image_backend=image_backend, reader_cache_size=int(spec.get("reader_cache_size", 4)))
                key = cached_feature_item_key(media_path=record.media_path)
            else:
                tensor = read_video_clip(
                    record.media_path,
                    clip_len=clip_len,
                    stride=clip_stride,
                    image_size=image_size,
                    reader_cache_size=int(spec.get("reader_cache_size", 4)),
                    video_backend=video_backend,
                )
                key = cached_feature_item_key(media_path=record.media_path, clip_len=clip_len, stride=clip_stride)
            item = model.backbone.build_cache_item(tensor.unsqueeze(0).to(device), media=dataset.media, split_layer=split_layer)
            items.append((key, item))
    store.write(spec=spec, items=items, shard_size=shard_size)
