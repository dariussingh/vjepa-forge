from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


@dataclass
class CachedFeatureItem:
    mode: str
    media: str
    split_layer: int
    token_state: torch.Tensor | None
    cached_outputs: list[torch.Tensor]
    height_patches: int
    width_patches: int
    temporal_tokens: int


@dataclass
class CachedFeatureBatch:
    mode: str
    media: str
    split_layer: int
    token_state: torch.Tensor | None
    cached_outputs: list[torch.Tensor]
    height_patches: int
    width_patches: int
    temporal_tokens: int


@dataclass(frozen=True)
class CacheLocation:
    shard: str
    index: int


def stack_cached_feature_items(items: list[CachedFeatureItem]) -> CachedFeatureBatch:
    if not items:
        raise ValueError("Cannot stack an empty cached feature batch")
    first = items[0]
    if any(
        item.mode != first.mode
        or item.media != first.media
        or item.split_layer != first.split_layer
        or item.height_patches != first.height_patches
        or item.width_patches != first.width_patches
        or item.temporal_tokens != first.temporal_tokens
        or len(item.cached_outputs) != len(first.cached_outputs)
        for item in items[1:]
    ):
        raise ValueError("Incompatible cached feature items in batch")
    token_state = None
    if first.token_state is not None:
        token_state = torch.stack([item.token_state for item in items], dim=0)
    cached_outputs = [torch.stack([item.cached_outputs[idx] for item in items], dim=0) for idx in range(len(first.cached_outputs))]
    return CachedFeatureBatch(
        mode=first.mode,
        media=first.media,
        split_layer=first.split_layer,
        token_state=token_state,
        cached_outputs=cached_outputs,
        height_patches=first.height_patches,
        width_patches=first.width_patches,
        temporal_tokens=first.temporal_tokens,
    )


def _json_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def default_feature_cache_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root).expanduser().resolve() / ".forge_cache" / "features"


def manifest_cache_dir(cache_root: str | Path, spec: dict[str, Any]) -> Path:
    cache_key = _json_hash(spec)
    return Path(cache_root).expanduser().resolve() / cache_key


class FeatureCacheStore:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest_path = self.cache_dir / "manifest.json"
        self._manifest: dict[str, Any] | None = None
        self._loaded_shard_name: str | None = None
        self._loaded_shard_items: list[dict[str, Any]] | None = None

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    def spec_matches(self, expected_spec: dict[str, Any]) -> bool:
        if not self.exists():
            return False
        manifest = self.load_manifest()
        return dict(manifest.get("spec", {})) == dict(expected_spec)

    def item_exists(self, key: str) -> bool:
        manifest = self.load_manifest()
        return key in manifest.get("items", {})

    def get(self, key: str) -> CachedFeatureItem:
        manifest = self.load_manifest()
        entry = dict(manifest["items"][key])
        shard_name = str(entry["shard"])
        items = self._load_shard(shard_name)
        payload = items[int(entry["index"])]
        return CachedFeatureItem(
            mode=str(payload["mode"]),
            media=str(payload["media"]),
            split_layer=int(payload["split_layer"]),
            token_state=payload["token_state"],
            cached_outputs=list(payload["cached_outputs"]),
            height_patches=int(payload["height_patches"]),
            width_patches=int(payload["width_patches"]),
            temporal_tokens=int(payload["temporal_tokens"]),
        )

    def write(self, *, spec: dict[str, Any], items: list[tuple[str, CachedFeatureItem]], shard_size: int = 64) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_items: dict[str, dict[str, Any]] = {}
        shard_payload: list[dict[str, Any]] = []
        shard_index = 0
        item_count = 0
        for key, item in items:
            shard_payload.append(
                {
                    "mode": item.mode,
                    "media": item.media,
                    "split_layer": item.split_layer,
                    "token_state": item.token_state,
                    "cached_outputs": item.cached_outputs,
                    "height_patches": item.height_patches,
                    "width_patches": item.width_patches,
                    "temporal_tokens": item.temporal_tokens,
                }
            )
            manifest_items[key] = {"shard": f"shard_{shard_index:05d}.pt", "index": len(shard_payload) - 1}
            item_count += 1
            if len(shard_payload) >= max(1, int(shard_size)):
                torch.save(shard_payload, self.cache_dir / f"shard_{shard_index:05d}.pt")
                shard_payload = []
                shard_index += 1
        if shard_payload:
            torch.save(shard_payload, self.cache_dir / f"shard_{shard_index:05d}.pt")
        manifest = {"version": 1, "spec": spec, "item_count": item_count, "items": manifest_items}
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        self._manifest = manifest
        self._loaded_shard_name = None
        self._loaded_shard_items = None

    def _load_shard(self, shard_name: str) -> list[dict[str, Any]]:
        if self._loaded_shard_name != shard_name:
            self._loaded_shard_name = shard_name
            self._loaded_shard_items = torch.load(self.cache_dir / shard_name, map_location="cpu", weights_only=False)
        if self._loaded_shard_items is None:
            raise RuntimeError(f"Failed to load cache shard {shard_name}")
        return self._loaded_shard_items


def cached_feature_item_key(*, media_path: str, clip_start: int = 0, clip_len: int | None = None, stride: int = 1) -> str:
    payload = {
        "media_path": str(Path(media_path).expanduser().resolve()),
        "clip_start": int(clip_start),
        "clip_len": None if clip_len is None else int(clip_len),
        "stride": int(stride),
    }
    return _json_hash(payload)


def recursive_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, CachedFeatureBatch):
        token_state = None if value.token_state is None else recursive_to_device(value.token_state, device)
        cached_outputs = [recursive_to_device(output, device) for output in value.cached_outputs]
        return CachedFeatureBatch(
            mode=value.mode,
            media=value.media,
            split_layer=value.split_layer,
            token_state=token_state,
            cached_outputs=cached_outputs,
            height_patches=value.height_patches,
            width_patches=value.width_patches,
            temporal_tokens=value.temporal_tokens,
        )
    if isinstance(value, list):
        return [recursive_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: recursive_to_device(item, device) for key, item in value.items()}
    return value


def serialize_spec(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))
