from __future__ import annotations

from collections.abc import Sequence

import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.data.cache import FeatureCacheStore, cached_feature_item_key, stack_cached_feature_items
from vjepa_forge.data.forge.schema import ForgeRecord
from vjepa_forge.data.image import read_image
from vjepa_forge.data.video import read_video_clip


class ClassifyLoader:
    def __init__(self, media: str, clip_len: int = 8, clip_stride: int = 1, image_size: int = 384, reader_cache_size: int = 4, video_backend: str = "auto", image_backend: str = "auto", feature_cache: FeatureCacheStore | None = None) -> None:
        self.media = media
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.image_size = image_size
        self.reader_cache_size = reader_cache_size
        self.video_backend = video_backend
        self.image_backend = image_backend
        self.feature_cache = feature_cache

    def collate(self, records: Sequence[ForgeRecord]) -> ForgeBatch:
        xs: list[torch.Tensor] = []
        cached = []
        labels: list[int] = []
        meta: list[dict] = []
        for record in records:
            if not record.annotations:
                raise ValueError(f"No annotations for {record.media_path} (expected label file at {record.label_path})")
            if self.feature_cache is not None:
                cached.append(
                    self.feature_cache.get(
                        cached_feature_item_key(
                            media_path=record.media_path,
                            clip_start=0,
                            clip_len=None if self.media == "image" else self.clip_len,
                            stride=1 if self.media == "image" else self.clip_stride,
                        )
                    )
                )
            elif self.media == "image":
                xs.append(read_image(record.media_path, image_size=self.image_size, image_backend=self.image_backend, reader_cache_size=self.reader_cache_size))
            else:
                xs.append(
                    read_video_clip(
                        record.media_path,
                        clip_len=self.clip_len,
                        stride=self.clip_stride,
                        image_size=self.image_size,
                        reader_cache_size=self.reader_cache_size,
                        video_backend=self.video_backend,
                    )
                )
            annotation = record.annotations[0]
            label_ids = annotation.payload.get("class_ids", [annotation.payload.get("class_id", 0)])
            labels.append(int(label_ids[0]))
            meta.append({"label_path": record.label_path})
        x = stack_cached_feature_items(cached) if self.feature_cache is not None else torch.stack(xs, dim=0)
        return ForgeBatch(
            x=x,
            media=self.media,
            task="classify",
            labels={"class_ids": torch.tensor(labels, dtype=torch.long)},
            paths=[record.media_path for record in records],
            meta=meta,
        )
