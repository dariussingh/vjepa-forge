from __future__ import annotations

from collections.abc import Sequence

import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.data.cache import FeatureCacheStore, cached_feature_item_key, stack_cached_feature_items
from vjepa_forge.data.forge.schema import ForgeRecord
from vjepa_forge.data.image import read_image
from vjepa_forge.data.video import read_video_clip


class AnomalyLoader:
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
        statuses: list[float] = []
        intervals: list[list[dict]] = []
        meta: list[dict] = []
        for record in records:
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
                frame_count = self.clip_len if self.media == "video" else 1
            elif self.media == "image":
                tensor = read_image(record.media_path, image_size=self.image_size, image_backend=self.image_backend, reader_cache_size=self.reader_cache_size)
                frame_count = 1
            else:
                tensor = read_video_clip(
                    record.media_path,
                    clip_len=self.clip_len,
                    stride=self.clip_stride,
                    image_size=self.image_size,
                    reader_cache_size=self.reader_cache_size,
                    video_backend=self.video_backend,
                )
                frame_count = int(tensor.shape[0])
            if self.feature_cache is None:
                xs.append(tensor)
            anomaly_annotations = [annotation.payload for annotation in record.annotations if annotation.op == "ano"]
            abnormal = any(ann.get("status") == "abnormal" for ann in anomaly_annotations)
            statuses.append(float(abnormal))
            intervals.append(anomaly_annotations)
            meta.append({"label_path": record.label_path, "frame_count": frame_count})
        return ForgeBatch(
            x=stack_cached_feature_items(cached) if self.feature_cache is not None else torch.stack(xs, dim=0),
            media=self.media,
            task="anomaly",
            labels={"targets": torch.tensor(statuses, dtype=torch.float32), "intervals": intervals},
            paths=[record.media_path for record in records],
            meta=meta,
        )
