from __future__ import annotations

from collections.abc import Sequence

import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.data.forge.schema import ForgeRecord
from vjepa_forge.data.image import read_image
from vjepa_forge.data.video import read_video_clip


class AnomalyLoader:
    def __init__(self, media: str, clip_len: int = 8, clip_stride: int = 1, image_size: int = 384, reader_cache_size: int = 4, video_backend: str = "auto", image_backend: str = "auto") -> None:
        self.media = media
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.image_size = image_size
        self.reader_cache_size = reader_cache_size
        self.video_backend = video_backend
        self.image_backend = image_backend

    def collate(self, records: Sequence[ForgeRecord]) -> ForgeBatch:
        xs: list[torch.Tensor] = []
        statuses: list[float] = []
        intervals: list[list[dict]] = []
        meta: list[dict] = []
        for record in records:
            if self.media == "image":
                tensor = read_image(record.media_path, image_size=self.image_size, image_backend=self.image_backend, reader_cache_size=self.reader_cache_size)
            else:
                tensor = read_video_clip(
                    record.media_path,
                    clip_len=self.clip_len,
                    stride=self.clip_stride,
                    image_size=self.image_size,
                    reader_cache_size=self.reader_cache_size,
                    video_backend=self.video_backend,
                )
            xs.append(tensor)
            anomaly_annotations = [annotation.payload for annotation in record.annotations if annotation.op == "ano"]
            abnormal = any(ann.get("status") == "abnormal" for ann in anomaly_annotations)
            statuses.append(float(abnormal))
            intervals.append(anomaly_annotations)
            meta.append({"label_path": record.label_path, "frame_count": int(tensor.shape[0] if self.media == "video" else 1)})
        return ForgeBatch(
            x=torch.stack(xs, dim=0),
            media=self.media,
            task="anomaly",
            labels={"targets": torch.tensor(statuses, dtype=torch.float32), "intervals": intervals},
            paths=[record.media_path for record in records],
            meta=meta,
        )
