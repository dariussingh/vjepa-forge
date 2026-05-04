from __future__ import annotations

from collections.abc import Sequence

import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.data.forge.schema import ForgeRecord
from vjepa_forge.data.image import read_image
from vjepa_forge.data.video import read_video_clip


class ClassifyLoader:
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
        labels: list[int] = []
        meta: list[dict] = []
        for record in records:
            if self.media == "image":
                xs.append(read_image(record.media_path, image_size=self.image_size, image_backend=self.image_backend, reader_cache_size=self.reader_cache_size))
                annotation = record.annotations[0]
                label_ids = annotation.payload.get("class_ids", [annotation.payload.get("class_id", 0)])
                labels.append(int(label_ids[0]))
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
                labels.append(int(annotation.payload.get("class_id", annotation.payload.get("class_ids", [0])[0])))
            meta.append({"label_path": record.label_path})
        x = torch.stack(xs, dim=0)
        return ForgeBatch(
            x=x,
            media=self.media,
            task="classify",
            labels={"class_ids": torch.tensor(labels, dtype=torch.long)},
            paths=[record.media_path for record in records],
            meta=meta,
        )
