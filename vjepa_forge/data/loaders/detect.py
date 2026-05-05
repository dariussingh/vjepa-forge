from __future__ import annotations

from collections.abc import Sequence

import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.data.forge.schema import ForgeRecord
from vjepa_forge.data.image import read_image
from vjepa_forge.data.video import read_video_clip


class DetectLoader:
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
        targets: list[dict] = []
        for record in records:
            annotations = [annotation.payload for annotation in record.annotations if annotation.op == "det"]
            if self.media == "image":
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
                annotations = [ann for ann in annotations if 0 <= int(ann["frame_idx"]) < self.clip_len]
            targets.append({"detections": annotations})
        return ForgeBatch(
            x=torch.stack(xs, dim=0),
            media=self.media,
            task="detect",
            labels={"detections": targets},
            paths=[record.media_path for record in records],
            meta=[{"label_path": record.label_path} for record in records],
        )
