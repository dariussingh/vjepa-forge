from __future__ import annotations

from pathlib import Path

import decord
import torch
from torch.utils.data import Dataset


def _discover_labeled_videos(split_dir: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for class_name in classes:
        class_dir = split_dir / class_name
        for video_path in sorted(class_dir.rglob("*")):
            if video_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                samples.append((video_path, class_to_idx[class_name]))
    if not samples:
        raise RuntimeError(f"No video files discovered under {split_dir}")
    return samples, classes


def _load_video_clip(path: Path, num_frames: int, image_size: int) -> torch.Tensor:
    reader = decord.VideoReader(str(path))
    total_frames = len(reader)
    if total_frames <= 0:
        raise RuntimeError(f"Video contains no frames: {path}")
    indices = torch.linspace(0, total_frames - 1, steps=num_frames).round().to(dtype=torch.int64).tolist()
    frames = reader.get_batch(indices).asnumpy()
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    tensor = torch.nn.functional.interpolate(
        tensor.permute(1, 0, 2, 3),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).permute(1, 0, 2, 3)
    return tensor


class RandomVideoDataset(Dataset):
    def __init__(self, length: int = 8, image_size: int = 384, num_frames: int = 8, num_classes: int = 10) -> None:
        self.length = length
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        video = torch.randn(3, self.num_frames, self.image_size, self.image_size)
        label = index % self.num_classes
        return video, label


class Kinetics400Dataset(Dataset):
    def __init__(self, root: str | Path, *, split: str = "train", image_size: int = 384, num_frames: int = 8) -> None:
        self.root = Path(root).expanduser().resolve()
        self.image_size = image_size
        self.num_frames = num_frames
        split_dir = self.root / ("train" if split.lower().startswith("train") else "val")
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected Kinetics split at {split_dir}")
        self.samples, self.classes = _discover_labeled_videos(split_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        video_path, label = self.samples[index]
        return _load_video_clip(video_path, self.num_frames, self.image_size), int(label)
