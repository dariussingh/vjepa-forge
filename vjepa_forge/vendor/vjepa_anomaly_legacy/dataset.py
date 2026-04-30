from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VideoRecord:
    name: str
    frame_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...] | None = None
    frame_labels: tuple[int, ...] | None = None


@dataclass(frozen=True)
class WindowRecord:
    video_name: str
    past_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    future_labels: tuple[int, ...] | None


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    dataset_root: Path
    train_videos: list[VideoRecord]
    val_videos: list[VideoRecord]
    test_videos: list[VideoRecord]


def _sorted_frames(frame_dir: Path, suffixes: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in suffixes))


def discover_ped2_root(root: str | Path, category: str = "UCSDped2") -> Path:
    root = Path(root)
    candidates = [
        root / "UCSD_Anomaly_Dataset.v1p2" / category,
        root / category,
    ]
    for candidate in candidates:
        if (candidate / "Train").exists() and (candidate / "Test").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {category} under {root}. Expected either "
        f"{root / 'UCSD_Anomaly_Dataset.v1p2' / category} or {root / category}."
    )


def discover_avenue_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = [
        root / "processed",
        root / "Avenue",
        root,
    ]
    for candidate in candidates:
        if (
            (candidate / "Train").exists()
            and (candidate / "Test").exists()
            and (candidate / "frame_labels.json").exists()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find processed CUHK Avenue data under {root}. "
        f"Expected Train/, Test/, and frame_labels.json in either "
        f"{root / 'processed'} or {root}."
    )


def discover_cafe_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = [
        root / "processed",
        root / "cafe",
        root,
    ]
    for candidate in candidates:
        if (
            (candidate / "Train").exists()
            and (candidate / "Test").exists()
            and (candidate / "frame_labels.json").exists()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find processed cafe data under {root}. "
        f"Expected Train/, Test/, and frame_labels.json in either "
        f"{root / 'processed'} or {root}."
    )


def _frame_labels_from_masks(mask_paths: tuple[Path, ...]) -> tuple[int, ...]:
    labels: list[int] = []
    for mask_path in mask_paths:
        arr = np.asarray(Image.open(mask_path))
        labels.append(int(np.any(arr > 0)))
    return tuple(labels)


def _load_label_map(label_path: Path) -> dict[str, tuple[int, ...]]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    return {name: tuple(int(x) for x in values) for name, values in payload.items()}


def load_frame_directory_records(
    split_dir: str | Path,
    *,
    label_map: dict[str, tuple[int, ...]] | None = None,
) -> list[VideoRecord]:
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    videos: list[VideoRecord] = []
    for frame_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.endswith("_gt")):
        frame_paths = _sorted_frames(frame_dir, (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"))
        if not frame_paths:
            continue
        gt_dir = split_dir / f"{frame_dir.name}_gt"
        mask_paths = None
        if gt_dir.exists():
            masks = _sorted_frames(gt_dir, (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            if len(masks) == len(frame_paths):
                mask_paths = masks

        frame_labels = None
        if label_map is not None:
            frame_labels = label_map.get(frame_dir.name)
            if frame_labels is not None and len(frame_labels) != len(frame_paths):
                raise ValueError(
                    f"Label count mismatch for {frame_dir.name}: "
                    f"{len(frame_labels)} labels vs {len(frame_paths)} frames"
                )
        videos.append(
            VideoRecord(
                name=frame_dir.name,
                frame_paths=frame_paths,
                mask_paths=mask_paths,
                frame_labels=frame_labels,
            )
        )
    if not videos:
        raise RuntimeError(f"No videos discovered under {split_dir}")
    return videos


def split_train_val_videos(
    train_videos: list[VideoRecord], val_ratio: float, seed: int
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    indices = list(range(len(train_videos)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_indices = set(indices[:val_count])
    train_split = [record for idx, record in enumerate(train_videos) if idx not in val_indices]
    val_split = [record for idx, record in enumerate(train_videos) if idx in val_indices]
    return train_split, val_split


def load_dataset_bundle(
    dataset_cfg: dict[str, Any],
) -> DatasetBundle:
    dataset_name = dataset_cfg["name"].lower()
    root = Path(dataset_cfg["root"])

    if dataset_name == "ped2":
        dataset_root = discover_ped2_root(root, category=dataset_cfg.get("category", "UCSDped2"))
        train_videos = load_frame_directory_records(dataset_root / "Train")
        test_videos = load_frame_directory_records(dataset_root / "Test")
    elif dataset_name == "avenue":
        dataset_root = discover_avenue_root(root)
        label_map = _load_label_map(dataset_root / "frame_labels.json")
        train_videos = load_frame_directory_records(dataset_root / "Train")
        test_videos = load_frame_directory_records(dataset_root / "Test", label_map=label_map)
    elif dataset_name == "cafe":
        dataset_root = discover_cafe_root(root)
        label_map = _load_label_map(dataset_root / "frame_labels.json")
        train_videos = load_frame_directory_records(dataset_root / "Train")
        test_videos = load_frame_directory_records(dataset_root / "Test", label_map=label_map)
    else:
        raise ValueError(f"Unsupported dataset.name: {dataset_name}")

    split_train, split_val = split_train_val_videos(
        train_videos,
        val_ratio=dataset_cfg["val_ratio"],
        seed=dataset_cfg["split_seed"],
    )
    return DatasetBundle(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        train_videos=split_train,
        val_videos=split_val,
        test_videos=test_videos,
    )


def build_window_records(
    videos: list[VideoRecord],
    past_frames: int,
    future_frames: int,
    stride: int,
) -> list[WindowRecord]:
    if past_frames != future_frames:
        raise ValueError("This first implementation requires past_frames == future_frames")
    total = past_frames + future_frames
    windows: list[WindowRecord] = []
    for record in videos:
        frame_count = len(record.frame_paths)
        if frame_count < total:
            continue
        if record.frame_labels is not None:
            labels = record.frame_labels
        elif record.mask_paths is not None:
            labels = _frame_labels_from_masks(record.mask_paths)
        else:
            labels = None

        for start in range(0, frame_count - total + 1, stride):
            past = tuple(range(start, start + past_frames))
            future = tuple(range(start + past_frames, start + total))
            future_labels = None if labels is None else tuple(labels[idx] for idx in future)
            windows.append(
                WindowRecord(
                    video_name=record.name,
                    past_indices=past,
                    future_indices=future,
                    future_labels=future_labels,
                )
            )
    if not windows:
        raise RuntimeError("No sliding windows could be built from the discovered videos")
    return windows


def _load_frame_tensor(frame_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(frame_path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.stack([arr, arr, arr], axis=0)
    return torch.from_numpy(rgb)


class ClipDataset(Dataset):
    def __init__(
        self,
        videos: list[VideoRecord],
        windows: list[WindowRecord],
        image_size: int,
    ) -> None:
        self.video_lookup = {record.name: record for record in videos}
        self.windows = windows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        record = self.video_lookup[window.video_name]
        past_frames = [
            _load_frame_tensor(record.frame_paths[frame_idx], self.image_size)
            for frame_idx in window.past_indices
        ]
        future_frames = [
            _load_frame_tensor(record.frame_paths[frame_idx], self.image_size)
            for frame_idx in window.future_indices
        ]
        past = torch.stack(past_frames, dim=1)
        future = torch.stack(future_frames, dim=1)
        sample: dict[str, Any] = {
            "past": past,
            "future": future,
            "video_name": window.video_name,
            "future_indices": torch.tensor(window.future_indices, dtype=torch.long),
        }
        if window.future_labels is not None:
            sample["future_labels"] = torch.tensor(window.future_labels, dtype=torch.long)
        return sample


def write_manifest(
    output_path: str | Path,
    dataset_name: str,
    dataset_root: str | Path,
    train_videos: list[VideoRecord],
    val_videos: list[VideoRecord],
    test_videos: list[VideoRecord],
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "train_videos": [video.name for video in train_videos],
        "val_videos": [video.name for video in val_videos],
        "test_videos": [video.name for video in test_videos],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
