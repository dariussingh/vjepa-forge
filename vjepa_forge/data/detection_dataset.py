from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, resize


def _resize_boxes_xyxy(boxes: torch.Tensor, *, src_size: tuple[int, int], dst_size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    src_width, src_height = src_size
    scale_x = float(dst_size) / max(src_width, 1)
    scale_y = float(dst_size) / max(src_height, 1)
    resized = boxes.clone().to(dtype=torch.float32)
    resized[:, [0, 2]] *= scale_x
    resized[:, [1, 3]] *= scale_y
    return resized


def _load_image(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    src_size = image.size
    image = resize(image, [image_size, image_size])
    tensor = pil_to_tensor(image).float() / 255.0
    return tensor, src_size


def _empty_frame_target() -> dict[str, torch.Tensor]:
    return {
        "labels": torch.empty(0, dtype=torch.int64),
        "boxes": torch.empty(0, 4, dtype=torch.float32),
        "track_ids": torch.empty(0, dtype=torch.int64),
    }


class RandomDetectionDataset(Dataset):
    def __init__(self, length: int = 8, image_size: int = 384, num_classes: int = 10) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = torch.randn(3, self.image_size, self.image_size)
        half = self.image_size * 0.5
        quarter = self.image_size * 0.25
        target = {
            "labels": torch.tensor([index % self.num_classes], dtype=torch.int64),
            "boxes": torch.tensor([[quarter, quarter, half, half]], dtype=torch.float32),
        }
        return image, target


class COCODetectionDataset(Dataset):
    def __init__(self, root: str | Path, *, split: str = "train", image_size: int = 384) -> None:
        self.root = Path(root).expanduser().resolve()
        self.image_size = image_size
        split_name = "train2017" if split.lower().startswith("train") else "val2017"
        self.images_dir = self.root / "images" / split_name
        self.annotations_path = self.root / "annotations" / f"instances_{split_name}.json"
        if not self.images_dir.exists() or not self.annotations_path.exists():
            raise FileNotFoundError(
                f"Expected COCO layout under {self.root} with images/{split_name} and annotations/instances_{split_name}.json"
            )
        with self.annotations_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.images = {image["id"]: image for image in payload["images"]}
        self.categories = {category["id"]: idx for idx, category in enumerate(sorted(payload["categories"], key=lambda item: item["id"]))}
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in payload["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            self.annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self.image_ids = sorted(self.images)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        record = self.images[image_id]
        image_path = self.images_dir / record["file_name"]
        image, src_size = _load_image(image_path, self.image_size)
        annotations = self.annotations_by_image.get(image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.categories[int(annotation["category_id"])])
        target = {
            "boxes": _resize_boxes_xyxy(torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 4), src_size=src_size, dst_size=self.image_size),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty(0, dtype=torch.int64),
        }
        return image, target


class RandomVideoDetectionDataset(Dataset):
    def __init__(self, length: int = 8, image_size: int = 384, num_frames: int = 8, num_classes: int = 10) -> None:
        self.length = length
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        video = torch.randn(3, self.num_frames, self.image_size, self.image_size)
        frames: list[dict[str, torch.Tensor]] = []
        quarter = self.image_size * 0.25
        half = self.image_size * 0.5
        for frame_idx in range(self.num_frames):
            frames.append(
                {
                    "labels": torch.tensor([(index + frame_idx) % self.num_classes], dtype=torch.int64),
                    "boxes": torch.tensor([[quarter, quarter, half, half]], dtype=torch.float32),
                    "track_ids": torch.tensor([0], dtype=torch.int64),
                }
            )
        return video, {"frames": frames, "video_id": f"synthetic_{index}", "frame_indices": list(range(self.num_frames))}


@dataclass(frozen=True)
class VideoFrameRecord:
    image_path: Path
    annotation_path: Path
    frame_index: int


@dataclass(frozen=True)
class VideoClipRecord:
    video_id: str
    frames: tuple[VideoFrameRecord, ...]


def _parse_imagenet_vid_xml(path: Path, class_to_idx: dict[str, int]) -> dict[str, torch.Tensor]:
    if not path.exists():
        return _empty_frame_target()
    root = ET.parse(path).getroot()
    labels: list[int] = []
    boxes: list[list[float]] = []
    track_ids: list[int] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="0")
        trackid = obj.findtext("trackid")
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        labels.append(class_to_idx.setdefault(name, len(class_to_idx)))
        boxes.append([xmin, ymin, xmax, ymax])
        track_ids.append(int(trackid) if trackid is not None and trackid.isdigit() else len(track_ids))
    if not labels:
        return _empty_frame_target()
    return {
        "labels": torch.tensor(labels, dtype=torch.int64),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "track_ids": torch.tensor(track_ids, dtype=torch.int64),
    }


class ImageNetVIDDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        image_size: int = 384,
        clip_length: int = 8,
        clip_stride: int = 1,
        max_clips: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.image_size = image_size
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.max_clips = max_clips
        self.records = self._discover_records()
        if not self.records:
            raise RuntimeError(f"No ImageNet VID clips discovered under {self.root}")
        self.class_to_idx = self._discover_label_map()

    def _discover_records(self) -> list[VideoClipRecord]:
        split_name = "train" if self.split.lower().startswith("train") else "val"
        data_root = self.root / "Data" / "VID" / split_name
        ann_root = self.root / "Annotations" / "VID" / split_name
        if not data_root.exists() or not ann_root.exists():
            raise FileNotFoundError(
                f"Expected ImageNet VID layout under {self.root} with Data/VID/{split_name} and Annotations/VID/{split_name}"
            )
        records: list[VideoClipRecord] = []
        for image_dir in sorted(path for path in data_root.rglob("*") if path.is_dir()):
            frame_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
            if not frame_paths:
                continue
            rel_dir = image_dir.relative_to(data_root)
            annotation_dir = ann_root / rel_dir
            if not annotation_dir.exists():
                continue
            frames = [
                VideoFrameRecord(
                    image_path=frame_path,
                    annotation_path=annotation_dir / f"{frame_path.stem}.xml",
                    frame_index=idx,
                )
                for idx, frame_path in enumerate(frame_paths)
            ]
            for start in range(0, len(frames), self.clip_stride):
                clip = list(frames[start : start + self.clip_length])
                if not clip:
                    continue
                if len(clip) < self.clip_length:
                    clip.extend([clip[-1]] * (self.clip_length - len(clip)))
                records.append(VideoClipRecord(video_id=str(rel_dir), frames=tuple(clip)))
                if self.max_clips is not None and len(records) >= self.max_clips:
                    return records
        return records

    def _discover_label_map(self) -> dict[str, int]:
        class_to_idx: dict[str, int] = {}
        for record in self.records:
            for frame in record.frames:
                if not frame.annotation_path.exists():
                    continue
                root = ET.parse(frame.annotation_path).getroot()
                for obj in root.findall("object"):
                    name = obj.findtext("name")
                    if name and name not in class_to_idx:
                        class_to_idx[name] = len(class_to_idx)
        return class_to_idx

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        frame_tensors: list[torch.Tensor] = []
        frame_targets: list[dict[str, torch.Tensor]] = []
        frame_indices: list[int] = []
        for frame in record.frames:
            image, src_size = _load_image(frame.image_path, self.image_size)
            target = _parse_imagenet_vid_xml(frame.annotation_path, self.class_to_idx)
            target["boxes"] = _resize_boxes_xyxy(target["boxes"], src_size=src_size, dst_size=self.image_size)
            frame_tensors.append(image)
            frame_targets.append(target)
            frame_indices.append(frame.frame_index)
        video = torch.stack(frame_tensors, dim=1)
        return video, {"frames": frame_targets, "video_id": record.video_id, "frame_indices": frame_indices}


def collate_detection_batch(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    inputs, targets = zip(*batch)
    return torch.stack(list(inputs), dim=0), list(targets)
