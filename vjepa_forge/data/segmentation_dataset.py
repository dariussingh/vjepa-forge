from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, resize


def _load_resized_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = resize(image, [image_size, image_size])
    return pil_to_tensor(image).float() / 255.0


def _load_semantic_mask(path: Path, output_size: int) -> torch.Tensor:
    mask = Image.open(path)
    mask = resize(mask, [output_size, output_size], interpolation=Image.Resampling.NEAREST)
    return torch.from_numpy(np.array(mask, dtype="int64"))


def _rasterize_polygons(segmentation: list[list[float]], *, width: int, height: int, output_size: int) -> torch.Tensor:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) >= 6:
            points = [(polygon[idx], polygon[idx + 1]) for idx in range(0, len(polygon), 2)]
            draw.polygon(points, outline=1, fill=1)
    mask = resize(mask, [output_size, output_size], interpolation=Image.Resampling.NEAREST)
    return torch.from_numpy(np.array(mask, dtype="float32"))


class RandomSegmentationDataset(Dataset):
    def __init__(self, length: int = 8, image_size: int = 384, num_classes: int = 8) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(3, self.image_size, self.image_size)
        mask = torch.randint(0, self.num_classes, (self.image_size // 16, self.image_size // 16))
        return image, mask


class ADE20KSegmentationDataset(Dataset):
    def __init__(self, root: str | Path, *, split: str = "train", image_size: int = 384, output_stride: int = 16) -> None:
        self.root = Path(root).expanduser().resolve()
        split_name = "training" if split.lower().startswith("train") else "validation"
        self.images_dir = self.root / "images" / split_name
        self.annotations_dir = self.root / "annotations" / split_name
        if not self.images_dir.exists() or not self.annotations_dir.exists():
            raise FileNotFoundError(
                f"Expected ADE20K layout under {self.root} with images/{split_name} and annotations/{split_name}"
            )
        self.image_size = image_size
        self.output_size = image_size // output_stride
        self.samples = sorted(path for path in self.images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.samples:
            raise RuntimeError(f"No ADE20K images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.samples[index]
        mask_path = self.annotations_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing ADE20K mask for {image_path.name}: {mask_path}")
        return _load_resized_image(image_path, self.image_size), _load_semantic_mask(mask_path, self.output_size)


class COCOInstanceSegmentationDataset(Dataset):
    def __init__(self, root: str | Path, *, split: str = "train", image_size: int = 384, output_stride: int = 16) -> None:
        self.root = Path(root).expanduser().resolve()
        split_name = "train2017" if split.lower().startswith("train") else "val2017"
        self.images_dir = self.root / "images" / split_name
        self.annotations_path = self.root / "annotations" / f"instances_{split_name}.json"
        if not self.images_dir.exists() or not self.annotations_path.exists():
            raise FileNotFoundError(
                f"Expected COCO layout under {self.root} with images/{split_name} and annotations/instances_{split_name}.json"
            )
        self.image_size = image_size
        self.output_size = image_size // output_stride
        with self.annotations_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.images = {image["id"]: image for image in payload["images"]}
        self.categories = {category["id"]: idx for idx, category in enumerate(sorted(payload["categories"], key=lambda item: item["id"]))}
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in payload["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            self.annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self.image_ids = [image_id for image_id in sorted(self.images) if self.annotations_by_image.get(image_id)]
        if not self.image_ids:
            raise RuntimeError(f"No COCO instance annotations found under {self.annotations_path}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        record = self.images[image_id]
        image_path = self.images_dir / record["file_name"]
        image = _load_resized_image(image_path, self.image_size)
        width = int(record["width"])
        height = int(record["height"])
        masks: list[torch.Tensor] = []
        labels: list[int] = []
        for annotation in self.annotations_by_image.get(image_id, []):
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list):
                continue
            mask = _rasterize_polygons(segmentation, width=width, height=height, output_size=self.output_size)
            if float(mask.sum()) <= 0:
                continue
            masks.append(mask)
            labels.append(self.categories[int(annotation["category_id"])])
        if not masks:
            masks_tensor = torch.empty(0, self.output_size, self.output_size, dtype=torch.float32)
            labels_tensor = torch.empty(0, dtype=torch.int64)
        else:
            masks_tensor = torch.stack(masks, dim=0)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        return image, {"labels": labels_tensor, "masks": masks_tensor}


def collate_instance_segmentation_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    inputs, targets = zip(*batch)
    return torch.stack(list(inputs), dim=0), list(targets)
