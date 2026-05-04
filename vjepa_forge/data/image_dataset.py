from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor, resize


def load_rgb_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = resize(image, [image_size, image_size])
    return pil_to_tensor(image).float() / 255.0


class RandomImageDataset(Dataset):
    def __init__(self, length: int = 8, image_size: int = 384, num_classes: int = 10) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.randn(3, self.image_size, self.image_size)
        label = index % self.num_classes
        return image, label


class ImageNet1KDataset(Dataset):
    def __init__(self, root: str | Path, *, split: str = "train", image_size: int = 384) -> None:
        self.root = Path(root).expanduser().resolve()
        self.image_size = image_size
        split_dir = self.root / ("train" if split.lower().startswith("train") else "val")
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected ImageNet split at {split_dir}")
        self.dataset = ImageFolder(split_dir)
        self.classes = self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.dataset.samples[index]
        return load_rgb_tensor(path, self.image_size), int(label)
