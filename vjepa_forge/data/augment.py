from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F


class Identity:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x.flip(-1)
        return x


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x.flip(-2)
        return x


class RandomResizedCrop:
    """Crop a random region and resize to `size`. Works on (C, H, W) or (C, T, H, W)."""

    def __init__(self, size: int | list[int], scale: tuple[float, float] = (0.8, 1.0), ratio: tuple[float, float] = (0.75, 1.333)) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            crop_w = int(round((target_area * aspect) ** 0.5))
            crop_h = int(round((target_area / aspect) ** 0.5))
            if 0 < crop_h <= h and 0 < crop_w <= w:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                x = x[..., top : top + crop_h, left : left + crop_w]
                return F.interpolate(x.unsqueeze(0) if x.dim() == 3 else x, size=self.size, mode="bilinear", align_corners=False).squeeze(0) if x.dim() == 3 else F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return F.interpolate(x.unsqueeze(0) if x.dim() == 3 else x, size=self.size, mode="bilinear", align_corners=False).squeeze(0) if x.dim() == 3 else F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)


class ColorJitter:
    """Randomly adjust brightness, contrast, saturation, and hue."""

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0, hue: float = 0.0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            x = (x * factor).clamp(0, 1)
        if self.contrast > 0:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            mean = x.mean(dim=(-3, -2, -1), keepdim=True)
            x = ((x - mean) * factor + mean).clamp(0, 1)
        return x


class GaussianBlur:
    """Apply Gaussian blur with random sigma."""

    def __init__(self, kernel_size: int = 3, sigma: tuple[float, float] = (0.1, 2.0)) -> None:
        self.kernel_size = kernel_size | 1  # ensure odd
        self.sigma = sigma

    def _make_kernel(self, sigma: float) -> torch.Tensor:
        k = self.kernel_size
        x = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        return g.outer(g)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(*self.sigma)
        kernel = self._make_kernel(sigma).to(x.device)
        k = self.kernel_size
        pad = k // 2
        # works on (C, H, W) — treat C as batch dim
        if x.dim() == 3:
            c = x.shape[0]
            xp = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
            kernel4d = kernel.view(1, 1, k, k).expand(c, 1, k, k)
            return F.conv2d(xp, kernel4d, groups=c).squeeze(0).clamp(0, 1)
        return x


class Normalize:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device).view(-1, *([1] * (x.dim() - 1)))
        std = self.std.to(x.device).view(-1, *([1] * (x.dim() - 1)))
        return (x - mean) / std.clamp(min=1e-7)


class RandomGrayscale:
    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p and x.shape[-3] >= 3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=x.dtype, device=x.device)
            gray = (x[..., :3, :, :] * weights.view(3, 1, 1)).sum(dim=-3, keepdim=True)
            x = gray.expand_as(x[..., :3, :, :])
        return x


def build_image_augment_pipeline(data_cfg: dict[str, Any]) -> Compose:
    """Build a composable image augmentation pipeline from config."""
    aug_cfg = data_cfg.get("augment", {}) or {}
    if not aug_cfg.get("enabled", False):
        return Compose([Identity()])
    transforms = []
    if aug_cfg.get("random_hflip", 0.0) > 0:
        transforms.append(RandomHorizontalFlip(p=float(aug_cfg["random_hflip"])))
    if aug_cfg.get("random_vflip", 0.0) > 0:
        transforms.append(RandomVerticalFlip(p=float(aug_cfg["random_vflip"])))
    crop = aug_cfg.get("random_resized_crop")
    if crop is not None:
        transforms.append(RandomResizedCrop(size=crop))
    jitter = aug_cfg.get("color_jitter")
    if jitter is not None:
        transforms.append(ColorJitter(*jitter))
    blur = aug_cfg.get("gaussian_blur")
    if blur is not None:
        transforms.append(GaussianBlur(kernel_size=int(blur[0]), sigma=(float(blur[1]), float(blur[2]))))
    if aug_cfg.get("random_grayscale", 0.0) > 0:
        transforms.append(RandomGrayscale(p=float(aug_cfg["random_grayscale"])))
    mean = aug_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = aug_cfg.get("normalize_std", [0.229, 0.224, 0.225])
    transforms.append(Normalize(mean=mean, std=std))
    return Compose(transforms)


def build_video_augment_pipeline(data_cfg: dict[str, Any]) -> Compose:
    """Build a per-frame video augmentation pipeline from config."""
    return build_image_augment_pipeline(data_cfg)


__all__ = [
    "ColorJitter",
    "Compose",
    "GaussianBlur",
    "Identity",
    "Normalize",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "RandomVerticalFlip",
    "build_image_augment_pipeline",
    "build_video_augment_pipeline",
]
