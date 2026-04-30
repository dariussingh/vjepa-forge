from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_2_1.models import vision_transformer as vjepa_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger(__name__)


BACKBONE_SPECS = {
    "vit_base": {
        "factory": "vit_base",
        "embed_dim": 768,
        "out_layers": [2, 5, 8, 11],
    },
    "vit_large": {
        "factory": "vit_large",
        "embed_dim": 1024,
        "out_layers": [5, 11, 17, 23],
    },
    "vit_giant": {
        "factory": "vit_giant_xformers",
        "embed_dim": 1408,
        "out_layers": [9, 19, 29, 39],
    },
    "vit_gigantic": {
        "factory": "vit_gigantic_xformers",
        "embed_dim": 1664,
        "out_layers": [11, 23, 37, 47],
    },
}


def _strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _select_checkpoint_state(checkpoint: dict, checkpoint_key: str | None) -> dict[str, torch.Tensor]:
    if checkpoint_key and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)!r}")
    return _strip_state_dict_prefixes(state_dict)


class VJEPAImageBackbone(nn.Module):
    def __init__(
        self,
        *,
        name: str = "vit_base",
        checkpoint: str | None = None,
        checkpoint_key: str = "ema_encoder",
        mode: str = "image",
        imgsz: int = 384,
        patch_size: int = 16,
        tubelet_size: int = 2,
        use_rope: bool = True,
        use_sdpa: bool = True,
        uniform_power: bool = True,
        modality_embedding: bool = True,
        interpolate_rope: bool = True,
        out_layers: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if name not in BACKBONE_SPECS:
            raise ValueError(f"Unsupported backbone name '{name}'. Expected one of {sorted(BACKBONE_SPECS)}")
        if mode != "image":
            raise ValueError(f"Unsupported backbone mode '{mode}'. Detection currently supports only image mode.")

        spec = BACKBONE_SPECS[name]
        factory = getattr(vjepa_vit, spec["factory"])
        effective_out_layers = tuple(spec["out_layers"] if out_layers is None else out_layers)
        self.encoder = factory(
            img_size=imgsz,
            patch_size=patch_size,
            num_frames=tubelet_size,
            tubelet_size=tubelet_size,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
            uniform_power=uniform_power,
            img_temporal_dim_size=1,
            interpolate_rope=interpolate_rope,
            modality_embedding=modality_embedding,
            n_output_distillation=1,
            out_layers=effective_out_layers,
        )
        self.name = name
        self.mode = mode
        self.embed_dim = spec["embed_dim"]
        self.patch_size = patch_size
        self.imgsz = imgsz
        self.out_layers = effective_out_layers

        if checkpoint:
            self.load_checkpoint(checkpoint, checkpoint_key=checkpoint_key)

    def load_checkpoint(self, checkpoint_path: str, checkpoint_key: str = "ema_encoder") -> None:
        logger.info("Loading V-JEPA checkpoint from %s (key=%s)", checkpoint_path, checkpoint_key)
        checkpoint = robust_checkpoint_loader(checkpoint_path, map_location="cpu")
        state_dict = _select_checkpoint_state(checkpoint, checkpoint_key)
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        logger.info("Loaded encoder checkpoint with msg: %s", msg)

    def freeze(self, unfreeze_last_n_blocks: int = 0) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            for block in self.encoder.blocks[-unfreeze_last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.encoder.norms_block.parameters():
                parameter.requires_grad = True

    def unfreeze(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW image tensor, got shape {tuple(x.shape)}")

        _, _, height, width = x.shape
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size
        # Route images through V-JEPA's image-mode branch while keeping a BCHW public API.
        x = x.unsqueeze(2)
        outputs = self.encoder(x)
        if not isinstance(outputs, Iterable):
            raise TypeError("Expected hierarchical encoder outputs")

        features = []
        for tokens in outputs:
            if tokens.ndim != 3:
                raise ValueError(f"Expected token tensor shaped [B, N, C], got {tuple(tokens.shape)}")
            batch_size, num_tokens, channels = tokens.shape
            expected_tokens = height_patches * width_patches
            if num_tokens != expected_tokens:
                raise ValueError(
                    f"Cannot reshape {num_tokens} tokens into feature map "
                    f"{height_patches}x{width_patches} for input {height}x{width}"
                )
            feature_map = tokens.transpose(1, 2).reshape(batch_size, channels, height_patches, width_patches)
            features.append(feature_map)
        return features


class VJEPAVideoBackbone(nn.Module):
    def __init__(
        self,
        *,
        name: str = "vit_base",
        checkpoint: str | None = None,
        checkpoint_key: str = "ema_encoder",
        imgsz: int = 384,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 8,
        use_rope: bool = True,
        use_sdpa: bool = True,
        uniform_power: bool = True,
        modality_embedding: bool = True,
        interpolate_rope: bool = True,
        out_layers: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if name not in BACKBONE_SPECS:
            raise ValueError(f"Unsupported backbone name '{name}'. Expected one of {sorted(BACKBONE_SPECS)}")
        spec = BACKBONE_SPECS[name]
        factory = getattr(vjepa_vit, spec["factory"])
        effective_out_layers = tuple(spec["out_layers"] if out_layers is None else out_layers)
        self.encoder = factory(
            img_size=imgsz,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
            uniform_power=uniform_power,
            img_temporal_dim_size=max(1, num_frames // tubelet_size),
            interpolate_rope=interpolate_rope,
            modality_embedding=modality_embedding,
            n_output_distillation=1,
            out_layers=effective_out_layers,
        )
        self.name = name
        self.embed_dim = spec["embed_dim"]
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.imgsz = imgsz
        self.out_layers = effective_out_layers
        if checkpoint:
            self.load_checkpoint(checkpoint, checkpoint_key=checkpoint_key)

    def load_checkpoint(self, checkpoint_path: str, checkpoint_key: str = "ema_encoder") -> None:
        checkpoint = robust_checkpoint_loader(checkpoint_path, map_location="cpu")
        state_dict = _select_checkpoint_state(checkpoint, checkpoint_key)
        self.encoder.load_state_dict(state_dict, strict=False)

    def freeze(self, unfreeze_last_n_blocks: int = 0) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if unfreeze_last_n_blocks > 0:
            for block in self.encoder.blocks[-unfreeze_last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def unfreeze(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected BCTHW video tensor, got shape {tuple(x.shape)}")
        batch_size, _, time, height, width = x.shape
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size
        temporal_tokens = max(1, time // self.tubelet_size)
        outputs = self.encoder(x)
        if not isinstance(outputs, Iterable):
            raise TypeError("Expected hierarchical encoder outputs")
        features = []
        for tokens in outputs:
            if tokens.ndim != 3:
                raise ValueError(f"Expected token tensor shaped [B, N, C], got {tuple(tokens.shape)}")
            _, num_tokens, channels = tokens.shape
            expected_tokens = temporal_tokens * height_patches * width_patches
            if num_tokens != expected_tokens:
                raise ValueError(
                    f"Cannot reshape {num_tokens} tokens into feature volume "
                    f"{temporal_tokens}x{height_patches}x{width_patches}"
                )
            feature_volume = tokens.transpose(1, 2).reshape(
                batch_size,
                channels,
                temporal_tokens,
                height_patches,
                width_patches,
            )
            features.append(feature_volume)
        return features


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        padding = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=padding, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, k=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


class VJEPAFeaturePyramidAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.proj0 = ConvBNAct(in_channels, out_channels, k=1)
        self.proj1 = ConvBNAct(in_channels, out_channels, k=1)
        self.proj2 = ConvBNAct(in_channels, out_channels, k=1)
        self.proj3 = ConvBNAct(in_channels, out_channels, k=1)

        self.fuse_p4 = ConvBNAct(out_channels * 4, out_channels, k=3)
        self.refine_p3 = ConvBNAct(out_channels * 2, out_channels, k=3)
        self.refine_p5 = ConvBNAct(out_channels * 2, out_channels, k=3)
        self.downsample = ConvBNAct(out_channels, out_channels, k=3, s=2)

        self.out_channels = (out_channels, out_channels, out_channels)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"Expected 4 hierarchical features, got {len(features)}")

        f0, f1, f2, f3 = features
        p0 = self.proj0(f0)
        p1 = self.proj1(f1)
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)

        p4 = self.fuse_p4(torch.cat([p0, p1, p2, p3], dim=1))
        p3_out = self.refine_p3(
            torch.cat([F.interpolate(p0, scale_factor=2.0, mode="bilinear", align_corners=False),
                       F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False)], dim=1)
        )
        p5_seed = self.downsample(p4)
        p5_out = self.refine_p5(torch.cat([p5_seed, F.interpolate(p3, size=p5_seed.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        return [p3_out, p4, p5_out]


class VJEPAEnhancedPyramidAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        in_image_channels: int = 3,
        detail_channels: int | None = None,
    ) -> None:
        super().__init__()
        detail_channels = out_channels // 2 if detail_channels is None else detail_channels

        self.proj0 = ConvBNAct(in_channels, out_channels, k=1)
        self.proj1 = ConvBNAct(in_channels, out_channels, k=1)
        self.proj2 = ConvBNAct(in_channels, out_channels, k=1)
        self.proj3 = ConvBNAct(in_channels, out_channels, k=1)
        self.fuse_weights = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.fuse_native = nn.Sequential(
            ConvBNAct(out_channels, out_channels, k=3),
            ResidualConvBlock(out_channels),
        )

        self.detail_s2 = ConvBNAct(in_image_channels, detail_channels, k=3, s=2)
        self.detail_s4 = nn.Sequential(
            ConvBNAct(detail_channels, detail_channels, k=3, s=2),
            ResidualConvBlock(detail_channels),
        )
        self.detail_s8 = nn.Sequential(
            ConvBNAct(detail_channels, out_channels, k=3, s=2),
            ResidualConvBlock(out_channels),
        )
        self.detail_to_p2 = ConvBNAct(detail_channels, out_channels, k=1)

        self.p3_td = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, k=3),
            ResidualConvBlock(out_channels),
        )
        self.p2_td = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, k=3),
            ResidualConvBlock(out_channels),
        )

        self.p2_to_p3 = ConvBNAct(out_channels, out_channels, k=3, s=2)
        self.p3_out = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, k=3),
            ResidualConvBlock(out_channels),
        )
        self.p3_to_p4 = ConvBNAct(out_channels, out_channels, k=3, s=2)
        self.p4_out = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, k=3),
            ResidualConvBlock(out_channels),
        )
        self.p4_to_p5 = ConvBNAct(out_channels, out_channels, k=3, s=2)
        self.p5_out = nn.Sequential(
            ConvBNAct(out_channels, out_channels, k=3),
            ResidualConvBlock(out_channels),
        )

        self.out_channels = (out_channels, out_channels, out_channels, out_channels)

    def forward(self, features: list[torch.Tensor], image: torch.Tensor) -> list[torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"Expected 4 hierarchical features, got {len(features)}")

        proj_features = [
            self.proj0(features[0]),
            self.proj1(features[1]),
            self.proj2(features[2]),
            self.proj3(features[3]),
        ]
        fuse_weights = torch.softmax(self.fuse_weights, dim=0)
        fused_native = sum(weight * feature for weight, feature in zip(fuse_weights, proj_features, strict=True))
        p4_native = self.fuse_native(fused_native)

        d2 = self.detail_s2(image)
        d4 = self.detail_s4(d2)
        d8 = self.detail_s8(d4)

        p3_td = self.p3_td(
            torch.cat(
                [
                    F.interpolate(p4_native, scale_factor=2.0, mode="bilinear", align_corners=False),
                    d8,
                ],
                dim=1,
            )
        )
        p2_td = self.p2_td(
            torch.cat(
                [
                    F.interpolate(p3_td, scale_factor=2.0, mode="bilinear", align_corners=False),
                    self.detail_to_p2(d4),
                ],
                dim=1,
            )
        )

        p3_out = self.p3_out(torch.cat([self.p2_to_p3(p2_td), p3_td], dim=1))
        p4_out = self.p4_out(torch.cat([self.p3_to_p4(p3_out), p4_native], dim=1))
        p5_out = self.p5_out(self.p4_to_p5(p4_out))
        return [p2_td, p3_out, p4_out, p5_out]
