from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from vjepa_forge.backbones.factory import (
    vjepa2_1_vit_base_384,
    vjepa2_1_vit_giant_384,
    vjepa2_1_vit_gigantic_384,
    vjepa2_1_vit_large_384,
)
from vjepa_forge.utils.checkpoint_loader import robust_checkpoint_loader


def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


@dataclass(frozen=True)
class NativeExtractedFeatures:
    context_tokens: torch.Tensor
    target_tokens: torch.Tensor


class NativeFeatureExtractor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        image_size: int,
        past_frames: int,
        future_frames: int,
        tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        if past_frames % tubelet_size != 0 or future_frames % tubelet_size != 0:
            raise ValueError("past_frames and future_frames must be divisible by tubelet_size for native V-JEPA anomaly mode")
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.grid_size = image_size // 16
        self.num_spatial_tokens = self.grid_size * self.grid_size
        self.tubelet_size = tubelet_size
        self.context_depth = past_frames // tubelet_size
        self.target_depth = future_frames // tubelet_size
        self.context_token_count = self.context_depth * self.num_spatial_tokens
        self.target_token_count = self.target_depth * self.num_spatial_tokens

    def forward(self, clip: torch.Tensor) -> NativeExtractedFeatures:
        tokens = self.encoder(clip)
        context_tokens = tokens[:, : self.context_token_count, :]
        target_tokens = tokens[:, self.context_token_count :, :]
        if target_tokens.shape[1] != self.target_token_count:
            raise ValueError(
                f"Expected {self.target_token_count} target tokens, got {target_tokens.shape[1]}"
            )
        return NativeExtractedFeatures(context_tokens=context_tokens, target_tokens=target_tokens)


class VJEPANativePredictorAdapter(nn.Module):
    def __init__(
        self,
        predictor: nn.Module,
        *,
        context_token_count: int,
        target_token_count: int,
        target_depth: int,
        num_spatial_tokens: int,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.target_projector = copy.deepcopy(predictor).eval()
        for parameter in self.target_projector.parameters():
            parameter.requires_grad = False
        self.target_depth = target_depth
        self.num_spatial_tokens = num_spatial_tokens
        self.register_buffer("context_indices", torch.arange(context_token_count, dtype=torch.long), persistent=False)
        self.register_buffer(
            "target_indices",
            torch.arange(context_token_count, context_token_count + target_token_count, dtype=torch.long),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_projector.eval()
        return self

    def _expand_indices(self, indices: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        return indices.unsqueeze(0).expand(batch_size, -1).to(device=device)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = context_tokens.size(0)
        masks_x = self._expand_indices(self.context_indices, batch_size, context_tokens.device)
        masks_y = self._expand_indices(self.target_indices, batch_size, context_tokens.device)
        predicted, _ = self.predictor(context_tokens, masks_x, masks_y, mod="video")
        return predicted.view(batch_size, self.target_depth, self.num_spatial_tokens, -1)

    @torch.no_grad()
    def project_targets(self, target_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = target_tokens.size(0)
        masks_x = self._expand_indices(self.target_indices, batch_size, target_tokens.device)
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=target_tokens.device)
        _, projected = self.target_projector(target_tokens, masks_x, empty, mod="video")
        return projected.view(batch_size, self.target_depth, self.num_spatial_tokens, -1)


_VJEPA21_MODEL_FACTORIES: dict[str, Callable[..., tuple[nn.Module, nn.Module]]] = {
    "vjepa2_1_vit_base_384": vjepa2_1_vit_base_384,
    "vjepa2_1_vit_large_384": vjepa2_1_vit_large_384,
    "vjepa2_1_vit_giant_384": vjepa2_1_vit_giant_384,
    "vjepa2_1_vit_gigantic_384": vjepa2_1_vit_gigantic_384,
}


def _resolve_vjepa21_factory(model_name: str) -> Callable[..., tuple[nn.Module, nn.Module]]:
    factory = _VJEPA21_MODEL_FACTORIES.get(model_name)
    if factory is None:
        raise ValueError(
            f"Unsupported V-JEPA 2.1 model_name '{model_name}'. "
            f"Expected one of {sorted(_VJEPA21_MODEL_FACTORIES)}"
        )
    return factory


def build_native_components(
    *,
    model_name: str,
    checkpoint_path: str | Path,
    checkpoint_key: str,
    predictor_checkpoint_key: str,
    past_frames: int,
    future_frames: int,
    image_size: int,
    device: torch.device,
) -> tuple[NativeFeatureExtractor, VJEPANativePredictorAdapter]:
    total_frames = int(past_frames) + int(future_frames)
    factory = _resolve_vjepa21_factory(model_name)
    encoder, predictor = factory(pretrained=False, num_frames=total_frames)
    checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location="cpu")
    encoder.load_state_dict(_clean_backbone_key(checkpoint[checkpoint_key]), strict=True)
    predictor.load_state_dict(_clean_backbone_key(checkpoint[predictor_checkpoint_key]), strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    feature_extractor = NativeFeatureExtractor(
        encoder,
        image_size=image_size,
        past_frames=past_frames,
        future_frames=future_frames,
    ).to(device)
    feature_extractor.eval()
    predictor_adapter = VJEPANativePredictorAdapter(
        predictor.to(device),
        context_token_count=feature_extractor.context_token_count,
        target_token_count=feature_extractor.target_token_count,
        target_depth=feature_extractor.target_depth,
        num_spatial_tokens=feature_extractor.num_spatial_tokens,
    ).to(device)
    predictor_adapter.target_projector.to(device)
    predictor_adapter.eval()
    return feature_extractor, predictor_adapter
