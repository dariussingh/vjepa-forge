from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from app.vjepa_2_1.models.utils.modules import Block
from src.hub.backbones import vjepa2_1_vit_base_384
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.tensors import trunc_normal_


def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


@dataclass(frozen=True)
class ExtractedFeatures:
    pooled: torch.Tensor
    tokens: torch.Tensor


class FuturePredictorHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialViTPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        predictor_embed_dim: int,
        past_steps: int,
        future_steps: int,
        grid_size: int,
        depth: int,
        num_heads: int,
        dropout: float,
        use_rope: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.grid_size = grid_size
        self.predictor_embed_dim = predictor_embed_dim
        self.use_rope = use_rope

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim * past_steps),
            nn.Linear(input_dim * past_steps, predictor_embed_dim),
        )
        self.pos_embed = None
        if not use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, predictor_embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=grid_size,
                    grid_depth=1,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=dropout,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    patch_size=16,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.output_proj = nn.Linear(
            predictor_embed_dim,
            future_steps * input_dim,
        )

    def forward(self, past_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, past_steps, num_spatial_tokens, embed_dim = past_tokens.shape
        if past_steps != self.past_steps:
            raise ValueError(f"Expected {self.past_steps} past steps, got {past_steps}")

        x = past_tokens.permute(0, 2, 1, 3).reshape(batch_size, num_spatial_tokens, past_steps * embed_dim)
        x = self.input_proj(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        x = self.output_proj(x)
        x = x.view(batch_size, num_spatial_tokens, self.future_steps, embed_dim)
        return x.permute(0, 2, 1, 3).contiguous()


class FeatureExtractor(nn.Module):
    def __init__(self, encoder: nn.Module, image_size: int, num_frames: int, tubelet_size: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.grid_depth = num_frames // tubelet_size
        self.grid_size = image_size // 16
        self.num_spatial_tokens = self.grid_size * self.grid_size
        self.tubelet_size = tubelet_size

    def forward(self, clip: torch.Tensor) -> ExtractedFeatures:
        tokens = self.encoder(clip)
        pooled = tokens.mean(dim=1)
        reshaped = tokens.view(tokens.size(0), self.grid_depth, self.num_spatial_tokens, self.embed_dim)
        return ExtractedFeatures(pooled=pooled, tokens=reshaped)


def build_feature_extractor(
    model_name: str,
    checkpoint_path: str | Path,
    checkpoint_key: str,
    num_frames: int,
    image_size: int,
    device: torch.device,
) -> FeatureExtractor:
    if model_name != "vjepa2_1_vit_base_384":
        raise ValueError(f"Unsupported model_name for v1: {model_name}")

    encoder, _ = vjepa2_1_vit_base_384(pretrained=False, num_frames=num_frames)
    checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location="cpu")
    cleaned_state = _clean_backbone_key(checkpoint[checkpoint_key])
    encoder.load_state_dict(cleaned_state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    feature_extractor = FeatureExtractor(encoder, image_size=image_size, num_frames=num_frames).to(device)
    feature_extractor.eval()
    return feature_extractor


def build_predictor(model_cfg: dict[str, object], feature_extractor: FeatureExtractor) -> nn.Module:
    predictor_type = str(model_cfg.get("predictor_type", "global_mlp"))
    if predictor_type == "global_mlp":
        return FuturePredictorHead(
            input_dim=feature_extractor.embed_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        )
    if predictor_type == "vit_patch":
        return SpatialViTPredictor(
            input_dim=feature_extractor.embed_dim,
            predictor_embed_dim=int(model_cfg.get("predictor_embed_dim", feature_extractor.embed_dim)),
            past_steps=feature_extractor.grid_depth,
            future_steps=feature_extractor.grid_depth,
            grid_size=feature_extractor.grid_size,
            depth=int(model_cfg.get("predictor_depth", 12)),
            num_heads=int(model_cfg.get("predictor_num_heads", 12)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_rope=bool(model_cfg.get("predictor_use_rope", True)),
        )
    raise ValueError(f"Unsupported predictor_type: {predictor_type}")
