# ------------------------------------------------------------------------
# RF-DETR + V-JEPA adapter
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch

from rfdetr.models.backbone.base import BackboneBase
from vjepa_tune.backbone import BACKBONE_SPECS, VJEPAImageBackbone


def _vjepa_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    layer_id = num_layers + 1
    if ".blocks." in name:
        layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    elif ".patch_embed" in name or ".pos_embed" in name:
        layer_id = 0
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _vjepa_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    if any(token in name for token in ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embed")):
        return 0.0
    return weight_decay_rate


class VJEPA2Encoder(BackboneBase):
    def __init__(
        self,
        *,
        name: str,
        target_shape: tuple[int, int],
        patch_size: int,
        out_feature_indexes: list[int],
        freeze_encoder: bool,
        checkpoint: str | None = None,
        checkpoint_key: str = "ema_encoder",
    ) -> None:
        super().__init__()
        if name not in {"vjepa2_windowed_base", "vjepa2_base", "vjepa2_vitb"}:
            raise ValueError(f"Unsupported V-JEPA RF-DETR encoder '{name}'")

        self.encoder = VJEPAImageBackbone(
            name="vit_base",
            checkpoint=checkpoint,
            checkpoint_key=checkpoint_key,
            mode="image",
            imgsz=max(target_shape),
            patch_size=patch_size,
            out_layers=out_feature_indexes,
        )
        self._out_feature_channels = [BACKBONE_SPECS["vit_base"]["embed_dim"]] * len(out_feature_indexes)
        self.out_feature_indexes = list(out_feature_indexes)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, tensors: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(tensors)

    def get_named_param_lr_pairs(self, args: Any, prefix: str = "backbone.0") -> dict[str, dict[str, Any]]:
        num_layers = max(self.out_feature_indexes) + 1 if self.out_feature_indexes else 12
        named_param_lr_pairs: dict[str, dict[str, Any]] = {}
        for name, parameter in self.named_parameters():
            full_name = prefix + "." + name
            if not parameter.requires_grad:
                continue
            lr = (
                args.lr_encoder
                * _vjepa_lr_decay_rate(full_name, lr_decay_rate=args.lr_vit_layer_decay, num_layers=num_layers)
                * args.lr_component_decay**2
            )
            wd = args.weight_decay * _vjepa_weight_decay_rate(full_name)
            named_param_lr_pairs[full_name] = {
                "params": parameter,
                "lr": lr,
                "weight_decay": wd,
            }
        return named_param_lr_pairs
