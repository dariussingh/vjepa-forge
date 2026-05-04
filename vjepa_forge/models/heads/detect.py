from __future__ import annotations

from typing import Any

from torch import nn

from vjepa_forge.heads.detection.rf_detr import ForgeRFDETRHead
from vjepa_forge.heads.detection.ultralytics_detect import UltralyticsLikeDetectHead


class ForgeDetectHead(nn.Module):
    """Forge-facing detection head adapter that selects the configured dense-task strategy."""

    def __init__(self, input_dim: int, num_classes: int, media: str, model_cfg: dict[str, Any] | None = None, num_queries: int = 100) -> None:
        super().__init__()
        model_cfg = {} if model_cfg is None else dict(model_cfg)
        detector_cfg = dict(model_cfg.get("detector", {}))
        strategy = str(detector_cfg.get("type", model_cfg.get("strategy", "rf_detr"))).lower()
        hidden_dim = int(detector_cfg.get("hidden_dim", model_cfg.get("head", {}).get("hidden_dim", 256)))
        self.strategy = strategy
        if strategy == "ultralytics":
            self.impl = UltralyticsLikeDetectHead(
                input_dim=input_dim,
                num_classes=num_classes,
                media=media,
                hidden_dim=hidden_dim,
                reg_max=int(detector_cfg.get("reg_max", 16)),
            )
        elif strategy == "rf_detr":
            self.impl = ForgeRFDETRHead(
                input_dim=input_dim,
                num_classes=num_classes,
                media=media,
                hidden_dim=hidden_dim,
                num_queries=int(detector_cfg.get("num_queries", model_cfg.get("head", {}).get("num_queries", num_queries))),
                num_heads=int(detector_cfg.get("num_heads", 8)),
                num_decoder_layers=int(detector_cfg.get("num_decoder_layers", 6)),
            )
        else:
            raise ValueError(f"Unsupported detector strategy: {strategy}")

    def forward(self, features):
        return self.impl(features)

    def compute_detection_loss(self, outputs: dict[str, Any], labels: dict[str, Any]) -> tuple[Any, dict[str, float]]:
        return self.impl.compute_loss(outputs, labels)

    def decode_predictions(self, outputs: dict[str, Any], **kwargs):
        return self.impl.decode_predictions(outputs, **kwargs)
