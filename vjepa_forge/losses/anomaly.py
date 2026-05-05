from __future__ import annotations

"""Forge-native predictive latent losses for anomaly training aligned with V-JEPA-style future prediction."""

from typing import Any

import torch

from vjepa_forge.heads.anomaly.modeling import NativeExtractedFeatures


def anomaly_future_prediction_loss(
    predictor: torch.nn.Module,
    past_features: NativeExtractedFeatures,
    future_features: NativeExtractedFeatures,
    model_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    predictor_type = str(model_cfg.get("predictor_type", "vjepa_native"))
    if predictor_type != "vjepa_native":
        raise ValueError(f"Unsupported predictor_type: {predictor_type}")
    loss_cfg = dict(model_cfg.get("loss", {}))
    loss_exp = float(loss_cfg.get("loss_exp", 1.0))
    predicted = predictor(past_features.context_tokens)
    target = predictor.project_targets(past_features.target_tokens).detach()
    loss = torch.mean(torch.abs(predicted - target) ** loss_exp) / loss_exp
    return loss, {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_exp": float(loss_exp),
    }
