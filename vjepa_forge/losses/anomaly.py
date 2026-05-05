from __future__ import annotations

"""Forge-native predictive latent losses for anomaly training aligned with V-JEPA-style future prediction."""

from typing import Any

import torch
import torch.nn.functional as F

from vjepa_forge.heads.anomaly.modeling import ExtractedFeatures


def anomaly_future_prediction_loss(
    predictor: torch.nn.Module,
    past_features: ExtractedFeatures,
    future_features: ExtractedFeatures,
    model_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    predictor_type = str(model_cfg.get("predictor_type", "global_mlp"))
    loss_cfg = dict(model_cfg.get("loss", {}))
    reg_type = str(loss_cfg.get("regression", "smooth_l1")).lower()
    lambda_cosine = float(loss_cfg.get("lambda_cosine", 0.1))
    if predictor_type == "global_mlp":
        predicted = predictor(past_features.pooled)
        target = future_features.pooled.detach()
    elif predictor_type == "vit_patch":
        predicted = predictor(past_features.tokens)
        target = future_features.tokens.detach()
    else:
        raise ValueError(f"Unsupported predictor_type: {predictor_type}")
    if reg_type == "mse":
        regression = F.mse_loss(predicted, target)
    else:
        regression = F.smooth_l1_loss(predicted, target)
    cosine = 1.0 - F.cosine_similarity(
        predicted.reshape(predicted.shape[0], -1),
        target.reshape(target.shape[0], -1),
        dim=-1,
    ).mean()
    total = regression + lambda_cosine * cosine
    return total, {
        "loss_reg": float(regression.detach().cpu().item()),
        "loss_cosine": float(cosine.detach().cpu().item()),
        "loss_total": float(total.detach().cpu().item()),
    }
