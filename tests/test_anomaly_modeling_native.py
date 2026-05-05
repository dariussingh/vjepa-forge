from __future__ import annotations

from pathlib import Path

import pytest
import torch

import vjepa_forge.heads.anomaly.modeling as anomaly_modeling


class _StubEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = dict(state_dict)
        self.loaded_strict = strict
        return object()


class _StubPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = dict(state_dict)
        self.loaded_strict = strict
        return object()


@pytest.mark.parametrize(
    ("model_name", "factory_name"),
    [
        ("vjepa2_1_vit_base_384", "base"),
        ("vjepa2_1_vit_large_384", "large"),
        ("vjepa2_1_vit_giant_384", "giant"),
        ("vjepa2_1_vit_gigantic_384", "gigantic"),
    ],
)
def test_build_native_components_supports_all_vjepa21_variants(monkeypatch, model_name: str, factory_name: str):
    called: dict[str, object] = {}

    def _factory(*, pretrained: bool, num_frames: int, **kwargs):
        called["factory"] = factory_name
        called["pretrained"] = pretrained
        called["num_frames"] = num_frames
        return _StubEncoder(), _StubPredictor()

    monkeypatch.setattr(anomaly_modeling, f"vjepa2_1_vit_{factory_name}_384", _factory)
    anomaly_modeling._VJEPA21_MODEL_FACTORIES[model_name] = _factory
    monkeypatch.setattr(
        anomaly_modeling,
        "robust_checkpoint_loader",
        lambda path, map_location="cpu": {"ema_encoder": {"enc": torch.tensor(1.0)}, "predictor": {"pred": torch.tensor(2.0)}},
    )

    feature_extractor, predictor = anomaly_modeling.build_native_components(
        model_name=model_name,
        checkpoint_path=Path("dummy.pt"),
        checkpoint_key="ema_encoder",
        predictor_checkpoint_key="predictor",
        past_frames=4,
        future_frames=4,
        image_size=32,
        device=torch.device("cpu"),
    )

    assert called == {"factory": factory_name, "pretrained": False, "num_frames": 8}
    assert feature_extractor.embed_dim == 8
    assert feature_extractor.context_token_count == 8
    assert feature_extractor.target_token_count == 8
    assert all(not parameter.requires_grad for parameter in feature_extractor.encoder.parameters())
    assert any(parameter.requires_grad for parameter in predictor.predictor.parameters())
    assert all(not parameter.requires_grad for parameter in predictor.target_projector.parameters())


def test_build_native_components_rejects_unknown_vjepa21_model():
    with pytest.raises(ValueError, match="Unsupported V-JEPA 2.1 model_name"):
        anomaly_modeling.build_native_components(
            model_name="unknown_model",
            checkpoint_path=Path("dummy.pt"),
            checkpoint_key="ema_encoder",
            predictor_checkpoint_key="predictor",
            past_frames=4,
            future_frames=4,
            image_size=32,
            device=torch.device("cpu"),
        )

