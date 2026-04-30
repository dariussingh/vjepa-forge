from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vjepa_forge.export.onnx import ONNXInferenceSession


def run_torch_inference(model: torch.nn.Module, batch: torch.Tensor) -> Any:
    model.eval()
    with torch.no_grad():
        return model(batch)


def run_onnx_inference(path: str | Path, batch: torch.Tensor, providers: list[str] | None = None) -> Any:
    session = ONNXInferenceSession(path, providers=providers)
    return session(batch)
