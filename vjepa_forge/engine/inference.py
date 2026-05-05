from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vjepa_forge.export.onnx import ONNXInferenceSession
from vjepa_forge.engine.runtime import setup_runtime


def run_torch_inference(model: torch.nn.Module, batch: torch.Tensor, *, device: str = "cpu", data_cfg: dict[str, Any] | None = None) -> Any:
    runtime = setup_runtime(device=device, data_cfg=data_cfg)
    model = runtime.prepare_module(model.eval(), training=False)
    batch = runtime.move_tensor(batch)
    with runtime.inference_context():
        with runtime.autocast_context():
            return model(batch)


def run_onnx_inference(path: str | Path, batch: torch.Tensor, providers: list[str] | None = None) -> Any:
    session = ONNXInferenceSession(path, providers=providers)
    return session(batch)
