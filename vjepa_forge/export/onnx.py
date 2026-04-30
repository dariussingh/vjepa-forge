from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def export_to_onnx(
    model: torch.nn.Module,
    sample: torch.Tensor,
    output_path: str | Path,
    *,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> str:
    path = str(Path(output_path).resolve())
    axes = None
    if dynamic_axes:
        axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    model.eval()
    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                sample,
                path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=axes,
                opset_version=opset,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            if exc.name == "onnxscript":
                raise RuntimeError("ONNX export requires the optional dependency `onnxscript`. Install the `onnx` extra.") from exc
            raise
    return path


class ONNXInferenceSession:
    def __init__(self, path: str | Path, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for backend=onnx") from exc
        self.session = ort.InferenceSession(str(Path(path).resolve()), providers=providers or ["CPUExecutionProvider"])

    def __call__(self, batch: torch.Tensor) -> Any:
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: batch.detach().cpu().numpy().astype(np.float32)})
        if len(outputs) == 1:
            return outputs[0]
        return outputs
