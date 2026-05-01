from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from vjepa_forge.export import export_to_onnx


@dataclass
class ExportResult:
    format: str
    output_path: str


class _ForwardExportModule(nn.Module):
    def __init__(self, forge_model) -> None:
        super().__init__()
        self.forge_model = forge_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.forge_model.media == "image":
            feats = self.forge_model.backbone.forward_image(x)
        else:
            feats = self.forge_model.backbone.forward_video(x)
        outputs = self.forge_model.head(feats)
        if isinstance(outputs, dict):
            raise ValueError("Dict outputs are not supported by the active ONNX export path")
        return outputs


class Exporter:
    def export(
        self,
        model,
        *,
        format: str = "onnx",
        output_path: str | Path = "model.onnx",
        sample: torch.Tensor,
        opset: int = 17,
        dynamic_axes: bool = True,
    ) -> ExportResult:
        if format != "onnx":
            raise ValueError(f"Unsupported export format: {format}")
        wrapper = _ForwardExportModule(model)
        path = export_to_onnx(wrapper, sample, output_path, opset=opset, dynamic_axes=dynamic_axes)
        return ExportResult(format=format, output_path=path)
