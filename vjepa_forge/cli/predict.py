from __future__ import annotations

import torch

from vjepa_forge.cli.common import parse_config_args
from vjepa_forge.engine.inference import run_onnx_inference, run_torch_inference
from vjepa_forge.engine.trainer import build_model


def main() -> None:
    _, _, config = parse_config_args("Run prediction with vjepa-forge.")
    backend = config.get("inference", {}).get("backend", "torch")
    media_type = config.get("media", "image")
    image_size = int(config["data"].get("image_size", 384))
    num_frames = int(config["data"].get("num_frames", 8))
    if media_type == "video":
        batch = torch.randn(1, 3, num_frames, image_size, image_size)
    else:
        batch = torch.randn(1, 3, image_size, image_size)
    if backend == "onnx":
        onnx_path = config.get("export", {}).get("output_path") or "model.onnx"
        outputs = run_onnx_inference(onnx_path, batch, providers=config.get("inference", {}).get("providers"))
    else:
        model = build_model(config)
        outputs = run_torch_inference(model, batch)
    if isinstance(outputs, dict):
        summary = {key: list(value.shape) if hasattr(value, "shape") else type(value).__name__ for key, value in outputs.items()}
        print(summary)
        return
    if hasattr(outputs, "shape"):
        print(list(outputs.shape))
        return
    print(type(outputs).__name__)
