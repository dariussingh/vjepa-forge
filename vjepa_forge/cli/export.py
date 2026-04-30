from __future__ import annotations

import torch

from vjepa_forge.cli.common import parse_recipe_args
from vjepa_forge.engine.trainer import build_model
from vjepa_forge.export import export_to_onnx


def main() -> None:
    _, _, config = parse_recipe_args("Export a vjepa-forge model.")
    if config["task"] == "detection" and config.get("input_type", "image") == "video":
        raise SystemExit("Video detection ONNX export is not supported yet. Use backend=torch for temporal detection inference.")
    model = build_model(config)
    input_type = config.get("input_type", "image")
    image_size = int(config["data"].get("image_size", 384))
    num_frames = int(config["data"].get("num_frames", 8))
    sample = torch.randn(1, 3, image_size, image_size)
    if input_type == "video":
        sample = torch.randn(1, 3, num_frames, image_size, image_size)
    output_path = config.get("export", {}).get("output_path") or f"{config['task']}.onnx"
    print(
        export_to_onnx(
            model,
            sample,
            output_path,
            opset=int(config.get("export", {}).get("opset", 17)),
            dynamic_axes=bool(config.get("export", {}).get("dynamic_axes", True)),
        )
    )
