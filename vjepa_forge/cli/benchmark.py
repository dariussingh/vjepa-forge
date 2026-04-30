from __future__ import annotations

from time import perf_counter

import torch

from vjepa_forge.cli.common import parse_recipe_args
from vjepa_forge.engine.trainer import build_model


def main() -> None:
    _, _, config = parse_recipe_args("Benchmark a vjepa-forge model.")
    model = build_model(config).eval()
    image_size = int(config["data"].get("image_size", 384))
    batch = torch.randn(1, 3, image_size, image_size)
    start = perf_counter()
    with torch.no_grad():
        _ = model(batch)
    elapsed = perf_counter() - start
    print({"task": config["task"], "seconds": elapsed})
