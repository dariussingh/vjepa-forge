from __future__ import annotations

from time import perf_counter

import torch

from vjepa_forge.cli.common import parse_config_args
from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.engine.model import ForgeModel
from vjepa_forge.engine.runtime import setup_runtime


def _build_batch(config: dict) -> ForgeBatch:
    task = str(config["task"])
    media = str(config["media"])
    image_size = int(config["data"].get("image_size", 384))
    num_frames = int(config["data"].get("num_frames", config["data"].get("past_frames", 8)))
    batch_size = int(config.get("benchmark", {}).get("batch_size", 1))
    if media == "video":
        tensor = torch.randn(batch_size, 3, num_frames, image_size, image_size)
    else:
        tensor = torch.randn(batch_size, 3, image_size, image_size)
    return ForgeBatch(x=tensor, media=media, task=task, labels={}, paths=[], meta=[])


def main() -> None:
    _, _, config = parse_config_args("Benchmark a vjepa-forge model.")
    model_cfg = dict(config["model"])
    model_cfg["task"] = config["task"]
    model_cfg["media"] = config["media"]
    data_cfg = dict(config["data"])
    data_cfg["distributed"] = dict(config.get("distributed", {}))
    runtime = setup_runtime(device=str(config["train"].get("device", "cpu")), data_cfg=data_cfg)
    model = ForgeModel(model_cfg, data=data_cfg)
    model = runtime.prepare_module(model.eval(), training=False)
    batch = _build_batch(config)
    batch.x = runtime.move_tensor(batch.x)
    warmup_iters = int(config.get("benchmark", {}).get("warmup_iters", 5))
    measure_iters = int(config.get("benchmark", {}).get("measure_iters", 20))
    with runtime.inference_context():
        for _ in range(warmup_iters):
            with runtime.autocast_context():
                _ = model(batch)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        start = perf_counter()
        for _ in range(measure_iters):
            with runtime.autocast_context():
                _ = model(batch)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
    elapsed = perf_counter() - start
    print(
        {
            "task": config["task"],
            "media": config["media"],
            "seconds_total": elapsed,
            "seconds_per_iter": elapsed / max(measure_iters, 1),
            "items_per_second": (int(batch.x.shape[0]) * measure_iters) / max(elapsed, 1.0e-9),
        }
    )
