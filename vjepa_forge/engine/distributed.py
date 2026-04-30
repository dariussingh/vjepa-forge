from __future__ import annotations

from typing import Any


def normalize_distributed_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    merged.setdefault("backend", "torchrun")
    merged.setdefault("strategy", "ddp")
    merged.setdefault("precision", "fp32")
    return merged
