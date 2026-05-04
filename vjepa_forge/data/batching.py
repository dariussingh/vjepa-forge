from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class ForgeBatch:
    x: torch.Tensor
    media: Literal["image", "video"]
    task: Literal["classify", "detect", "segment", "anomaly"]
    labels: dict
    paths: list[str]
    meta: list[dict]
