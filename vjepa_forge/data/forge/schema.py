from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class ForgeAnnotation:
    op: str
    payload: dict[str, Any]


@dataclass
class ForgeRecord:
    media_path: str
    label_path: str
    media: Literal["image", "video"]
    task: Literal["classify", "detect", "segment", "anomaly"]
    annotations: list[ForgeAnnotation] = field(default_factory=list)


@dataclass
class ForgeSplit:
    name: str
    records: list[ForgeRecord]
    root: Path
