from __future__ import annotations

from torch import nn

from .heads import ForgeAnomalyHead, ForgeClassificationHead, ForgeDetectHead, ForgeSegmentHead
from .vjepa.model import VJEPA21Backbone


def build_backbone(model_cfg: dict, data_cfg: dict) -> VJEPA21Backbone:
    merged = {
        "backbone": dict(model_cfg.get("backbone", {})),
        "image_size": int(model_cfg.get("image_size", data_cfg.get("image_size", 64))),
        "num_frames": int(model_cfg.get("num_frames", data_cfg.get("clip_len", data_cfg.get("num_frames", 8)))),
    }
    return VJEPA21Backbone(merged)


def build_head(task: str, media: str, model_cfg: dict, embed_dim: int) -> nn.Module:
    num_classes = int(model_cfg.get("num_classes", 2))
    head_cfg = dict(model_cfg.get("head", {}))
    if task == "classify":
        return ForgeClassificationHead(embed_dim, num_classes, media)
    if task == "detect":
        return ForgeDetectHead(embed_dim, num_classes, media, num_queries=int(head_cfg.get("num_queries", 100)))
    if task == "segment":
        return ForgeSegmentHead(embed_dim, num_classes, media)
    if task == "anomaly":
        return ForgeAnomalyHead(embed_dim, media)
    raise ValueError(f"Unsupported task: {task}")
