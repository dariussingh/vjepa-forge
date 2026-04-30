from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping")

    defaults = {
        "dataset": {
            "name": "ped2",
            "root": "data/ucsd_ped2",
            "split_seed": 0,
            "val_ratio": 0.25,
            "category": "UCSDped2",
            "image_size": 384,
            "past_frames": 8,
            "future_frames": 8,
            "stride": 1,
        },
        "model": {
            "name": "vjepa2_1_vit_base_384",
            "checkpoint": "weights/vjepa2_1_vitb_dist_vitG_384.pt",
            "checkpoint_key": "ema_encoder",
            "predictor_type": "global_mlp",
            "hidden_dim": 1024,
            "dropout": 0.1,
            "predictor_embed_dim": 768,
            "predictor_depth": 12,
            "predictor_num_heads": 12,
            "predictor_use_rope": True,
            "token_aggregation": "topk_mean",
            "token_topk_fraction": 0.1,
        },
        "train": {
            "batch_size": 4,
            "epochs": 10,
            "lr_mode": "manual",
            "lr": 1.0e-4,
            "reference_batch_size": 4,
            "reference_lr": 1.0e-4,
            "lr_scale_rule": "sqrt",
            "weight_decay": 1.0e-4,
            "num_workers": 2,
            "device": "cuda",
            "seed": 0,
            "save_latest_every_epoch": True,
            "save_epoch_checkpoints": False,
        },
        "eval": {
            "batch_size": 4,
            "num_workers": 2,
            "threshold_std_multiplier": 3.0,
            "smoothing_window": 9,
            "checkpoint_target": "best",
            "checkpoint_path": None,
        },
        "output": {
            "root": "outputs/vjepa-anomaly/ucsd_ped2_vitb",
        },
    }
    cfg = _deep_update(defaults, cfg)
    cfg["config_path"] = str(path)
    return cfg


def resolve_path(path_str: str | Path, repo_root: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(repo_root) / path).resolve()
