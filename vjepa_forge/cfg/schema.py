from __future__ import annotations

from typing import Any


KNOWN_KEYS: dict[str, set[str]] = {
    "train": {
        "epochs", "batch_size", "save", "save_period", "resume", "project", "name",
        "exist_ok", "lr", "lr_mode", "lr_scale_rule", "reference_batch_size",
        "reference_lr", "start_lr", "final_lr", "weight_decay", "final_weight_decay",
        "warmup", "num_workers", "prefetch_factor",
        "persistent_workers", "pin_memory", "reader_cache_size", "device", "seed",
        "save_latest_every_epoch", "save_epoch_checkpoints", "scheduler",
        "early_stopping", "stages",
    },
    "val": {
        "batch_size", "num_workers", "prefetch_factor", "persistent_workers",
        "pin_memory", "reader_cache_size", "split", "threshold_std_multiplier",
        "smoothing_window", "checkpoint_target", "checkpoint_path", "predictor_source",
    },
    "data": {
        "path", "_path", "task", "media", "image_size", "image_backend",
        "video_backend", "train_fraction", "train_seed", "past_frames", "future_frames",
        "stride", "augment", "dataset_yaml", "config", "clip_len", "clip_stride",
        "num_frames", "num_classes", "root", "names", "splits", "labels", "masks",
    },
    "predict": {
        "source", "batch_size", "num_workers", "split", "threshold", "visualize",
        "output_dir",
    },
    "export": {
        "format", "output_path", "opset", "dynamic_axes", "checkpoint_target",
        "checkpoint_path", "predictor_source",
    },
    "distributed": {
        "backend", "strategy", "precision", "sync_batchnorm", "compile",
        "compile_mode", "tf32", "channels_last", "ddp_eval",
    },
    "output": {"root"},
}


def _fuzzy_match(key: str, known: set[str], n: int = 3) -> list[str]:
    """Return up to `n` close matches by edit-distance (no external deps)."""
    def _edit(a: str, b: str) -> int:
        if a == b:
            return 0
        m, n_ = len(a), len(b)
        dp = list(range(n_ + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n_ + 1):
                prev, dp[j] = dp[j], prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
        return dp[n_]

    ranked = sorted(known, key=lambda candidate: _edit(key, candidate))
    return [c for c in ranked[:n] if _edit(key, c) <= max(2, len(key) // 3)]


def validate_config(config: dict[str, Any]) -> None:
    """Raise ValueError with an actionable message on unknown top-level config section keys."""
    for section, known in KNOWN_KEYS.items():
        value = config.get(section)
        if not isinstance(value, dict):
            continue
        for key in value:
            if key not in known:
                suggestions = _fuzzy_match(key, known)
                hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
                raise ValueError(
                    f"Unknown config key '{section}.{key}'.{hint} "
                    f"Valid keys for '{section}': {sorted(known)}"
                )
