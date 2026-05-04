from __future__ import annotations

from pathlib import Path


VALID_TASKS = {"classify", "detect", "segment", "anomaly"}
VALID_MEDIA = {"image", "video"}


def validate_dataset_config(config: dict) -> None:
    task = config.get("task")
    media = config.get("media")
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if media not in VALID_MEDIA:
        raise ValueError(f"Unsupported media: {media}")
    if "path" not in config:
        raise ValueError("Dataset config must define path")
    if "splits" not in config or not isinstance(config["splits"], dict):
        raise ValueError("Dataset config must define splits")


def validate_task_media(task: str, media: str) -> None:
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if media not in VALID_MEDIA:
        raise ValueError(f"Unsupported media: {media}")


def resolve_label_path(dataset_root: Path, config: dict, media_rel_path: str) -> Path:
    labels_root = Path(config.get("labels", {}).get("root", "labels"))
    media_path = Path(media_rel_path)
    split_name = media_path.parts[1] if len(media_path.parts) > 1 else ""
    stem = media_path.with_suffix(".txt").name
    return dataset_root / labels_root / split_name / stem
