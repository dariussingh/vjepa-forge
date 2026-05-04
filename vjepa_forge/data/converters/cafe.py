from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
from pathlib import Path
import subprocess
from typing import Any

import yaml

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class CafeConversionResult:
    dataset_yaml: str
    train_clips: int
    val_clips: int
    test_clips: int


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_source_video(source_root: Path, item: dict[str, Any]) -> Path:
    candidates = [
        source_root / "normal" / item["source_video"],
        source_root / "anomaly" / item["source_video"],
        source_root / item["source_video"],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to find source video for clip {item['clip_name']}: {item['source_video']}")


def _write_trimmed_clip(source: Path, dest: Path, item: dict[str, Any]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Cafe conversion requires ffmpeg to write trimmed .mp4 clips")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    fps = float(item.get("fps", 25.0))
    frame_count = max(1, int(item.get("frame_count") or (int(item["frame_end"]) - int(item["frame_start"]))))
    start_s = float(item.get("start_s", 0.0))
    duration_s = frame_count / fps
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_s:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while trimming {item['clip_name']} from {source}: {result.stderr.strip()}"
        )


def _derive_interval(item: dict[str, Any], frame_labels: dict[str, list[int]]) -> tuple[int, int] | None:
    start = item.get("label_frame_start")
    end = item.get("label_frame_end")
    if start is not None and end is not None:
        return int(start) - int(item["frame_start"]), int(end) - int(item["frame_start"])
    labels = frame_labels.get(item["clip_name"])
    if not labels:
        return None
    positive = [idx for idx, value in enumerate(labels) if int(value) == 1]
    if not positive:
        return None
    return min(positive), max(positive)


def _write_label(dest: Path, item: dict[str, Any], frame_labels: dict[str, list[int]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if item["label"] == "normal":
        dest.write_text("ano normal\n", encoding="utf-8")
        return
    interval = _derive_interval(item, frame_labels)
    if interval is None:
        raise ValueError(f"Missing anomaly interval for clip {item['clip_name']}")
    dest.write_text(f"ano abnormal {interval[0]} {interval[1]} 0\n", encoding="utf-8")


def _split_name(item: dict[str, Any]) -> str:
    return "train" if item["split"] == "Train" else "test"


def convert_cafe_to_forge(source: str | Path, out: str | Path) -> CafeConversionResult:
    source_root = Path(source).expanduser().resolve()
    out_root = Path(out).expanduser().resolve()
    manifest = _load_json(source_root / "processed" / "clip_manifest.json")
    frame_labels = _load_json(source_root / "processed" / "frame_labels.json")

    split_entries: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    for item in _progress(manifest, desc="convert cafe", total=len(manifest)):
        split = _split_name(item)
        source_video = _resolve_source_video(source_root, item)
        video_rel = Path("videos") / split / f"{item['clip_name']}.mp4"
        label_rel = Path("labels") / split / f"{item['clip_name']}.txt"
        _write_trimmed_clip(source_video, out_root / video_rel, item)
        _write_label(out_root / label_rel, item, frame_labels)
        split_entries[split].append(video_rel.as_posix())
        counts[split] += 1
        if split == "test":
            split_entries["val"].append(video_rel.as_posix())
            counts["val"] += 1

    splits_root = out_root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    for split_name, entries in split_entries.items():
        (splits_root / f"{split_name}.txt").write_text(
            "\n".join(entries) + ("\n" if entries else ""),
            encoding="utf-8",
        )

    dataset_yaml = out_root / "forge.yaml"
    payload = {
        "path": str(out_root),
        "task": "anomaly",
        "media": "video",
        "names": {0: "anomaly"},
        "splits": {
            "train": "splits/train.txt",
            "val": "splits/val.txt",
            "test": "splits/test.txt",
        },
        "labels": {"format": "forge-yolo", "root": "labels"},
    }
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return CafeConversionResult(
        dataset_yaml=str(dataset_yaml),
        train_clips=counts["train"],
        val_clips=counts["val"],
        test_clips=counts["test"],
    )
