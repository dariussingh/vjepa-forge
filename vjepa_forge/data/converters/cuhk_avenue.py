from __future__ import annotations

from dataclasses import dataclass
import json
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
class CukhAvenueConversionResult:
    dataset_yaml: str
    train_clips: int
    val_clips: int
    test_clips: int

    def __str__(self) -> str:
        return (
            f"CukhAvenueConversionResult("
            f"dataset_yaml={self.dataset_yaml!r}, "
            f"train_clips={self.train_clips}, "
            f"val_clips={self.val_clips}, "
            f"test_clips={self.test_clips})"
        )


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _convert_full_video(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("CUHK Avenue conversion requires ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {src}: {result.stderr.strip()}")


def _contiguous_runs(labels: list[int]) -> list[tuple[int, int]]:
    """Find all contiguous runs of 1s in a binary label list."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_run:
            in_run = True
            start = i
        elif v == 0 and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(labels) - 1))
    return runs


def _write_cuhk_label(dest: Path, vid_id: str, frame_labels: dict[str, list[int]], is_test: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not is_test:
        dest.write_text("ano normal\n", encoding="utf-8")
        return
    labels = frame_labels.get(vid_id, [])
    runs = _contiguous_runs(labels)
    if not runs:
        dest.write_text("ano normal\n", encoding="utf-8")
        return
    lines = [f"ano abnormal {start} {end} 0\n" for start, end in runs]
    dest.write_text("".join(lines), encoding="utf-8")


def convert_cuhk_avenue_to_forge(source: str | Path, out: str | Path) -> CukhAvenueConversionResult:
    source_root = Path(source).expanduser().resolve()
    out_root = Path(out).expanduser().resolve()

    raw = source_root / "raw" / "Avenue Dataset"
    frame_labels: dict[str, list[int]] = _load_json(source_root / "processed" / "frame_labels.json")

    split_entries: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split, video_dir, is_test in [
        ("train", raw / "training_videos", False),
        ("test", raw / "testing_videos", True),
    ]:
        videos = sorted(video_dir.glob("*.avi"))
        for video in _progress(videos, desc=f"convert cuhk_avenue:{split}", total=len(videos)):
            vid_id = video.stem  # "01", "02", ...
            video_rel = Path("videos") / split / f"{vid_id}.mp4"
            label_rel = Path("labels") / split / f"{vid_id}.txt"
            _convert_full_video(video, out_root / video_rel)
            _write_cuhk_label(out_root / label_rel, vid_id, frame_labels, is_test=is_test)
            split_entries[split].append(video_rel.as_posix())
            counts[split] += 1
            if is_test:
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
    return CukhAvenueConversionResult(
        dataset_yaml=str(dataset_yaml),
        train_clips=counts["train"],
        val_clips=counts["val"],
        test_clips=counts["test"],
    )
