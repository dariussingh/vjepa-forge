from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = base if base is not None else _repo_root()
    return (root / path).resolve()


def default_run_name(data: str | Path | None, *, fallback: str = "exp") -> str:
    if data is None:
        return fallback
    path = Path(data)
    if path.suffix == ".yaml":
        return path.parent.name or path.stem or fallback
    return path.stem or path.name or fallback


def increment_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 2
    while True:
        candidate = path.parent / f"{path.name}{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def resolve_run_dir(
    *,
    task: str,
    data: str | Path | None,
    project: str | Path | None,
    name: str | None,
    exist_ok: bool,
    resume: bool | str,
) -> Path:
    project_root = _resolve_path(project) or (_repo_root() / "outputs" / "vjepa-forge")
    run_name = name or default_run_name(data, fallback=task)
    run_dir = (project_root / task / run_name).resolve()
    if resume or exist_ok:
        return run_dir
    return increment_path(run_dir)


def results_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=csv_path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    os.replace(tmp_path, csv_path)


def checkpoint_payload(
    *,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any] | None,
    scheduler_state: dict[str, Any] | None,
    epoch: int,
    global_step: int,
    best_fitness: float,
    metrics: dict[str, Any],
    config: dict[str, Any],
    task: str,
    media: str,
    checkpoint_kind: str,
    component: str = "model",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_fitness": float(best_fitness),
        "metrics": dict(metrics),
        "config": config,
        "task": str(task),
        "media": str(media),
        "checkpoint_kind": str(checkpoint_kind),
        "component": str(component),
    }
    if extras:
        payload["extras"] = dict(extras)
    return payload


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def resolve_resume_path(resume: bool | str | Path | None, *, run_dir: Path) -> Path | None:
    if not resume:
        return None
    if resume is True:
        return run_dir / "weights" / "last.pt"
    return _resolve_path(resume)


@dataclass(frozen=True)
class CheckpointPaths:
    run_dir: Path
    weights_dir: Path
    last: Path
    best: Path
    results_csv: Path


def checkpoint_paths(run_dir: str | Path) -> CheckpointPaths:
    root = Path(run_dir).resolve()
    weights_dir = root / "weights"
    return CheckpointPaths(
        run_dir=root,
        weights_dir=weights_dir,
        last=weights_dir / "last.pt",
        best=weights_dir / "best.pt",
        results_csv=root / "results.csv",
    )
