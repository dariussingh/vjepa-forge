from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, target)


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
