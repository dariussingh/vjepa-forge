from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config)
    for dotted_key, value in overrides.items():
        target = merged
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = value
    return merged


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    for parent_key, child_key in (
        ("model", "checkpoint"),
        ("data", "root"),
        ("eval", "checkpoint_path"),
        ("predict", "source"),
    ):
        section = config.get(parent_key)
        if isinstance(section, dict) and section.get(child_key):
            value = Path(str(section[child_key])).expanduser()
            if not value.is_absolute():
                section[child_key] = str((repo_root / value).resolve())
    return config


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping config at {config_path}")
    config["_config_path"] = str(config_path)
    return resolve_paths(config, config_path.parent)


def load_defaults() -> dict[str, Any]:
    return load_yaml(Path(__file__).with_name("defaults.yaml"))


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = deep_merge(load_defaults(), load_yaml(path))
    if overrides:
        config = apply_overrides(config, overrides)
    if "references" not in config:
        raise ValueError("Configs must define a references block")
    return config


def load_recipe(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return load_config(path, overrides=overrides)
