from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


CFG_ROOT = Path(__file__).resolve().parent
RECIPE_ROOT = CFG_ROOT / "recipes"


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


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping YAML at {resolved}")
    payload["_path"] = str(resolved)
    return payload


def _resolve_cfg_path(kind: str, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix == ".yaml":
        search = CFG_ROOT / kind / candidate.name
        if search.exists():
            return search.resolve()
    else:
        search = CFG_ROOT / kind / f"{candidate.name}.yaml"
        if search.exists():
            return search.resolve()
    raise FileNotFoundError(f"Unable to resolve {kind} config: {value}")


def load_default_config() -> dict[str, Any]:
    return _load_yaml(CFG_ROOT / "default.yaml")


def load_recipe_defaults() -> dict[str, Any]:
    return _load_yaml(RECIPE_ROOT / "defaults.yaml")


def load_model_config(value: str | Path) -> dict[str, Any]:
    return _load_yaml(_resolve_cfg_path("models", value))


def load_data_config(value: str | Path) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        return _load_yaml(path)
    return _load_yaml(_resolve_cfg_path("datasets", value))


def load_runtime_config(
    *,
    task: str,
    mode: str,
    model: str | Path,
    data: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_default_config()
    model_payload = load_model_config(model)
    config["model"] = deep_merge(config.get("model", {}), {k: v for k, v in model_payload.items() if k != "_path"})
    if "name" in model_payload:
        config["model"]["name"] = model_payload["name"]
    config["model"]["_path"] = model_payload["_path"]
    config["task"] = task
    config["mode"] = mode
    if data is not None:
        config["data"] = load_data_config(data)
        config["data"]["config"] = str(data)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def _resolve_recipe_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix == ".yaml":
        name = candidate.name
    else:
        name = f"{candidate.name}.yaml"
    direct = RECIPE_ROOT / name
    if direct.exists():
        return direct.resolve()
    for path in RECIPE_ROOT.rglob(name):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Unable to resolve recipe config: {value}")


def _resolve_recipe_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    for parent_key, child_key in (
        ("model", "checkpoint"),
        ("data", "root"),
        ("dataset", "root"),
        ("eval", "checkpoint_path"),
        ("predict", "source"),
    ):
        section = config.get(parent_key)
        if isinstance(section, dict) and section.get(child_key):
            value = Path(str(section[child_key])).expanduser()
            if not value.is_absolute():
                section[child_key] = str((repo_root / value).resolve())
    return config


def load_recipe_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = deep_merge(load_recipe_defaults(), _load_yaml(_resolve_recipe_path(path)))
    if overrides:
        config = apply_overrides(config, overrides)
    config = _resolve_recipe_paths(config, Path(config["_path"]).parent)
    if "references" not in config:
        raise ValueError("Recipe configs must define a references block")
    return config
