from __future__ import annotations

import argparse
from typing import Any

from vjepa_forge.configs.loader import load_config, parse_override_value


def parse_kv_pairs(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        overrides[key] = parse_override_value(raw_value)
    return overrides


def parse_config_args(description: str) -> tuple[argparse.ArgumentParser, argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    overrides = parse_kv_pairs(args.overrides)
    config_path = overrides.pop("config", None)
    if config_path is None:
        config_path = overrides.pop("recipe", None)
    if not config_path:
        raise SystemExit("Pass config=<path-to-yaml>")
    config = load_config(config_path, overrides=overrides)
    return parser, args, config


def parse_recipe_args(description: str) -> tuple[argparse.ArgumentParser, argparse.Namespace, dict[str, Any]]:
    return parse_config_args(description)
