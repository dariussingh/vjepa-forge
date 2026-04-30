from __future__ import annotations

import argparse
from typing import Any

from vjepa_forge.configs.loader import load_recipe, parse_override_value


def parse_kv_pairs(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        overrides[key] = parse_override_value(raw_value)
    return overrides


def parse_recipe_args(description: str) -> tuple[argparse.ArgumentParser, argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    overrides = parse_kv_pairs(args.overrides)
    recipe = overrides.pop("recipe", None)
    if not recipe:
        raise SystemExit("Pass recipe=<path-to-yaml>")
    config = load_recipe(recipe, overrides=overrides)
    return parser, args, config
