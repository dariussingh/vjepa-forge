from __future__ import annotations

from vjepa_forge.cli.common import parse_recipe_args


def main() -> None:
    _, _, config = parse_recipe_args("Validate a vjepa-forge recipe.")
    print({"task": config["task"], "backend": config.get("inference", {}).get("backend", "torch")})
