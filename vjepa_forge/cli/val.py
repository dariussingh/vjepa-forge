from __future__ import annotations

from vjepa_forge.cli.common import parse_recipe_args
from vjepa_forge.engine.trainer import evaluate


def main() -> None:
    _, _, config = parse_recipe_args("Validate a vjepa-forge recipe.")
    print(evaluate(config))
