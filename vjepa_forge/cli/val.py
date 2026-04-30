from __future__ import annotations

from vjepa_forge.cli.common import parse_config_args
from vjepa_forge.engine.trainer import evaluate


def main() -> None:
    _, _, config = parse_config_args("Validate a vjepa-forge config.")
    print(evaluate(config))
