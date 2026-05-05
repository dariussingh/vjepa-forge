from __future__ import annotations

from vjepa_forge.cli.common import parse_config_args
from vjepa_forge.engine.trainer import train


def main() -> None:
    _, _, config = parse_config_args("Train a vjepa-forge config.")
    result = train(config)
    print(result.checkpoint_path)
