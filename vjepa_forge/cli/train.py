from __future__ import annotations

from vjepa_forge.cli.common import parse_recipe_args
from vjepa_forge.engine.trainer import train
from vjepa_forge.heads.anomaly.engine import main_train as anomaly_train


def main() -> None:
    _, _, config = parse_recipe_args("Train a vjepa-forge recipe.")
    if config["task"] == "anomaly":
        anomaly_train(["--config", config["_recipe_path"]])
        return
    result = train(config)
    print(result.checkpoint_path)
