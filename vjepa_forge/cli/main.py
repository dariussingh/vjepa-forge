from __future__ import annotations

import sys
from typing import Any

from vjepa_forge.cfg.loader import load_runtime_config, parse_override_value
from vjepa_forge.engine.model import ForgeModel


def parse_kv_pairs(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        overrides[key] = parse_override_value(raw_value)
    return overrides


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: forge <task|convert> <mode|converter> key=value ...")
    namespace = sys.argv[1]
    action = sys.argv[2]
    overrides = parse_kv_pairs(sys.argv[3:])
    if namespace == "convert":
        print({"converter": action, "args": overrides})
        return
    model_ref = overrides.pop("model", "vjepa21-b.yaml")
    data_ref = overrides.pop("data", None)
    config = load_runtime_config(task=namespace, mode=action, model=model_ref, data=data_ref, overrides=overrides)
    model_cfg = dict(config["model"])
    model_cfg["task"] = namespace
    model_cfg["media"] = config["data"]["media"]
    model_cfg["num_classes"] = max(len(config["data"].get("names", {})), int(model_cfg.get("num_classes", 2)))
    runtime_data = dict(config["data"])
    runtime_data.update(config.get(action, {}))
    runtime_data["image_size"] = int(runtime_data.get("image_size", model_cfg.get("image_size", 64)))
    forge_model = ForgeModel(model_cfg, data=runtime_data)
    if action == "train":
        result = forge_model.train(data=config["data"]["_path"], epochs=int(config["train"].get("epochs", 1)), batch_size=int(config["train"].get("batch_size", 2)), num_workers=int(config["train"].get("num_workers", 0)), device=str(config["train"].get("device", "cpu")))
    elif action == "val":
        result = forge_model.val(data=config["data"]["_path"], batch_size=int(config["val"].get("batch_size", 2)), num_workers=int(config["val"].get("num_workers", 0)), device=str(config["train"].get("device", "cpu")))
    elif action == "predict":
        result = forge_model.predict(data=config["data"]["_path"], batch_size=1, num_workers=0, device=str(config["train"].get("device", "cpu")))
    else:
        raise SystemExit(f"Unknown mode: {action}")
    print(result)
