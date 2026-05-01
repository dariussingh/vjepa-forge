from __future__ import annotations

import sys
from typing import Any

from vjepa_forge.cfg.loader import load_runtime_config, parse_override_value
from vjepa_forge.data.converters import convert_cafe_to_forge
from vjepa_forge.engine.exporter import Exporter
from vjepa_forge.engine.model import ForgeModel
from vjepa_forge.tasks.anomaly.runtime import export_from_runtime_config, predict_from_runtime_config, train_from_runtime_config, validate_from_runtime_config
import torch


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
        source = overrides.get("source")
        out = overrides.get("out")
        if action == "cafe":
            if not source or not out:
                raise SystemExit("forge convert cafe requires source=<path> and out=<path>")
            print(convert_cafe_to_forge(source, out))
            return
        raise SystemExit(f"Unknown converter: {action}")
    if action == "export" and not overrides.get("data"):
        raise SystemExit("forge <task> export requires data=<forge.yaml>")
    if action not in {"train", "val", "predict", "export"}:
        raise SystemExit(f"Unknown mode: {action}")
    model_ref = overrides.pop("model", "vjepa21-b.yaml")
    data_ref = overrides.pop("data", None)
    config = load_runtime_config(task=namespace, mode=action, model=model_ref, data=data_ref, overrides=overrides)
    if namespace == "anomaly":
        if action == "train":
            print(train_from_runtime_config(config))
            return
        if action == "val":
            print(validate_from_runtime_config(config))
            return
        if action == "predict":
            print(predict_from_runtime_config(config))
            return
        if action == "export":
            print(export_from_runtime_config(config))
            return
    model_cfg = dict(config["model"])
    model_cfg["task"] = namespace
    model_cfg["media"] = config["data"]["media"]
    model_cfg["num_classes"] = max(len(config["data"].get("names", {})), int(model_cfg.get("num_classes", 2)))
    runtime_data = dict(config["data"])
    runtime_data.update(config.get(action, {}))
    runtime_data["image_size"] = int(runtime_data.get("image_size", model_cfg.get("image_size", 384)))
    forge_model = ForgeModel(model_cfg, data=runtime_data)
    if action == "train":
        result = forge_model.train(
            data=config["data"]["_path"],
            epochs=int(config["train"].get("epochs", 1)),
            batch_size=int(config["train"].get("batch_size", 2)),
            num_workers=int(config["train"].get("num_workers", 0)),
            device=str(config["train"].get("device", "cpu")),
            save=bool(config["train"].get("save", True)),
            save_period=int(config["train"].get("save_period", 0)),
            resume=config["train"].get("resume", False),
            project=config["train"].get("project"),
            name=config["train"].get("name"),
            exist_ok=bool(config["train"].get("exist_ok", False)),
        )
    elif action == "val":
        result = forge_model.val(data=config["data"]["_path"], batch_size=int(config["val"].get("batch_size", 2)), num_workers=int(config["val"].get("num_workers", 0)), device=str(config["train"].get("device", "cpu")), split=str(config["val"].get("split", "val")))
    elif action == "predict":
        result = forge_model.predict(data=config["data"]["_path"], batch_size=int(config["predict"].get("batch_size", 1)), num_workers=int(config["predict"].get("num_workers", 0)), device=str(config["train"].get("device", "cpu")), split=str(config["predict"].get("split", "test")))
    else:  # export
        if namespace != "anomaly":
            raise SystemExit("The active forge export path is currently implemented for anomaly only.")
        export_cfg = config.get("export", {})
        image_size = int(runtime_data.get("image_size", 384))
        clip_len = int(runtime_data.get("clip_len", runtime_data.get("num_frames", 8)))
        if runtime_data["media"] == "video":
            sample = torch.randn(1, clip_len, 3, image_size, image_size)
        else:
            sample = torch.randn(1, 3, image_size, image_size)
        result = Exporter().export(
            forge_model,
            format=str(export_cfg.get("format", "onnx")),
            output_path=str(export_cfg.get("output_path", f"{namespace}.onnx")),
            sample=sample,
            opset=int(export_cfg.get("opset", 17)),
            dynamic_axes=bool(export_cfg.get("dynamic_axes", True)),
        )
    print(result)
