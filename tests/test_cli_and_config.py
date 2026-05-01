from pathlib import Path

import pytest

from vjepa_forge.cfg.loader import load_runtime_config
from vjepa_forge.cli.main import parse_kv_pairs


def test_parse_kv_pairs_supports_public_model_and_data_args():
    parsed = parse_kv_pairs(["model=vjepa21-b.yaml", "data=kinetics400.yaml", "train.epochs=3"])
    assert parsed["model"] == "vjepa21-b.yaml"
    assert parsed["data"] == "kinetics400.yaml"
    assert parsed["train.epochs"] == 3


def test_runtime_config_uses_media_not_input_type():
    config = load_runtime_config(task="detect", mode="train", model="vjepa21-rfdetr.yaml", data="coco.yaml")
    assert config["task"] == "detect"
    assert config["mode"] == "train"
    assert config["data"]["media"] == "image"
    assert "input_type" not in config


def test_model_and_dataset_cfg_paths_resolve_from_cfg_package():
    config = load_runtime_config(task="classify", mode="train", model="vjepa21-b", data="kinetics400")
    assert Path(config["model"]["_path"]).name == "vjepa21-b.yaml"
    assert Path(config["data"]["_path"]).name == "kinetics400.yaml"


def test_cafe_dataset_cfg_is_shipped():
    config = load_runtime_config(task="anomaly", mode="train", model="vjepa21-predictor.yaml", data="cafe.yaml")
    assert config["data"]["task"] == "anomaly"
    assert config["data"]["media"] == "video"
