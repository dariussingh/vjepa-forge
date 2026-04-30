from pathlib import Path

from vjepa_forge.configs.loader import load_config, load_recipe


def test_config_loads_with_references():
    config_path = Path("vjepa_forge/configs/classification/imagenet1k_vitb.yaml")
    config = load_config(config_path)
    assert config["references"]["backbone"] == ["vjepa2_1"]
    assert config["task"] == "classification"


def test_video_detection_config_loads():
    config_path = Path("vjepa_forge/configs/detection/imagenet_vid_temporal_detr.yaml")
    config = load_config(config_path)
    assert config["task"] == "detection"
    assert config["input_type"] == "video"


def test_load_recipe_alias_still_works():
    config_path = Path("vjepa_forge/configs/classification/imagenet1k_vitb.yaml")
    config = load_recipe(config_path)
    assert config["task"] == "classification"
