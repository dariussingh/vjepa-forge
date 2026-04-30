from pathlib import Path

from vjepa_forge.configs.loader import load_recipe


def test_recipe_loads_with_references():
    recipe = Path("vjepa_forge/recipes/classification/imagenet1k_vitb.yaml")
    config = load_recipe(recipe)
    assert config["references"]["backbone"] == ["vjepa2_1"]
    assert config["task"] == "classification"


def test_video_detection_recipe_loads():
    recipe = Path("vjepa_forge/recipes/detection/imagenet_vid_temporal_detr.yaml")
    config = load_recipe(recipe)
    assert config["task"] == "detection"
    assert config["input_type"] == "video"
