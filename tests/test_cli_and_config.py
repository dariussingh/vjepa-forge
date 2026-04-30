from pathlib import Path

from vjepa_forge.configs.loader import load_recipe


def test_recipe_loads_with_references():
    recipe = Path("vjepa_forge/recipes/classification/imagenet1k_vitb.yaml")
    config = load_recipe(recipe)
    assert config["references"]["backbone"] == ["vjepa2_1"]
    assert config["task"] == "classification"
