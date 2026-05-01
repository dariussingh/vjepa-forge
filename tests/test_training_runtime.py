from pathlib import Path

from PIL import Image

from vjepa_forge.engine.model import ForgeModel


def _write_dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        Image.new("RGB", (32, 32), color=(64, 64, 64)).save(root / "images" / "train" / f"{idx}.jpg")
        (root / "labels" / "train" / f"{idx}.txt").write_text(f"cls {idx % 2}\n", encoding="utf-8")
    (root / "splits" / "train.txt").write_text("images/train/0.jpg\nimages/train/1.jpg\n", encoding="utf-8")
    (root / "splits" / "val.txt").write_text("images/train/0.jpg\n", encoding="utf-8")
    dataset_yaml = root / "forge.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "task: classify",
                "media: image",
                "names:",
                "  0: zero",
                "  1: one",
                "splits:",
                "  train: splits/train.txt",
                "  val: splits/val.txt",
                "labels:",
                "  format: forge-yolo",
                "  root: labels",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def test_forge_model_train_val_predict_roundtrip(tmp_path: Path):
    dataset_yaml = _write_dataset(tmp_path / "dataset")
    model = ForgeModel(
        {
            "task": "classify",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 32,
            "num_classes": 2,
        },
        data={"task": "classify", "media": "image", "image_size": 32},
    )
    train_result = model.train(data=str(dataset_yaml), epochs=1, batch_size=1, num_workers=0, device="cpu")
    val_result = model.val(data=str(dataset_yaml), batch_size=1, num_workers=0, device="cpu")
    pred_result = model.predict(data=str(dataset_yaml), batch_size=1, num_workers=0, device="cpu")
    assert train_result.steps == 2
    assert val_result.batches == 1
    assert len(pred_result.outputs) == 1
