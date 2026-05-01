from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from vjepa_forge.engine.model import ForgeModel
import vjepa_forge.tasks.anomaly.predict as anomaly_predict_mod
import vjepa_forge.tasks.anomaly.val as anomaly_val_mod


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
    val_result = model.val(data=str(dataset_yaml), batch_size=1, num_workers=0, device="cpu", split="val")
    pred_result = model.predict(data=str(dataset_yaml), batch_size=1, num_workers=0, device="cpu", split="val")
    assert train_result.steps == 2
    assert val_result.batches == 1
    assert len(pred_result.outputs) == 1


def _write_anomaly_dataset(root: Path) -> Path:
    (root / "videos" / "train").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "val").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "test").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for idx, abnormal in enumerate((False, True)):
            name = f"{split}_{idx}"
            torch.save(torch.randn(4, 3, 32, 32), root / "videos" / split / f"{name}.pt")
            label = "ano normal\n" if not abnormal else "ano abnormal 1 2 0\n"
            (root / "labels" / split / f"{name}.txt").write_text(label, encoding="utf-8")
        (root / "splits" / f"{split}.txt").write_text(
            f"videos/{split}/{split}_0.pt\nvideos/{split}/{split}_1.pt\n",
            encoding="utf-8",
        )
    dataset_yaml = root / "forge.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "task: anomaly",
                "media: video",
                "names:",
                "  0: anomaly",
                "splits:",
                "  train: splits/train.txt",
                "  val: splits/val.txt",
                "  test: splits/test.txt",
                "labels:",
                "  format: forge-yolo",
                "  root: labels",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def test_anomaly_val_and_predict_return_metric_summaries(tmp_path: Path, monkeypatch):
    dataset_yaml = _write_anomaly_dataset(tmp_path / "anomaly")
    model = ForgeModel(
        {
            "task": "anomaly",
            "media": "video",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 32,
            "num_frames": 2,
        },
        data={"task": "anomaly", "media": "video", "image_size": 32, "clip_len": 4, "past_frames": 2, "future_frames": 2},
    )
    monkeypatch.setattr(
        anomaly_val_mod,
        "validate_from_runtime_config",
        lambda config: SimpleNamespace(
            metrics={"clip_roc_auc": 1.0, "frame_roc_auc": 1.0},
            split=config["val"]["split"],
        ),
    )
    monkeypatch.setattr(
        anomaly_predict_mod,
        "predict_from_runtime_config",
        lambda config: SimpleNamespace(
            summary={"clip_roc_auc": 1.0, "clips": [{"name": "a"}, {"name": "b"}]},
            split=config["predict"]["split"],
        ),
    )
    val_result = model.val(data=str(dataset_yaml), batch_size=1, num_workers=0, device="cpu", split="val")
    pred_result = model.predict(data=str(dataset_yaml), batch_size=1, num_workers=0, device="cpu")
    assert val_result.metrics["clip_roc_auc"] is not None
    assert val_result.metrics["frame_roc_auc"] is not None
    assert pred_result.summary["clip_roc_auc"] is not None
    assert len(pred_result.summary["clips"]) == 2
