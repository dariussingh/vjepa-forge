from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from vjepa_forge.engine.model import ForgeModel
from vjepa_forge.heads.anomaly.modeling import ExtractedFeatures
import vjepa_forge.tasks.anomaly.runtime as anomaly_runtime_mod
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


def test_forge_model_train_saves_and_resumes_checkpoints(tmp_path: Path):
    dataset_yaml = _write_dataset(tmp_path / "dataset_resume")
    project = tmp_path / "runs"
    model = ForgeModel(
        {
            "task": "classify",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 32,
            "num_classes": 2,
        },
        data={"task": "classify", "media": "image", "image_size": 32, "lr": 1.0e-4},
    )
    first = model.train(data=str(dataset_yaml), epochs=1, batch_size=1, num_workers=0, device="cpu", project=str(project), name="resume-exp", exist_ok=True)
    assert Path(first.last_checkpoint).exists()
    assert Path(first.best_checkpoint).exists()
    assert Path(first.run_dir, "results.csv").exists()

    resumed_model = ForgeModel(
        {
            "task": "classify",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 32,
            "num_classes": 2,
        },
        data={"task": "classify", "media": "image", "image_size": 32, "lr": 1.0e-4},
    )
    second = resumed_model.train(
        data=str(dataset_yaml),
        epochs=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        project=str(project),
        name="resume-exp",
        exist_ok=True,
        resume=str(Path(first.run_dir) / "weights" / "last.pt"),
    )
    rows = (Path(second.run_dir) / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
    assert rows[-1].startswith("2,")


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


def test_anomaly_runtime_saves_and_resumes_unified_checkpoints(tmp_path: Path, monkeypatch):
    dataset_yaml = _write_anomaly_dataset(tmp_path / "anomaly_resume")

    class _FakeExtractor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tubelet_size = 1
            self.embed_dim = 3
            self.grid_depth = 2
            self.grid_size = 1

        def forward(self, clip: torch.Tensor) -> ExtractedFeatures:
            pooled = clip.mean(dim=(2, 3, 4))
            tokens = pooled.unsqueeze(1).unsqueeze(2).repeat(1, clip.shape[2], 1, 1)
            return ExtractedFeatures(pooled=pooled, tokens=tokens)

    monkeypatch.setattr(
        anomaly_runtime_mod,
        "build_feature_extractor",
        lambda **kwargs: _FakeExtractor(),
    )
    monkeypatch.setattr(
        anomaly_runtime_mod,
        "build_predictor",
        lambda model_cfg, feature_extractor: torch.nn.Linear(feature_extractor.embed_dim, feature_extractor.embed_dim, bias=False),
    )
    base_config = {
        "model": {
            "name": "vjepa2_1_vit_base_384",
            "backbone": {"checkpoint": "dummy.pt", "checkpoint_key": "ema_encoder"},
            "predictor_type": "global_mlp",
            "hidden_dim": 4,
            "dropout": 0.0,
        },
        "data": {"_path": str(dataset_yaml), "image_size": 32, "past_frames": 2, "future_frames": 2, "stride": 1},
        "train": {"epochs": 1, "batch_size": 1, "num_workers": 0, "device": "cpu", "project": str(tmp_path / "runs"), "name": "anomaly-exp", "exist_ok": True},
        "val": {"batch_size": 1, "num_workers": 0},
        "predict": {},
        "export": {},
        "output": {},
    }
    first = anomaly_runtime_mod.train_from_runtime_config(base_config)
    assert Path(first.best_checkpoint).exists()
    assert Path(first.last_checkpoint).exists()
    assert Path(first.run_dir, "results.csv").exists()

    resumed = anomaly_runtime_mod.train_from_runtime_config(
        {
            **base_config,
            "train": {**base_config["train"], "epochs": 2, "resume": str(Path(first.run_dir) / "weights" / "last.pt")},
        }
    )
    rows = (Path(resumed.run_dir) / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
    assert rows[-1].startswith("2,")
