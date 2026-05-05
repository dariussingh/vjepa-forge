from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.engine.checkpointing import load_checkpoint
from vjepa_forge.engine.model import ForgeModel
from vjepa_forge.engine.trainer import BaseTrainer
from vjepa_forge.heads.anomaly.modeling import NativeExtractedFeatures
import vjepa_forge.tasks.anomaly.runtime as anomaly_runtime_mod
import vjepa_forge.tasks.anomaly.predict as anomaly_predict_mod
import vjepa_forge.tasks.anomaly.val as anomaly_val_mod
from vjepa_forge.engine.validator import ValidationResult
from vjepa_forge.tasks import TASK_REGISTRY


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
    assert val_result.metrics is not None
    assert "top1" in val_result.metrics
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
    header = (Path(first.run_dir) / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "top1" in header
    checkpoint = load_checkpoint(first.last_checkpoint)
    assert "top1" in checkpoint["metrics"]

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
    assert "top1" in rows[0].split(",")
    assert rows[-1].startswith("2,")


def test_trainer_skips_missing_validation_split_with_warning(caplog):
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
    trainer = BaseTrainer(model, data="unused", num_workers=0)
    original_validator = TASK_REGISTRY["classify"]["val"]

    class MissingValValidator:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            raise FileNotFoundError("missing val split")

    TASK_REGISTRY["classify"]["val"] = MissingValValidator
    with caplog.at_level("WARNING"):
        try:
            assert trainer.validate_epoch() is None
        finally:
            TASK_REGISTRY["classify"]["val"] = original_validator
    assert "Skipping validation split" in caplog.text


def test_trainer_propagates_unexpected_validation_errors():
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
    trainer = BaseTrainer(model, data="unused", num_workers=0)
    original_validator = TASK_REGISTRY["classify"]["val"]

    class BrokenValidator:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            raise RuntimeError("cuda oom")

    TASK_REGISTRY["classify"]["val"] = BrokenValidator
    try:
        with pytest.raises(RuntimeError, match="cuda oom"):
            trainer.validate_epoch()
    finally:
        TASK_REGISTRY["classify"]["val"] = original_validator


def test_trainer_validate_epoch_returns_task_validation_result():
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
    trainer = BaseTrainer(model, data="unused", num_workers=0)
    original_validator = TASK_REGISTRY["classify"]["val"]

    class StubValidator:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            return ValidationResult(loss=0.25, batches=1, metrics={"top1": 1.0}, split="val")

    TASK_REGISTRY["classify"]["val"] = StubValidator
    try:
        result = trainer.validate_epoch()
    finally:
        TASK_REGISTRY["classify"]["val"] = original_validator
    assert result is not None
    assert result.loss == 0.25
    assert result.metrics == {"top1": 1.0}


def test_detect_and_segment_losses_fail_fast():
    detect_model = ForgeModel(
        {
            "task": "detect",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_classes": 5,
            "head": {"num_queries": 8},
        },
        data={"task": "detect", "media": "image", "image_size": 64},
    )
    detect_batch = ForgeBatch(x=torch.randn(1, 3, 64, 64), media="image", task="detect", labels={"detections": []}, paths=[], meta=[])
    detect_trainer = BaseTrainer(detect_model, data="unused", num_workers=0)
    with pytest.raises(NotImplementedError, match="Detection training/validation loss"):
        detect_trainer.compute_loss(detect_batch, detect_model(detect_batch))

    segment_model = ForgeModel(
        {
            "task": "segment",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_classes": 3,
        },
        data={"task": "segment", "media": "image", "image_size": 64},
    )
    segment_batch = ForgeBatch(x=torch.randn(1, 3, 64, 64), media="image", task="segment", labels={"segments": []}, paths=[], meta=[])
    segment_trainer = BaseTrainer(segment_model, data="unused", num_workers=0)
    with pytest.raises(NotImplementedError, match="Segmentation training/validation loss"):
        segment_trainer.compute_loss(segment_batch, segment_model(segment_batch))


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


def _write_video(path: Path, *, frames: int = 6, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (size, size))
    assert writer.isOpened()
    try:
        for idx in range(frames):
            frame = np.full((size, size, 3), idx * 20, dtype=np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


class _FakeNativeExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tubelet_size = 1

    def forward(self, clip: torch.Tensor) -> NativeExtractedFeatures:
        batch, _, frames, _, _ = clip.shape
        half = frames // 2
        pooled = clip.mean(dim=(3, 4)).permute(0, 2, 1)
        context = pooled[:, :half, :]
        target = pooled[:, half:, :]
        return NativeExtractedFeatures(context_tokens=context, target_tokens=target)


class _FakeNativePredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.predictor.weight.copy_(torch.eye(3))

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.predictor(context_tokens).unsqueeze(2)

    def project_targets(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return target_tokens.unsqueeze(2)


def _patch_native_components(monkeypatch, predictor: torch.nn.Module | None = None) -> torch.nn.Module:
    native_predictor = predictor or _FakeNativePredictor()
    monkeypatch.setattr(
        anomaly_runtime_mod,
        "_build_components",
        lambda cfg, device: (_FakeNativeExtractor(), native_predictor),
    )
    return native_predictor


def _native_runtime_config(
    tmp_path: Path,
    dataset_yaml: Path,
    *,
    data: dict | None = None,
    train: dict | None = None,
    val: dict | None = None,
    predict: dict | None = None,
) -> dict:
    return {
        "model": {
            "arch": "vjepa_native",
            "name": "vjepa2_1_vit_base_384",
            "backbone": {"checkpoint": "dummy.pt", "checkpoint_key": "ema_encoder"},
            "predictor_type": "vjepa_native",
        },
        "data": {"_path": str(dataset_yaml), "image_size": 32, "past_frames": 2, "future_frames": 2, "stride": 1, **(data or {})},
        "train": {"device": "cpu", "project": str(tmp_path / "runs"), "name": "anomaly-exp", "exist_ok": True, **(train or {})},
        "val": {"batch_size": 1, "num_workers": 0, **(val or {})},
        "predict": {**(predict or {})},
        "export": {},
        "output": {},
    }


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
    _patch_native_components(monkeypatch)
    base_config = _native_runtime_config(
        tmp_path,
        dataset_yaml,
        train={"epochs": 1, "batch_size": 1, "num_workers": 0},
    )
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
    checkpoint = load_checkpoint(first.last_checkpoint)
    assert "model_state" in checkpoint
    assert checkpoint.get("extras", {}).get("effective_lr") is not None
    assert "predictor_state" not in checkpoint
    assert "predictor_state" not in checkpoint.get("extras", {})


def test_anomaly_runtime_accepts_legacy_predictor_state_checkpoints(tmp_path: Path, monkeypatch):
    dataset_yaml = _write_anomaly_dataset(tmp_path / "anomaly_legacy")
    predictor = _patch_native_components(monkeypatch)
    ckpt = tmp_path / "legacy_predictor.pt"
    torch.save({"predictor_state": predictor.predictor.state_dict()}, ckpt)
    result = anomaly_runtime_mod.predict_from_runtime_config(
        _native_runtime_config(
            tmp_path,
            dataset_yaml,
            train={"name": "legacy-ckpt"},
            val={"checkpoint_path": str(ckpt), "predictor_source": "anomaly_checkpoint"},
            predict={"split": "test", "batch_size": 1, "num_workers": 0},
        )
    )
    assert result.metrics["predictor_frame_auc"] is not None


def test_anomaly_eval_uses_validation_split_for_threshold_calibration(tmp_path: Path, monkeypatch):
    dataset_yaml = _write_anomaly_dataset(tmp_path / "anomaly_eval")
    _patch_native_components(monkeypatch)
    monkeypatch.setattr(anomaly_runtime_mod, "_make_loaders", lambda cfg, include_test=True: {"val_loader": "val", "test_loader": "test"})
    monkeypatch.setattr(anomaly_runtime_mod, "load_checkpoint", lambda path: {"model_state": _FakeNativePredictor().predictor.state_dict()})
    monkeypatch.setattr(anomaly_runtime_mod, "_make_output_root", lambda cfg: tmp_path / "runs" / "eval-threshold")

    def _summary(normal_score: float, anomaly_score: float) -> dict[str, object]:
        return {
            "videos": {
                "normal": {
                    "frame_ids": [0],
                    "predictor_scores": [normal_score],
                    "frozen_scores": [normal_score],
                    "labels": [0],
                },
                "anomaly": {
                    "frame_ids": [0],
                    "predictor_scores": [anomaly_score],
                    "frozen_scores": [anomaly_score],
                    "labels": [1],
                },
            }
        }

    def _fake_aggregate_scores(loader, predictor, feature_extractor, runtime, desc, model_cfg):
        if loader == "val":
            return _summary(0.2, 0.8), {"avg_decode_time": 0.0, "avg_model_time": 0.0}
        return _summary(0.6, 0.7), {"avg_decode_time": 0.0, "avg_model_time": 0.0}

    monkeypatch.setattr(anomaly_runtime_mod, "_aggregate_scores", _fake_aggregate_scores)
    result = anomaly_runtime_mod.validate_from_runtime_config(
        _native_runtime_config(
            tmp_path,
            dataset_yaml,
            train={"name": "eval-threshold"},
            val={"split": "test", "batch_size": 1, "num_workers": 0, "checkpoint_path": "dummy.pt", "predictor_source": "anomaly_checkpoint", "threshold_std_multiplier": 0.0},
        )
    )
    assert result.metrics["predictor_threshold"] == pytest.approx(0.2)
    assert result.metrics["predictor_val_false_positive_rate"] == 0.0


def test_anomaly_clip_metrics_include_confusion_matrix():
    summary = {
        "videos": {
            "normal_clip": {
                "frame_ids": [0, 1],
                "predictor_scores": [0.10, 0.20],
                "frozen_scores": [0.12, 0.18],
                "labels": [0, 0],
            },
            "anomaly_hit": {
                "frame_ids": [0, 1],
                "predictor_scores": [0.60, 0.40],
                "frozen_scores": [0.55, 0.35],
                "labels": [0, 1],
            },
            "anomaly_miss": {
                "frame_ids": [0, 1],
                "predictor_scores": [0.25, 0.30],
                "frozen_scores": [0.22, 0.28],
                "labels": [1, 1],
            },
        }
    }
    metrics = anomaly_runtime_mod._clip_level_metrics(summary, "predictor_scores", threshold=0.5, reduction="max")
    assert metrics["auc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 1, "tp": 1}
    assert metrics["counts"] == {"total": 3, "normal": 1, "anomaly": 2}
    assert metrics["accuracy"] == 2.0 / 3.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 2.0 / 3.0


def test_anomaly_predict_returns_thresholded_clip_summaries(tmp_path: Path, monkeypatch):
    dataset_yaml = _write_anomaly_dataset(tmp_path / "anomaly_predict")
    predictor = _patch_native_components(monkeypatch)
    ckpt = tmp_path / "predictor.pt"
    torch.save({"model_state": predictor.predictor.state_dict()}, ckpt)
    result = anomaly_runtime_mod.predict_from_runtime_config(
        _native_runtime_config(
            tmp_path,
            dataset_yaml,
            train={"name": "predict-threshold"},
            val={"checkpoint_path": str(ckpt), "predictor_source": "anomaly_checkpoint"},
            predict={"split": "test", "batch_size": 1, "num_workers": 0, "threshold": 0.1},
        )
    )
    assert result.metrics["predictor_thresholded"]["threshold"] == 0.1
    assert len(result.metrics["predictor_thresholded"]["clips"]) == 2


def test_anomaly_predict_visualize_requires_threshold(tmp_path: Path, monkeypatch):
    source = tmp_path / "single.mp4"
    _write_video(source)
    predictor = _patch_native_components(monkeypatch)
    ckpt = tmp_path / "predictor_source.pt"
    torch.save({"model_state": predictor.predictor.state_dict()}, ckpt)
    try:
        anomaly_runtime_mod.predict_from_runtime_config(
            _native_runtime_config(
                tmp_path,
                dataset_yaml=tmp_path / "unused.yaml",
                data={"video_backend": "decord"},
                train={"name": "predict-source"},
                val={"checkpoint_path": str(ckpt), "predictor_source": "anomaly_checkpoint"},
                predict={"source": str(source), "batch_size": 1, "num_workers": 0, "visualize": True},
            )
        )
    except ValueError as exc:
        assert "predict.threshold" in str(exc)
    else:
        raise AssertionError("Expected visualize without threshold to fail")


def test_anomaly_predict_standalone_source_writes_visualized_mp4(tmp_path: Path, monkeypatch):
    source = tmp_path / "single_vis.mp4"
    _write_video(source, frames=8)
    predictor = _patch_native_components(monkeypatch)
    ckpt = tmp_path / "predictor_source_vis.pt"
    torch.save({"model_state": predictor.predictor.state_dict()}, ckpt)
    result = anomaly_runtime_mod.predict_from_runtime_config(
        _native_runtime_config(
            tmp_path,
            dataset_yaml=tmp_path / "unused.yaml",
            data={"video_backend": "decord"},
            train={"name": "predict-source-vis"},
            val={"checkpoint_path": str(ckpt), "predictor_source": "anomaly_checkpoint"},
            predict={
                "source": str(source),
                "batch_size": 1,
                "num_workers": 0,
                "threshold": 0.1,
                "visualize": True,
                "output_dir": str(tmp_path / "viz"),
            },
        )
    )
    assert result.rendered_outputs is not None
    assert len(result.rendered_outputs) == 1
    rendered = Path(result.rendered_outputs[0])
    assert rendered.exists()
    assert rendered.stat().st_size > 0


def test_native_anomaly_predictor_source_defaults_to_pretrained_when_no_anomaly_checkpoint(tmp_path: Path):
    cfg = {
        "model": {"arch": "vjepa_native", "predictor_type": "vjepa_native"},
        "eval": {"predictor_source": "auto", "checkpoint_path": None},
        "output": {"root": str(tmp_path / "runs")},
        "dataset": {"dataset_yaml": str(tmp_path / "dummy.yaml")},
        "train": {"project": str(tmp_path / "runs"), "name": "native-source", "exist_ok": True, "resume": False},
    }
    assert anomaly_runtime_mod._resolve_predictor_source(cfg, "eval") == "pretrained_vjepa"


def test_anomaly_runtime_rejects_removed_predictor_types(tmp_path: Path):
    dataset_yaml = _write_anomaly_dataset(tmp_path / "anomaly_removed_predictor")
    config = _native_runtime_config(tmp_path, dataset_yaml)
    config["model"]["predictor_type"] = "global_mlp"
    with pytest.raises(ValueError, match="Legacy predictors were removed"):
        anomaly_runtime_mod._build_cfg(config, action="train")


@pytest.mark.parametrize("model_name", ["vjepa2_1_vit_base_384", "vjepa2_1_vit_large_384"])
def test_native_anomaly_validation_uses_pretrained_predictor_without_checkpoint(tmp_path: Path, monkeypatch, model_name: str):
    dataset_yaml = _write_anomaly_dataset(tmp_path / "anomaly_native_eval")

    class _NativeExtractor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tubelet_size = 1

        def forward(self, clip: torch.Tensor) -> NativeExtractedFeatures:
            batch, _, frames, _, _ = clip.shape
            context = torch.zeros(batch, frames // 2, 1, 3, device=clip.device)
            target = torch.ones(batch, frames // 2, 1, 3, device=clip.device)
            return NativeExtractedFeatures(context_tokens=context.reshape(batch, -1, 3), target_tokens=target.reshape(batch, -1, 3))

    class _NativePredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predictor = torch.nn.Linear(3, 3, bias=False)

        def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
            projected = self.predictor(context_tokens)
            return projected.unsqueeze(2)

        def project_targets(self, target_tokens: torch.Tensor) -> torch.Tensor:
            projected = target_tokens + 0.5
            return projected.unsqueeze(2)

    predictor = _NativePredictor()
    monkeypatch.setattr(anomaly_runtime_mod, "_build_components", lambda cfg, device: (_NativeExtractor(), predictor))
    monkeypatch.setattr(anomaly_runtime_mod, "_make_loaders", lambda cfg, include_test=True, **kwargs: {"val_loader": "val", "test_loader": "test"})
    monkeypatch.setattr(anomaly_runtime_mod, "_make_output_root", lambda cfg: tmp_path / "runs" / "native-pretrained")

    def _summary() -> dict[str, object]:
        return {
            "videos": {
                "normal": {"frame_ids": [0], "predictor_scores": [0.2], "frozen_scores": [0.1], "labels": [0]},
                "anomaly": {"frame_ids": [0], "predictor_scores": [0.9], "frozen_scores": [0.8], "labels": [1]},
            }
        }

    monkeypatch.setattr(
        anomaly_runtime_mod,
        "_aggregate_scores",
        lambda *args, **kwargs: (_summary(), {"avg_decode_time": 0.0, "avg_model_time": 0.0}),
    )

    def _fail_load_checkpoint(path):
        raise AssertionError(f"load_checkpoint should not be called for pretrained native validation: {path}")

    monkeypatch.setattr(anomaly_runtime_mod, "load_checkpoint", _fail_load_checkpoint)
    result = anomaly_runtime_mod.validate_from_runtime_config(
        {
            "model": {
                "arch": "vjepa_native",
                "name": model_name,
                "backbone": {"checkpoint": "dummy.pt", "checkpoint_key": "ema_encoder"},
                "predictor_type": "vjepa_native",
            },
            "data": {"_path": str(dataset_yaml), "image_size": 32, "past_frames": 2, "future_frames": 2, "stride": 1},
            "train": {"device": "cpu", "project": str(tmp_path / "runs"), "name": "native-pretrained", "exist_ok": True},
            "val": {"split": "val", "batch_size": 1, "num_workers": 0, "predictor_source": "auto"},
            "predict": {},
            "export": {},
            "output": {},
        }
    )
    assert result.metrics["checkpoint_path"] is None
    assert result.metrics["predictor_frame_auc"] == pytest.approx(1.0)
