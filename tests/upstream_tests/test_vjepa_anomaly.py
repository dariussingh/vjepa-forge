import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "vjepa-anomaly"
for path in (REPO_ROOT, EXPERIMENT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from vjepa_anomaly.dataset import (
    VideoRecord,
    discover_avenue_root,
    discover_cafe_root,
    discover_ped2_root,
    load_dataset_bundle,
    split_train_val_videos,
)
from vjepa_anomaly.engine import (
    _predict_sample_scores,
    _resolve_effective_lr,
    _resolve_eval_checkpoint_path,
    _roc_auc_score,
    _smooth_scores,
)
from vjepa_anomaly.modeling import ExtractedFeatures, build_predictor


class VjepaAnomalyTests(unittest.TestCase):
    def test_roc_auc_score_perfect_separation(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        self.assertEqual(_roc_auc_score(labels, scores), 1.0)

    def test_discover_ped2_root_prefers_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ped2_root = tmp_path / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
            (ped2_root / "Train").mkdir(parents=True)
            (ped2_root / "Test").mkdir(parents=True)
            self.assertEqual(discover_ped2_root(tmp_path), ped2_root)

    def test_discover_avenue_root_prefers_processed_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            avenue_root = tmp_path / "processed"
            (avenue_root / "Train").mkdir(parents=True)
            (avenue_root / "Test").mkdir(parents=True)
            (avenue_root / "frame_labels.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_avenue_root(tmp_path), avenue_root)

    def test_discover_cafe_root_prefers_processed_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cafe_root = tmp_path / "processed"
            (cafe_root / "Train").mkdir(parents=True)
            (cafe_root / "Test").mkdir(parents=True)
            (cafe_root / "frame_labels.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_cafe_root(tmp_path), cafe_root)

    def test_split_train_val_videos_produces_non_empty_validation(self):
        videos = [
            VideoRecord(name=f"Train{i:03d}", frame_paths=tuple(), mask_paths=None)
            for i in range(1, 5)
        ]
        train_split, val_split = split_train_val_videos(videos, val_ratio=0.25, seed=0)
        self.assertEqual(len(val_split), 1)
        self.assertEqual(len(train_split), 3)

    def test_load_dataset_bundle_for_avenue_uses_frame_label_map(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            processed = tmp_path / "processed"
            train_dir = processed / "Train" / "01"
            test_dir = processed / "Test" / "01"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            for frame_dir in (train_dir, test_dir):
                for idx in range(1, 4):
                    (frame_dir / f"{idx:04d}.jpg").write_bytes(b"x")
            (processed / "frame_labels.json").write_text(json.dumps({"01": [0, 1, 0]}), encoding="utf-8")

            bundle = load_dataset_bundle(
                {
                    "name": "avenue",
                    "root": str(tmp_path),
                    "split_seed": 0,
                    "val_ratio": 0.5,
                }
            )
            self.assertEqual(bundle.dataset_name, "avenue")
            self.assertEqual(bundle.test_videos[0].frame_labels, (0, 1, 0))

    def test_load_dataset_bundle_for_cafe_uses_frame_label_map(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            processed = tmp_path / "processed"
            train_dir = processed / "Train" / "train_clip"
            test_dir = processed / "Test" / "test_clip"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            for frame_dir in (train_dir, test_dir):
                for idx in range(1, 4):
                    (frame_dir / f"{idx:04d}.jpg").write_bytes(b"x")
            (processed / "frame_labels.json").write_text(json.dumps({"test_clip": [1, 1, 1]}), encoding="utf-8")

            bundle = load_dataset_bundle(
                {
                    "name": "cafe",
                    "root": str(tmp_path),
                    "split_seed": 0,
                    "val_ratio": 0.5,
                }
            )
            self.assertEqual(bundle.dataset_name, "cafe")
            self.assertEqual(bundle.test_videos[0].frame_labels, (1, 1, 1))

    def test_smooth_scores_preserves_length(self):
        values = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        smoothed = _smooth_scores(values, 3)
        self.assertEqual(smoothed.shape, values.shape)
        self.assertTrue(np.all(smoothed >= 0.0))
        self.assertTrue(np.all(smoothed <= 1.0))
        self.assertLess(smoothed.max(), values.max())

    def test_build_predictor_vit_patch_and_score_shape(self):
        class StubExtractor:
            embed_dim = 768
            grid_depth = 4
            grid_size = 2
            tubelet_size = 2

        predictor = build_predictor(
            {
                "predictor_type": "vit_patch",
                "predictor_embed_dim": 768,
                "predictor_depth": 1,
                "predictor_num_heads": 12,
                "dropout": 0.1,
                "predictor_use_rope": False,
                "token_aggregation": "topk_mean",
                "token_topk_fraction": 0.25,
            },
            StubExtractor(),
        )
        past = ExtractedFeatures(
            pooled=torch.randn(2, 768),
            tokens=torch.randn(2, 4, 4, 768),
        )
        future = ExtractedFeatures(
            pooled=torch.randn(2, 768),
            tokens=torch.randn(2, 4, 4, 768),
        )
        pred_scores, frozen_scores = _predict_sample_scores(
            predictor,
            past,
            future,
            {
                "predictor_type": "vit_patch",
                "token_aggregation": "topk_mean",
                "token_topk_fraction": 0.25,
            },
            tubelet_size=2,
        )
        self.assertEqual(tuple(pred_scores.shape), (2, 8))
        self.assertEqual(tuple(frozen_scores.shape), (2, 8))

    def test_resolve_effective_lr_autoscale_sqrt(self):
        lr, metadata = _resolve_effective_lr(
            {
                "batch_size": 16,
                "lr_mode": "autoscale",
                "reference_batch_size": 4,
                "reference_lr": 1.0e-4,
                "lr_scale_rule": "sqrt",
                "lr": 1.0e-4,
            }
        )
        self.assertAlmostEqual(lr, 2.0e-4)
        self.assertEqual(metadata["lr_mode"], "autoscale")

    def test_resolve_eval_checkpoint_path_latest_and_explicit(self):
        cfg = {
            "output": {"root": "outputs/vjepa-anomaly/cuhk_avenue_vitb"},
            "eval": {"checkpoint_target": "latest", "checkpoint_path": None},
        }
        latest_path = _resolve_eval_checkpoint_path(cfg)
        self.assertTrue(str(latest_path).endswith("latest_predictor.pt"))

        cfg["eval"]["checkpoint_path"] = "custom/checkpoint.pt"
        explicit_path = _resolve_eval_checkpoint_path(cfg)
        self.assertTrue(str(explicit_path).endswith("custom/checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
