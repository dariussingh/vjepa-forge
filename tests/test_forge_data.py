from pathlib import Path

import pytest
import torch
from PIL import Image

import vjepa_forge.data.converters.cafe as cafe_converter
from vjepa_forge.data import AnomalyLoader, ClassifyLoader, DetectLoader, ForgeDataset, ForgeLabelParser, SegmentLoader, convert_cafe_to_forge
from vjepa_forge.data.cache import CachedFeatureItem, FeatureCacheStore, cached_feature_item_key
import vjepa_forge.data.image as image_mod
import vjepa_forge.data.video as video_mod
from vjepa_forge.data.forge.validator import resolve_label_path


def _write_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    image.save(path)


def _make_dataset(root: Path, *, media: str, task: str, label_lines: list[str]) -> Path:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    if media == "image":
        rel = "images/train/sample.jpg"
        _write_image(root / rel)
    else:
        rel = "videos/train/sample.pt"
        torch.save(torch.randn(4, 3, 32, 32), root / rel)
    (root / "labels" / "train" / "sample.txt").write_text("\n".join(label_lines), encoding="utf-8")
    (root / "splits" / "train.txt").write_text(rel + "\n", encoding="utf-8")
    (root / "splits" / "val.txt").write_text(rel + "\n", encoding="utf-8")
    yaml_path = root / "forge.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root}",
                f"task: {task}",
                f"media: {media}",
                "names:",
                "  0: sample",
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
    return yaml_path


def test_parser_understands_image_and_video_ops():
    parser = ForgeLabelParser()
    assert parser.parse_line("cls 3 7 9", media="image", task="classify").payload["class_ids"] == [3, 7, 9]
    assert parser.parse_line("det 0 1 0.5 0.5 0.2 0.2", media="video", task="detect").payload["frame_idx"] == 0
    assert parser.parse_line("seg 0 0 17 0.1 0.2 0.3 0.4", media="video", task="segment").payload["object_id"] == 17
    assert parser.parse_line("ano abnormal 2 4 1", media="video", task="anomaly").payload["class_id"] == 1


def test_forge_dataset_and_loaders_build_expected_batch_shapes(tmp_path: Path):
    image_yaml = _make_dataset(tmp_path / "image_cls", media="image", task="classify", label_lines=["cls 0"])
    image_dataset = ForgeDataset(image_yaml, split="train")
    image_batch = ClassifyLoader("image", image_size=32).collate([image_dataset[0]])
    assert image_batch.x.shape == (1, 3, 32, 32)

    video_yaml = _make_dataset(
        tmp_path / "video_detect",
        media="video",
        task="detect",
        label_lines=["det 0 0 0.5 0.5 0.2 0.2", "det 1 0 0.5 0.5 0.2 0.2"],
    )
    video_dataset = ForgeDataset(video_yaml, split="train")
    video_batch = DetectLoader("video", clip_len=4, image_size=32).collate([video_dataset[0]])
    assert video_batch.x.shape == (1, 4, 3, 32, 32)
    assert len(video_batch.labels["detections"][0]["detections"]) == 2

    seg_yaml = _make_dataset(
        tmp_path / "video_seg",
        media="video",
        task="segment",
        label_lines=["seg 0 0 17 0.1 0.2 0.3 0.4"],
    )
    seg_dataset = ForgeDataset(seg_yaml, split="train")
    seg_batch = SegmentLoader("video", clip_len=4, image_size=32).collate([seg_dataset[0]])
    assert seg_batch.x.shape == (1, 4, 3, 32, 32)

    ano_yaml = _make_dataset(
        tmp_path / "image_ano",
        media="image",
        task="anomaly",
        label_lines=["ano abnormal 0"],
    )
    ano_dataset = ForgeDataset(ano_yaml, split="train")
    ano_batch = AnomalyLoader("image", image_size=32).collate([ano_dataset[0]])
    assert ano_batch.labels["targets"].tolist() == [1.0]


def test_classify_loader_can_read_cached_feature_batches(tmp_path: Path):
    image_yaml = _make_dataset(tmp_path / "image_cached", media="image", task="classify", label_lines=["cls 0"])
    dataset = ForgeDataset(image_yaml, split="train")
    record = dataset[0]
    cache_dir = tmp_path / "feature_cache"
    store = FeatureCacheStore(cache_dir)
    store.write(
        spec={"kind": "test"},
        items=[
            (
                cached_feature_item_key(media_path=record.media_path),
                CachedFeatureItem(
                    mode="final",
                    media="image",
                    split_layer=12,
                    token_state=None,
                    cached_outputs=[torch.randn(768, 2, 2)],
                    height_patches=2,
                    width_patches=2,
                    temporal_tokens=1,
                ),
            )
        ],
        shard_size=8,
    )

    batch = ClassifyLoader("image", image_size=32, feature_cache=store).collate([record])
    assert batch.x.mode == "final"
    assert len(batch.x.cached_outputs) == 1
    assert batch.x.cached_outputs[0].shape == (1, 768, 2, 2)


def test_resolve_label_path_supports_two_component_split_paths(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    label_path = resolve_label_path(dataset_root, {"labels": {"root": "labels"}}, "train/sample.mp4")
    assert label_path == dataset_root / "labels" / "train" / "sample.txt"


def test_classify_loader_raises_actionable_error_for_missing_annotations(tmp_path: Path):
    yaml_path = _make_dataset(tmp_path / "missing_labels", media="image", task="classify", label_lines=["cls 0"])
    (tmp_path / "missing_labels" / "labels" / "train" / "sample.txt").unlink()
    dataset = ForgeDataset(yaml_path, split="train")
    with pytest.raises(ValueError, match="No annotations"):
        ClassifyLoader("image", image_size=32).collate([dataset[0]])


def test_video_backend_auto_prefers_dali_for_offset_reads_when_available(monkeypatch):
    monkeypatch.setattr(video_mod, "_has_dali", lambda: True)
    backend = video_mod._resolve_backend(source=Path("clip.mp4"), video_backend="auto")
    assert backend == "dali"


def test_image_backend_auto_prefers_dali_when_available(monkeypatch):
    monkeypatch.setattr(image_mod, "_has_dali", lambda: True)
    monkeypatch.setattr(image_mod.torch.cuda, "is_available", lambda: True)
    backend = image_mod._resolve_backend(image_backend="auto")
    assert backend == "dali"


def test_read_image_routes_to_dali_when_requested(monkeypatch):
    monkeypatch.setattr(image_mod, "_has_dali", lambda: True)
    monkeypatch.setattr(image_mod.torch.cuda, "is_available", lambda: True)

    calls: dict[str, int] = {"dali": 0}

    def _fake_dali(*args, **kwargs):
        calls["dali"] += 1
        return torch.zeros(3, 16, 16)

    monkeypatch.setattr(image_mod, "_read_image_dali", _fake_dali)
    tensor = image_mod.read_image("sample.jpg", image_size=16, image_backend="dali")
    assert calls["dali"] == 1
    assert tuple(tensor.shape) == (3, 16, 16)


def test_explicit_dali_image_backend_errors_cleanly_when_unavailable(monkeypatch):
    monkeypatch.setattr(image_mod, "_has_dali", lambda: False)
    try:
        image_mod.read_image("sample.jpg", image_backend="dali")
    except RuntimeError as exc:
        assert "DALI" in str(exc)
    else:
        raise AssertionError("Expected explicit dali image backend to fail when DALI is unavailable")


def test_read_video_clip_routes_nonzero_offsets_to_dali_when_requested(monkeypatch):
    monkeypatch.setattr(video_mod, "_has_dali", lambda: True)

    calls: dict[str, int] = {"dali": 0}

    def _fake_dali(*args, **kwargs):
        calls["dali"] += 1
        return torch.zeros(4, 3, 16, 16)

    monkeypatch.setattr(video_mod, "_read_video_clip_dali", _fake_dali)
    clip = video_mod.read_video_clip("clip.mp4", clip_start=5, clip_len=4, stride=1, image_size=16, video_backend="dali")
    assert calls["dali"] == 1
    assert tuple(clip.shape) == (4, 3, 16, 16)


def test_explicit_dali_backend_errors_cleanly_when_unavailable(monkeypatch):
    monkeypatch.setattr(video_mod, "_has_dali", lambda: False)
    try:
        video_mod.read_video_clip("clip.mp4", clip_start=3, clip_len=4, video_backend="dali")
    except RuntimeError as exc:
        assert "DALI" in str(exc)
    else:
        raise AssertionError("Expected explicit dali backend to fail when DALI is unavailable")


def test_convert_cafe_to_forge_generates_splits_and_labels(tmp_path: Path, monkeypatch):
    source = tmp_path / "cafe"
    (source / "normal").mkdir(parents=True, exist_ok=True)
    (source / "anomaly").mkdir(parents=True, exist_ok=True)
    (source / "processed").mkdir(parents=True, exist_ok=True)
    (source / "normal" / "normal_1.mp4").write_bytes(b"normal")
    (source / "anomaly" / "anomaly_1.mp4").write_bytes(b"anomaly")
    (source / "processed" / "clip_manifest.json").write_text(
        """
[
  {"split": "Train", "label": "normal", "clip_name": "train_clip", "source_video": "normal_1.mp4", "frame_start": 0, "frame_end": 3, "start_s": 0.0, "fps": 25.0, "label_frame_start": null, "label_frame_end": null},
  {"split": "Test", "label": "anomaly", "clip_name": "test_clip", "source_video": "anomaly_1.mp4", "frame_start": 10, "frame_end": 13, "start_s": 0.4, "fps": 25.0, "label_frame_start": 11, "label_frame_end": 12}
]
        """.strip(),
        encoding="utf-8",
    )
    (source / "processed" / "frame_labels.json").write_text('{"test_clip": [0, 1, 1, 0]}', encoding="utf-8")

    def _fake_trim(source_path: Path, dest_path: Path, item: dict):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(source_path.read_bytes())

    monkeypatch.setattr(cafe_converter, "_write_trimmed_clip", _fake_trim)
    out = tmp_path / "cafe_forge"
    result = convert_cafe_to_forge(source, out)
    assert Path(result.dataset_yaml).exists()
    assert (out / "splits" / "val.txt").read_text(encoding="utf-8") == (out / "splits" / "test.txt").read_text(encoding="utf-8")
    assert (out / "labels" / "train" / "train_clip.txt").read_text(encoding="utf-8").strip() == "ano normal"
    assert (out / "labels" / "test" / "test_clip.txt").read_text(encoding="utf-8").strip() == "ano abnormal 1 2 0"
