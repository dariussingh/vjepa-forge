from pathlib import Path

import torch

from vjepa_forge.configs.loader import load_config
from vjepa_forge.data import RandomVideoDetectionDataset, collate_detection_batch
from vjepa_forge.engine.trainer import build_dataloader, build_model, compute_detection_loss, build_detection_criterion


def test_random_video_detection_dataset_contract():
    dataset = RandomVideoDetectionDataset(length=2, image_size=32, num_frames=2, num_classes=5)
    video, target = dataset[0]
    assert video.shape == (3, 2, 32, 32)
    assert target["video_id"] == "synthetic_0"
    assert len(target["frames"]) == 2
    assert target["frames"][0]["boxes"].shape == (1, 4)


def test_detection_collate_stacks_video_batch():
    dataset = RandomVideoDetectionDataset(length=2, image_size=32, num_frames=2, num_classes=5)
    batch = [dataset[0], dataset[1]]
    inputs, targets = collate_detection_batch(batch)
    assert inputs.shape == (2, 3, 2, 32, 32)
    assert len(targets) == 2


def test_temporal_detection_model_forward_and_loss():
    config = load_config(
        Path("vjepa_forge/configs/detection/imagenet_vid_temporal_detr.yaml"),
        overrides={"data.name": "random_video_detection", "data.image_size": 32, "data.num_frames": 2, "model.num_classes": 5},
    )
    model = build_model(config)
    dataset = RandomVideoDetectionDataset(length=1, image_size=32, num_frames=2, num_classes=5)
    inputs, targets = collate_detection_batch([dataset[0]])
    outputs = model(inputs)
    assert outputs["pred_logits"].shape == (1, 2, 100, 6)
    assert outputs["pred_boxes"].shape == (1, 2, 100, 4)
    criterion = build_detection_criterion(config)
    loss, metrics = compute_detection_loss(config, criterion, outputs, targets)
    assert torch.isfinite(loss)
    assert "loss_ce" in metrics


def test_detection_video_dataloader_dispatches_temporal_recipe():
    config = load_config(
        Path("vjepa_forge/configs/detection/imagenet_vid_temporal_detr.yaml"),
        overrides={"data.name": "random_video_detection", "data.image_size": 32, "data.num_frames": 2, "data.batch_size": 1, "model.num_classes": 5},
    )
    loader = build_dataloader(config)
    inputs, targets = next(iter(loader))
    assert inputs.shape == (1, 3, 2, 32, 32)
    assert len(targets[0]["frames"]) == 2
