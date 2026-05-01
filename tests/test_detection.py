import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.engine.model import ForgeModel


def test_detect_model_emits_image_queries():
    model = ForgeModel(
        {
            "task": "detect",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_classes": 5,
            "head": {"num_queries": 32},
        },
        data={"task": "detect", "media": "image", "image_size": 64},
    )
    batch = ForgeBatch(
        x=torch.randn(2, 3, 64, 64),
        media="image",
        task="detect",
        labels={"detections": []},
        paths=[],
        meta=[],
    )
    outputs = model(batch)
    assert outputs["pred_logits"].shape == (2, 32, 6)
    assert outputs["pred_boxes"].shape == (2, 32, 4)


def test_detect_model_emits_video_queries():
    model = ForgeModel(
        {
            "task": "detect",
            "media": "video",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_frames": 4,
            "num_classes": 5,
            "head": {"num_queries": 16},
        },
        data={"task": "detect", "media": "video", "image_size": 64, "clip_len": 4},
    )
    batch = ForgeBatch(
        x=torch.randn(1, 4, 3, 64, 64),
        media="video",
        task="detect",
        labels={"detections": []},
        paths=[],
        meta=[],
    )
    outputs = model(batch)
    assert outputs["pred_logits"].shape == (1, 2, 16, 6)
    assert outputs["pred_boxes"].shape == (1, 2, 16, 4)
