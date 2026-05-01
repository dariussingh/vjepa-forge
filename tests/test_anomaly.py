import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.engine.model import ForgeModel


def test_anomaly_head_emits_binary_scores_for_image_and_video():
    image_model = ForgeModel(
        {
            "task": "anomaly",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
        },
        data={"task": "anomaly", "media": "image", "image_size": 64},
    )
    image_batch = ForgeBatch(
        x=torch.randn(2, 3, 64, 64),
        media="image",
        task="anomaly",
        labels={"targets": torch.tensor([0.0, 1.0])},
        paths=[],
        meta=[],
    )
    assert image_model(image_batch).shape == (2,)

    video_model = ForgeModel(
        {
            "task": "anomaly",
            "media": "video",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_frames": 4,
        },
        data={"task": "anomaly", "media": "video", "image_size": 64, "clip_len": 4},
    )
    video_batch = ForgeBatch(
        x=torch.randn(2, 4, 3, 64, 64),
        media="video",
        task="anomaly",
        labels={"targets": torch.tensor([0.0, 1.0])},
        paths=[],
        meta=[],
    )
    assert video_model(video_batch).shape == (2,)
