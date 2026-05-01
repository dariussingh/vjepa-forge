import torch

from vjepa_forge.data.batching import ForgeBatch
from vjepa_forge.engine.model import ForgeModel


def test_segment_model_supports_image_and_video_modes():
    image_model = ForgeModel(
        {
            "task": "segment",
            "media": "image",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_classes": 3,
        },
        data={"task": "segment", "media": "image", "image_size": 64},
    )
    image_batch = ForgeBatch(
        x=torch.randn(2, 3, 64, 64),
        media="image",
        task="segment",
        labels={"segments": []},
        paths=[],
        meta=[],
    )
    image_out = image_model(image_batch)
    assert image_out.shape == (2, 3, 4, 4)

    video_model = ForgeModel(
        {
            "task": "segment",
            "media": "video",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_frames": 4,
            "num_classes": 3,
        },
        data={"task": "segment", "media": "video", "image_size": 64, "clip_len": 4},
    )
    video_batch = ForgeBatch(
        x=torch.randn(1, 4, 3, 64, 64),
        media="video",
        task="segment",
        labels={"segments": []},
        paths=[],
        meta=[],
    )
    video_out = video_model(video_batch)
    assert video_out["pred_logits"].shape[:2] == (1, 2)
    assert video_out["pred_masks"].shape[:2] == (1, 2)
