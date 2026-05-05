import torch

from vjepa_forge.data.cache import stack_cached_feature_items
from vjepa_forge.models.vjepa import VJEPA21Backbone, VJEPAImageTokenizer, VJEPAVideoTokenizer


def test_image_and_video_tokenizers_enforce_public_batch_shapes():
    image_tokens = VJEPAImageTokenizer()(torch.randn(2, 3, 32, 32))
    video_tokens = VJEPAVideoTokenizer()(torch.randn(2, 4, 3, 32, 32))
    assert image_tokens.shape == (2, 3, 32, 32)
    assert video_tokens.shape == (2, 3, 4, 32, 32)


def test_vjepa21_backbone_routes_image_and_video_branches():
    backbone = VJEPA21Backbone(
        {
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_frames": 4,
        }
    )
    image_features = backbone.forward_image(torch.randn(1, 3, 64, 64))
    video_features = backbone.forward_video(torch.randn(1, 4, 3, 64, 64))
    assert image_features[-1].shape == (1, 768, 4, 4)
    assert video_features[-1].shape == (1, 768, 2, 4, 4)


def test_vjepa21_backbone_final_cache_matches_live_outputs():
    backbone = VJEPA21Backbone(
        {
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 64,
            "num_frames": 4,
        }
    )
    image = torch.randn(1, 3, 64, 64)
    live = backbone.forward_image(image)
    item = backbone.build_cache_item(image, media="image", split_layer=backbone.get_num_layers())
    cached = backbone.forward_cached(stack_cached_feature_items([item]))
    assert len(live) == len(cached)
    assert all(torch.allclose(a, b, atol=1.0e-5, rtol=1.0e-4) for a, b in zip(live, cached, strict=True))
