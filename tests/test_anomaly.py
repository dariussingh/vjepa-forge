import torch

from vjepa_forge.heads.anomaly.modeling import ExtractedFeatures, build_predictor


def test_build_predictor_vit_patch_and_score_shape():
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
        },
        StubExtractor(),
    )
    past = ExtractedFeatures(pooled=torch.randn(2, 768), tokens=torch.randn(2, 4, 4, 768))
    out = predictor(past.tokens)
    assert out.shape == (2, 4, 4, 768)
