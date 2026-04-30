import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
VJEPA_TUNE_ROOT = ROOT / "vjepa-tune"
if str(VJEPA_TUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_TUNE_ROOT))

from vjepa_tune.backbone import VJEPAFeaturePyramidAdapter, VJEPAImageBackbone


def test_vjepa_image_backbone_outputs_hierarchical_maps():
    model = VJEPAImageBackbone(name="vit_base", checkpoint=None, modality_embedding=False, use_sdpa=False)
    x = torch.randn(1, 3, 384, 384)
    outputs = model(x)

    assert len(outputs) == 4
    for feature in outputs:
        assert feature.shape == (1, 768, 24, 24)


def test_vjepa_feature_adapter_emits_detection_pyramid():
    adapter = VJEPAFeaturePyramidAdapter(in_channels=768, out_channels=128)
    features = [torch.randn(2, 768, 24, 24) for _ in range(4)]

    p3, p4, p5 = adapter(features)

    assert p3.shape == (2, 128, 48, 48)
    assert p4.shape == (2, 128, 24, 24)
    assert p5.shape == (2, 128, 12, 12)
