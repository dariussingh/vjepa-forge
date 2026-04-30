import torch

from vjepa_forge.heads.segmentation import InstanceSegmentationHead, SemanticSegmentationHead


def test_semantic_head_shape():
    model = SemanticSegmentationHead(768, 19)
    x = torch.randn(2, 768, 24, 24)
    y = model(x)
    assert y.shape == (2, 19, 24, 24)


def test_instance_head_shape():
    model = InstanceSegmentationHead(768, 16, 19)
    x = torch.randn(2, 768, 24, 24)
    y = model(x)
    assert y["pred_logits"].shape == (2, 16, 20)
    assert y["pred_masks"].shape == (2, 16, 24, 24)
