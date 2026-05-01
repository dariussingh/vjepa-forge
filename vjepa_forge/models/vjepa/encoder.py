from __future__ import annotations

import torch
from torch import nn

from vjepa_forge.backbones import VJEPAImageBackbone, VJEPAVideoBackbone

from .tokenizer_image import VJEPAImageTokenizer
from .tokenizer_video import VJEPAVideoTokenizer


class VJEPA21Backbone(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        backbone_cfg = dict(config.get("backbone", config))
        image_size = int(config.get("image_size", config.get("imgsz", 64)))
        num_frames = int(config.get("num_frames", 8))
        self.image_tokenizer = VJEPAImageTokenizer()
        self.video_tokenizer = VJEPAVideoTokenizer()
        self.image_backbone = VJEPAImageBackbone(
            name=backbone_cfg.get("name", "vit_base"),
            checkpoint=backbone_cfg.get("checkpoint"),
            checkpoint_key=backbone_cfg.get("checkpoint_key", "ema_encoder"),
            imgsz=image_size,
            use_sdpa=backbone_cfg.get("use_sdpa", False),
            modality_embedding=backbone_cfg.get("modality_embedding", False),
        )
        self.video_backbone = VJEPAVideoBackbone(
            name=backbone_cfg.get("name", "vit_base"),
            checkpoint=backbone_cfg.get("checkpoint"),
            checkpoint_key=backbone_cfg.get("checkpoint_key", "ema_encoder"),
            imgsz=image_size,
            num_frames=num_frames,
            use_sdpa=backbone_cfg.get("use_sdpa", False),
            modality_embedding=backbone_cfg.get("modality_embedding", False),
        )
        self.embed_dim = self.image_backbone.embed_dim

    def forward_image(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.image_backbone(self.image_tokenizer(x))

    def forward_video(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.video_backbone(self.video_tokenizer(x))
