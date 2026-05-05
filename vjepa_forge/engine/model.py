from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from vjepa_forge.cfg.loader import load_model_config
from vjepa_forge.data import ForgeBatch
from vjepa_forge.models.builders import build_backbone, build_head


class BaseForgeModel(nn.Module):
    pass


class ForgeModel(BaseForgeModel):
    def __init__(self, model: str | Path | dict[str, Any], data: dict[str, Any] | None = None) -> None:
        super().__init__()
        if isinstance(model, (str, Path)):
            model_cfg = load_model_config(model)
        else:
            model_cfg = dict(model)
        data_cfg = {} if data is None else dict(data)
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.task = str(model_cfg.get("task", data_cfg.get("task", "classify")))
        self.media = str(model_cfg.get("media", data_cfg.get("media", "image")))
        self.backbone = build_backbone(model_cfg, data_cfg)
        self.head = build_head(self.task, self.media, model_cfg, self.backbone.embed_dim)

    def forward(self, batch: ForgeBatch):
        if batch.media == "image":
            feats = self.backbone.forward_image(batch.x)
        elif batch.media == "video":
            feats = self.backbone.forward_video(batch.x)
        else:
            raise ValueError(f"Unsupported media: {batch.media}")
        return self.head(feats)

    def train(self, mode: bool = True, **kwargs):
        if kwargs:
            from vjepa_forge.tasks import TASK_REGISTRY

            trainer = TASK_REGISTRY[self.task]["train"](self, **kwargs)
            return trainer.run()
        return super().train(mode)

    def val(self, **kwargs):
        from vjepa_forge.tasks import TASK_REGISTRY

        validator = TASK_REGISTRY[self.task]["val"](self, **kwargs)
        return validator.run()

    def predict(self, **kwargs):
        from vjepa_forge.tasks import TASK_REGISTRY

        predictor = TASK_REGISTRY[self.task]["predict"](self, **kwargs)
        return predictor.run()
