from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vjepa_forge.backbones.vjepa21 import (
    BACKBONE_SPECS,
    VJEPAEnhancedPyramidAdapter,
    VJEPAFeaturePyramidAdapter,
    VJEPAImageBackbone,
)


def _import_ultralytics_modules():
    try:
        from ultralytics.nn.modules.head import Detect
        from ultralytics.nn.tasks import BaseModel
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.utils.torch_utils import initialize_weights
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "Failed to import Ultralytics runtime. Ensure ultralytics and its OpenCV dependencies are available."
        ) from exc
    return BaseModel, Detect, v8DetectionLoss, initialize_weights


@dataclass
class ModelConfig:
    nc: int
    class_names: list[str] | None
    imgsz: int
    in_channels: int
    adapter_channels: int
    neck: dict[str, Any]
    reg_max: int
    backbone: dict[str, Any]


def build_model_config(config: dict[str, Any], data: dict[str, Any], *, imgsz: int | None = None) -> ModelConfig:
    model_cfg = dict(config["model"])
    nc = model_cfg.get("nc")
    if nc is None:
        nc = int(data["nc"])
    class_names = model_cfg.get("class_names")
    if class_names is None and "names" in data:
        if isinstance(data["names"], dict):
            class_names = [data["names"][i] for i in sorted(data["names"])]
        else:
            class_names = list(data["names"])
    return ModelConfig(
        nc=nc,
        class_names=class_names,
        imgsz=int(model_cfg.get("imgsz", 384) if imgsz is None else imgsz),
        in_channels=int(model_cfg.get("in_channels", 3)),
        adapter_channels=int(model_cfg.get("adapter_channels", 256)),
        neck=dict(model_cfg.get("neck", {})),
        reg_max=int(model_cfg.get("reg_max", 16)),
        backbone=dict(model_cfg["backbone"]),
    )


BaseModel, Detect, v8DetectionLoss, initialize_weights = _import_ultralytics_modules()


class VJEPADetectionModel(BaseModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        backbone_cfg = dict(cfg.backbone)
        self.backbone = VJEPAImageBackbone(
            name=backbone_cfg.get("name", "vit_base"),
            checkpoint=backbone_cfg.get("checkpoint"),
            checkpoint_key=backbone_cfg.get("checkpoint_key", "ema_encoder"),
            mode=backbone_cfg.get("mode", "image"),
            imgsz=cfg.imgsz,
            patch_size=backbone_cfg.get("patch_size", 16),
            tubelet_size=backbone_cfg.get("tubelet_size", 2),
            use_rope=backbone_cfg.get("use_rope", True),
            use_sdpa=backbone_cfg.get("use_sdpa", True),
            uniform_power=backbone_cfg.get("uniform_power", True),
            modality_embedding=backbone_cfg.get("modality_embedding", True),
            interpolate_rope=backbone_cfg.get("interpolate_rope", True),
        )
        neck_cfg = dict(cfg.neck)
        neck_type = neck_cfg.get("type", "enhanced_p2")
        if neck_type == "legacy":
            self.adapter = VJEPAFeaturePyramidAdapter(
                in_channels=BACKBONE_SPECS[self.backbone.name]["embed_dim"],
                out_channels=cfg.adapter_channels,
            )
        elif neck_type == "enhanced_p2":
            self.adapter = VJEPAEnhancedPyramidAdapter(
                in_channels=BACKBONE_SPECS[self.backbone.name]["embed_dim"],
                out_channels=int(neck_cfg.get("out_channels", cfg.adapter_channels)),
                in_image_channels=cfg.in_channels,
                detail_channels=neck_cfg.get("detail_channels"),
            )
        else:
            raise ValueError(f"Unsupported neck type '{neck_type}'")
        self.detect = Detect(nc=cfg.nc, reg_max=cfg.reg_max, ch=self.adapter.out_channels)
        self.model = nn.ModuleList([self.backbone, self.adapter, self.detect])
        for index, module in enumerate(self.model):
            module.i = index
            module.f = -1
            module.type = module.__class__.__name__
            module.np = sum(parameter.numel() for parameter in module.parameters())
        self.save = []
        self.names = {i: name for i, name in enumerate(cfg.class_names or [str(i) for i in range(cfg.nc)])}
        self.nc = cfg.nc
        self.yaml = {"nc": cfg.nc, "channels": cfg.in_channels, "imgsz": cfg.imgsz}
        self.inplace = True
        self.end2end = False
        self.args = {}
        initialize_weights(self)
        self.stride = self._infer_stride(cfg.in_channels, cfg.imgsz)
        self.detect.stride = self.stride
        self.detect.bias_init()

    def _infer_stride(self, in_channels: int, imgsz: int) -> torch.Tensor:
        was_training = self.training
        self.eval()
        head = self.detect
        head.training = True
        with torch.no_grad():
            feats = self.backbone(torch.zeros(1, in_channels, imgsz, imgsz))
            pyramid = self._build_pyramid(feats, torch.zeros(1, in_channels, imgsz, imgsz))
        stride = torch.tensor([imgsz / feature.shape[-2] for feature in pyramid], dtype=torch.float32)
        self.train(was_training)
        head.training = was_training
        return stride

    def configure_trainable(self, freeze_backbone: bool, unfreeze_last_n_blocks: int = 0) -> None:
        if freeze_backbone:
            self.backbone.freeze(unfreeze_last_n_blocks=unfreeze_last_n_blocks)
        else:
            self.backbone.unfreeze()
        for parameter in self.adapter.parameters():
            parameter.requires_grad = True
        for parameter in self.detect.parameters():
            parameter.requires_grad = True

    def _build_pyramid(self, features: list[torch.Tensor], image: torch.Tensor) -> list[torch.Tensor]:
        if isinstance(self.adapter, VJEPAEnhancedPyramidAdapter):
            return self.adapter(features, image)
        return self.adapter(features)

    def _predict_once(self, x, profile: bool = False, visualize: bool = False, embed=None):
        features = self.backbone(x)
        pyramid = self._build_pyramid(features, x)
        return self.detect(pyramid)

    def init_criterion(self):
        return v8DetectionLoss(self)


def create_vjepa_detection_model(model_cfg: ModelConfig):
    return VJEPADetectionModel(model_cfg)
