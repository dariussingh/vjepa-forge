from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from vjepa_forge.backbones.vjepa21 import BACKBONE_SPECS, VJEPAEnhancedPyramidAdapter, VJEPAImageBackbone


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0), dim=-1)


def build_sine_position_embedding(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_embed = torch.linspace(0, 1, height, device=device, dtype=dtype).view(height, 1).expand(height, width)
    x_embed = torch.linspace(0, 1, width, device=device, dtype=dtype).view(1, width).expand(height, width)
    num_pos_feats = dim // 4
    temperature = 10000
    dim_t = temperature ** (torch.arange(num_pos_feats, device=device, dtype=dtype) / max(num_pos_feats, 1))
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    if pos.shape[-1] < dim:
        pos = F.pad(pos, (0, dim - pos.shape[-1]))
    elif pos.shape[-1] > dim:
        pos = pos[..., :dim]
    return pos.view(1, height * width, dim)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for idx in range(num_layers):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RFDETRDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_feature_levels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(num_feature_levels)]
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memories: list[torch.Tensor],
        pos_embeds: list[torch.Tensor],
        query_embed: torch.Tensor,
    ) -> torch.Tensor:
        q = queries + query_embed
        attn_out, _ = self.self_attn(q, q, queries)
        queries = self.norm1(queries + self.dropout(attn_out))
        for attn, memory, pos in zip(self.cross_attn, memories, pos_embeds, strict=True):
            cross_out, _ = attn(queries + query_embed, memory + pos, memory)
            queries = self.norm2(queries + self.dropout(cross_out))
        ffn_out = self.ffn(queries)
        queries = self.norm3(queries + self.dropout(ffn_out))
        return queries


class RFDETRDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_feature_levels: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RFDETRDecoderLayer(hidden_dim, num_heads, num_feature_levels) for _ in range(num_layers)]
        )

    def forward(
        self,
        queries: torch.Tensor,
        memories: list[torch.Tensor],
        pos_embeds: list[torch.Tensor],
        query_embed: torch.Tensor,
    ) -> list[torch.Tensor]:
        outputs = []
        x = queries
        for layer in self.layers:
            x = layer(x, memories, pos_embeds, query_embed)
            outputs.append(x)
        return outputs


@dataclass
class RFDETRConfig:
    nc: int
    class_names: list[str] | None
    imgsz: int
    in_channels: int
    adapter_channels: int
    neck: dict[str, Any]
    backbone: dict[str, Any]
    hidden_dim: int
    num_queries: int
    num_heads: int
    num_decoder_layers: int


def build_rf_detr_config(config: dict[str, Any], data: dict[str, Any], *, imgsz: int | None = None) -> RFDETRConfig:
    model_cfg = dict(config["model"])
    detector_cfg = dict(config.get("detector", {}))
    nc = int(model_cfg.get("nc") or data["nc"])
    class_names = model_cfg.get("class_names")
    if class_names is None and "names" in data:
        if isinstance(data["names"], dict):
            class_names = [data["names"][i] for i in sorted(data["names"])]
        else:
            class_names = list(data["names"])
    return RFDETRConfig(
        nc=nc,
        class_names=class_names,
        imgsz=int(model_cfg.get("imgsz", 384) if imgsz is None else imgsz),
        in_channels=int(model_cfg.get("in_channels", 3)),
        adapter_channels=int(model_cfg.get("adapter_channels", 256)),
        neck=dict(model_cfg.get("neck", {})),
        backbone=dict(model_cfg["backbone"]),
        hidden_dim=int(detector_cfg.get("hidden_dim", model_cfg.get("adapter_channels", 256))),
        num_queries=int(detector_cfg.get("num_queries", 300)),
        num_heads=int(detector_cfg.get("num_heads", 8)),
        num_decoder_layers=int(detector_cfg.get("num_decoder_layers", 6)),
    )


class FrozenVJEPARFDETR(nn.Module):
    def __init__(self, cfg: RFDETRConfig) -> None:
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
        self.neck = VJEPAEnhancedPyramidAdapter(
            in_channels=BACKBONE_SPECS[self.backbone.name]["embed_dim"],
            out_channels=int(cfg.neck.get("out_channels", cfg.adapter_channels)),
            in_image_channels=cfg.in_channels,
            detail_channels=cfg.neck.get("detail_channels"),
        )
        hidden_dim = cfg.hidden_dim
        self.input_proj = nn.ModuleList(
            [nn.Conv2d(channel, hidden_dim, kernel_size=1) for channel in self.neck.out_channels]
        )
        self.level_embed = nn.Parameter(torch.zeros(len(self.neck.out_channels), hidden_dim))
        self.query_embed = nn.Embedding(cfg.num_queries, hidden_dim)
        self.decoder = RFDETRDecoder(hidden_dim, cfg.num_heads, len(self.neck.out_channels), cfg.num_decoder_layers)
        self.class_head = nn.Linear(hidden_dim, cfg.nc + 1)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.nc = cfg.nc
        self.num_queries = cfg.num_queries
        self.names = cfg.class_names or [str(i) for i in range(cfg.nc)]
        nn.init.normal_(self.level_embed, std=0.02)

    def configure_trainable(self, freeze_backbone: bool, unfreeze_last_n_blocks: int = 0) -> None:
        if freeze_backbone:
            self.backbone.freeze(unfreeze_last_n_blocks=unfreeze_last_n_blocks)
        else:
            self.backbone.unfreeze()
        for parameter in self.neck.parameters():
            parameter.requires_grad = True
        for parameter in self.input_proj.parameters():
            parameter.requires_grad = True
        for parameter in self.decoder.parameters():
            parameter.requires_grad = True
        for parameter in self.class_head.parameters():
            parameter.requires_grad = True
        for parameter in self.box_head.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        features = self.backbone(images)
        pyramid = self.neck(features, images)
        memories: list[torch.Tensor] = []
        pos_embeds: list[torch.Tensor] = []
        for level_idx, feature in enumerate(pyramid):
            projected = self.input_proj[level_idx](feature)
            batch_size, channels, height, width = projected.shape
            memory = projected.flatten(2).transpose(1, 2)
            pos = build_sine_position_embedding(height, width, channels, feature.device, feature.dtype)
            memories.append(memory + self.level_embed[level_idx].view(1, 1, -1))
            pos_embeds.append(pos)
        batch_size = images.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        queries = torch.zeros_like(query_embed)
        decoder_outputs = self.decoder(queries, memories, pos_embeds, query_embed)
        logits = [self.class_head(output) for output in decoder_outputs]
        boxes = [self.box_head(output).sigmoid() for output in decoder_outputs]
        return {"pred_logits": logits[-1], "pred_boxes": boxes[-1], "aux_outputs": [{"pred_logits": l, "pred_boxes": b} for l, b in zip(logits[:-1], boxes[:-1], strict=True)]}


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost: float = 2.0, bbox_cost: float = 5.0, giou_cost: float = 2.0) -> None:
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"].softmax(-1)
        pred_boxes = outputs["pred_boxes"]
        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_idx, target in enumerate(targets):
            tgt_ids = target["labels"]
            tgt_boxes = target["boxes"]
            if tgt_ids.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_boxes.device)
                indices.append((empty, empty))
                continue
            out_prob = pred_logits[batch_idx]
            out_bbox = pred_boxes[batch_idx]
            class_cost = -out_prob[:, tgt_ids]
            bbox_cost = torch.cdist(out_bbox, tgt_boxes, p=1)
            giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))
            cost_matrix = self.class_cost * class_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
            src_idx, tgt_idx = linear_sum_assignment(cost_matrix.cpu())
            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.int64, device=pred_boxes.device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=pred_boxes.device),
                )
            )
        return indices


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel():
                target_classes[batch_idx, src_idx] = targets[batch_idx]["labels"][tgt_idx]
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_boxes: list[torch.Tensor] = []
        tgt_boxes: list[torch.Tensor] = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel():
                src_boxes.append(outputs["pred_boxes"][batch_idx, src_idx])
                tgt_boxes.append(targets[batch_idx]["boxes"][tgt_idx])
        if not src_boxes:
            zero = outputs["pred_boxes"].sum() * 0.0
            return zero, zero
        src_boxes_cat = torch.cat(src_boxes, dim=0)
        tgt_boxes_cat = torch.cat(tgt_boxes, dim=0)
        loss_bbox = F.l1_loss(src_boxes_cat, tgt_boxes_cat, reduction="none").sum() / max(src_boxes_cat.shape[0], 1)
        loss_giou = (1.0 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes_cat), box_cxcywh_to_xyxy(tgt_boxes_cat)))).sum() / max(src_boxes_cat.shape[0], 1)
        return loss_bbox, loss_giou

    def _single_output_loss(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], suffix: str = "") -> dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)
        return {
            f"loss_ce{suffix}": loss_ce,
            f"loss_bbox{suffix}": loss_bbox,
            f"loss_giou{suffix}": loss_giou,
        }

    def forward(self, outputs: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        losses = self._single_output_loss({"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}, targets)
        aux_outputs = outputs.get("aux_outputs", [])
        for idx, aux in enumerate(aux_outputs):
            losses.update(self._single_output_loss(aux, targets, suffix=f"_{idx}"))
        return losses


def prepare_targets(targets: list[dict[str, torch.Tensor]], imgsz: int) -> list[dict[str, torch.Tensor]]:
    prepared: list[dict[str, torch.Tensor]] = []
    for target in targets:
        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes[:, [0, 2]] /= imgsz
            boxes[:, [1, 3]] /= imgsz
            boxes = box_xyxy_to_cxcywh(boxes)
        prepared.append({"boxes": boxes, "labels": target["labels"]})
    return prepared
