from __future__ import annotations

"""Shared optimizer, scheduler, staging, and monitor utilities for Forge training."""

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class MonitorSpec:
    metric: str
    mode: str


@dataclass(frozen=True)
class EarlyStoppingSpec:
    enabled: bool
    patience: int
    min_delta: float
    min_epochs: int
    restore_best: bool
    scope: str


@dataclass(frozen=True)
class StageSpec:
    index: int
    name: str
    epochs: int
    freeze: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    monitor: MonitorSpec
    early_stopping: EarlyStoppingSpec


@dataclass
class EarlyStoppingState:
    best: float | None = None
    bad_epochs: int = 0
    stopped: bool = False


def _deepcopy_dict(data: dict[str, Any] | None) -> dict[str, Any]:
    return {} if data is None else {key: value for key, value in data.items()}


def build_train_settings(data_cfg: dict[str, Any], *, epochs: int, batch_size: int) -> dict[str, Any]:
    train_section = dict(data_cfg.get("train", {}))
    merged = dict(train_section)
    for key, value in data_cfg.items():
        if key not in merged:
            merged[key] = value
    merged["epochs"] = int(merged.get("epochs", epochs))
    merged["batch_size"] = int(merged.get("batch_size", batch_size))
    return merged


def resolve_autoscaled_lr(optimizer_cfg: dict[str, Any], *, batch_size: int) -> float:
    lr_mode = str(optimizer_cfg.get("lr_mode", "manual")).lower()
    lr = float(optimizer_cfg.get("lr", 1.0e-4))
    if lr_mode == "manual":
        return lr
    if lr_mode != "autoscale":
        raise ValueError(f"Unsupported lr_mode: {lr_mode}")
    reference_batch_size = max(1, int(optimizer_cfg.get("reference_batch_size", batch_size)))
    reference_lr = float(optimizer_cfg.get("reference_lr", lr))
    ratio = float(batch_size) / float(reference_batch_size)
    scale_rule = str(optimizer_cfg.get("lr_scale_rule", "sqrt")).lower()
    if scale_rule == "sqrt":
        return reference_lr * math.sqrt(ratio)
    if scale_rule == "linear":
        return reference_lr * ratio
    raise ValueError(f"Unsupported lr_scale_rule: {scale_rule}")


def resolve_monitor_spec(*, task: str, model, metric: str | None, mode: str | None) -> MonitorSpec:
    metric_name = str(metric or "auto").lower()
    mode_name = str(mode or "auto").lower()
    if metric_name == "auto":
        if task == "classify":
            metric_name = "top1"
        elif task == "detect":
            metric_name = "box_ap"
        elif task == "segment":
            metric_name = "miou"
        else:
            metric_name = "val_loss"
    if mode_name == "auto":
        mode_name = "min" if metric_name == "val_loss" else "max"
    return MonitorSpec(metric=metric_name, mode=mode_name)


def resolve_early_stopping(stage_cfg: dict[str, Any], *, final_stage: bool) -> EarlyStoppingSpec:
    early_cfg = dict(stage_cfg.get("early_stopping", {}))
    scope = str(early_cfg.get("scope", "final_stage"))
    enabled = bool(early_cfg.get("enabled", final_stage if scope == "final_stage" else True))
    if scope == "final_stage" and not final_stage:
        enabled = False
    return EarlyStoppingSpec(
        enabled=enabled,
        patience=max(1, int(early_cfg.get("patience", 10))),
        min_delta=float(early_cfg.get("min_delta", 0.0)),
        min_epochs=max(0, int(early_cfg.get("min_epochs", 0))),
        restore_best=bool(early_cfg.get("restore_best", True)),
        scope=scope,
    )


def normalize_stages(
    *,
    task: str,
    model,
    train_cfg: dict[str, Any],
    default_epochs: int,
    batch_size: int,
) -> list[StageSpec]:
    global_optimizer = dict(train_cfg.get("optimizer", {}))
    global_scheduler = dict(train_cfg.get("scheduler", {}))
    global_monitor = dict(train_cfg.get("monitor", {}))
    global_early = dict(train_cfg.get("early_stopping", {}))
    explicit_stages = list(train_cfg.get("stages", []))
    if not explicit_stages:
        explicit_stages = [
            {
                "name": "train",
                "epochs": int(train_cfg.get("epochs", default_epochs)),
                "optimizer": {
                    "lr_mode": train_cfg.get("lr_mode", "manual"),
                    "lr": train_cfg.get("lr", 1.0e-4),
                    "reference_batch_size": train_cfg.get("reference_batch_size", batch_size),
                    "reference_lr": train_cfg.get("reference_lr", train_cfg.get("lr", 1.0e-4)),
                    "lr_scale_rule": train_cfg.get("lr_scale_rule", "sqrt"),
                    "weight_decay": train_cfg.get("weight_decay", 1.0e-4),
                },
            }
        ]
    stages: list[StageSpec] = []
    for index, raw_stage in enumerate(explicit_stages):
        stage = dict(raw_stage)
        optimizer_cfg = {**global_optimizer, **dict(stage.get("optimizer", {}))}
        scheduler_cfg = {**global_scheduler, **dict(stage.get("scheduler", {}))}
        monitor_cfg = {**global_monitor, **dict(stage.get("monitor", {}))}
        early_cfg = {**global_early, **dict(stage.get("early_stopping", {}))}
        stage_cfg = {
            "optimizer": optimizer_cfg,
            "scheduler": scheduler_cfg,
            "monitor": monitor_cfg,
            "early_stopping": early_cfg,
        }
        stages.append(
            StageSpec(
                index=index,
                name=str(stage.get("name", f"stage{index + 1}")),
                epochs=max(1, int(stage.get("epochs", default_epochs))),
                freeze=dict(stage.get("freeze", {})),
                optimizer=optimizer_cfg,
                scheduler=scheduler_cfg,
                monitor=resolve_monitor_spec(task=task, model=model, metric=monitor_cfg.get("metric"), mode=monitor_cfg.get("mode")),
                early_stopping=resolve_early_stopping(stage_cfg, final_stage=index == len(explicit_stages) - 1),
            )
        )
    return stages


def apply_stage_freeze(model, stage: StageSpec) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    freeze_cfg = stage.freeze
    if not freeze_cfg:
        return
    freeze_backbone = bool(freeze_cfg.get("backbone", False))
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    backbone_blocks = freeze_cfg.get("backbone_blocks")
    if backbone_blocks is not None:
        total_blocks = int(model.backbone.get_num_layers())
        trainable_from = max(0, total_blocks - int(backbone_blocks))
        patch_embeds = []
        if hasattr(model.backbone, "image_backbone") and hasattr(model.backbone, "video_backbone"):
            patch_embeds.extend(
                [
                    getattr(model.backbone.image_backbone, "patch_embed", None),
                    getattr(model.backbone.video_backbone, "patch_embed", None),
                ]
            )
            block_groups = [
                getattr(model.backbone.image_backbone, "blocks", []),
                getattr(model.backbone.video_backbone, "blocks", []),
            ]
        else:
            patch_embeds.append(getattr(model.backbone, "patch_embed", None))
            block_groups = [getattr(model.backbone, "blocks", [])]
        for patch_embed in patch_embeds:
            if patch_embed is not None:
                for parameter in patch_embed.parameters():
                    parameter.requires_grad = False
        for blocks in block_groups:
            for idx, block in enumerate(blocks):
                trainable = idx >= trainable_from
                for parameter in block.parameters():
                    parameter.requires_grad = trainable


def _parameter_component(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if ".decoder" in name:
        return "decoder"
    if ".adapter" in name or ".input_proj" in name or ".level_embed" in name or ".query_embed" in name or ".cls_towers" in name or ".reg_towers" in name:
        return "neck"
    if "predictor" in name:
        return "predictor"
    return "head"


def _layer_decay_multiplier(name: str, *, num_layers: int, layer_decay: float) -> float:
    if not name.startswith("backbone.") or layer_decay >= 0.9999:
        return 1.0
    layer_id = num_layers + 1
    if ".pos_embed" in name or ".patch_embed" in name:
        layer_id = 0
    elif ".blocks." in name:
        try:
            layer_id = int(name.split(".blocks.")[1].split(".")[0]) + 1
        except Exception:
            layer_id = num_layers
    return float(layer_decay ** (num_layers + 1 - layer_id))


def _no_weight_decay_names(model) -> set[str]:
    if hasattr(model.backbone, "no_weight_decay"):
        names = model.backbone.no_weight_decay()
        if isinstance(names, set):
            return {str(name) for name in names}
    return set()


def build_optimizer(model, stage: StageSpec, *, batch_size: int) -> torch.optim.Optimizer:
    optimizer_cfg = dict(stage.optimizer)
    base_lr = resolve_autoscaled_lr(optimizer_cfg, batch_size=batch_size)
    weight_decay = float(optimizer_cfg.get("weight_decay", 1.0e-4))
    scales = {
        "backbone": float(optimizer_cfg.get("backbone_lr_scale", 0.1)),
        "neck": float(optimizer_cfg.get("neck_lr_scale", 1.0)),
        "decoder": float(optimizer_cfg.get("decoder_lr_scale", 1.0)),
        "head": float(optimizer_cfg.get("head_lr_scale", 1.0)),
        "predictor": float(optimizer_cfg.get("predictor_lr_scale", 1.0)),
    }
    layer_decay = float(optimizer_cfg.get("layer_decay", 1.0))
    no_decay_names = _no_weight_decay_names(model)
    num_layers = int(model.backbone.get_num_layers()) if hasattr(model.backbone, "get_num_layers") else 0
    param_groups: dict[tuple[str, bool, float], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = _parameter_component(name)
        decay = not (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "embed" in name.lower()
            or name.startswith(tuple(f"backbone.{entry}" for entry in no_decay_names))
        )
        layer_mult = _layer_decay_multiplier(name, num_layers=num_layers, layer_decay=layer_decay)
        lr = base_lr * scales.get(component, 1.0) * layer_mult
        key = (component, decay, lr)
        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": weight_decay if decay else 0.0,
                "group_name": f"{component}_{'decay' if decay else 'nodecay'}",
            }
        param_groups[key]["params"].append(parameter)
    betas = tuple(float(value) for value in optimizer_cfg.get("betas", (0.9, 0.999)))
    eps = float(optimizer_cfg.get("eps", 1.0e-8))
    return torch.optim.AdamW(list(param_groups.values()), betas=betas, eps=eps)


class WarmupLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_cfg: dict[str, Any], *, total_steps: int) -> None:
        self.optimizer = optimizer
        self.scheduler_cfg = dict(scheduler_cfg)
        self.total_steps = max(1, int(total_steps))
        self.current_step = 0
        self.base_beta1 = [group.get("betas", (0.9, 0.999))[0] for group in optimizer.param_groups]
        self.step(0)

    def state_dict(self) -> dict[str, Any]:
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.current_step = int(state_dict.get("current_step", 0))
        self.step(self.current_step)

    def _lr_multiplier(self, step_index: int) -> float:
        schedule_type = str(self.scheduler_cfg.get("type", "cosine")).lower()
        warmup_steps = min(self.total_steps, int(self.scheduler_cfg.get("warmup_steps", 0)))
        warmup_start = float(self.scheduler_cfg.get("warmup_start_ratio", 0.1))
        min_lr_ratio = float(self.scheduler_cfg.get("min_lr_ratio", 0.01))
        if warmup_steps > 0 and step_index < warmup_steps:
            progress = float(step_index + 1) / float(max(1, warmup_steps))
            return warmup_start + (1.0 - warmup_start) * progress
        if schedule_type == "constant":
            return 1.0
        if self.total_steps <= warmup_steps:
            return 1.0
        progress = float(step_index - warmup_steps) / float(max(1, self.total_steps - warmup_steps - 1))
        progress = min(max(progress, 0.0), 1.0)
        if schedule_type == "linear":
            return 1.0 + (min_lr_ratio - 1.0) * progress
        if schedule_type == "multistep":
            milestones = [int(step) for step in self.scheduler_cfg.get("milestones", [])]
            gamma = float(self.scheduler_cfg.get("gamma", 0.1))
            drops = sum(1 for milestone in milestones if step_index >= milestone)
            return gamma**drops
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    def step(self, step_index: int | None = None) -> float:
        if step_index is None:
            self.current_step += 1
        else:
            self.current_step = int(step_index)
        multiplier = self._lr_multiplier(self.current_step)
        warmup_steps = min(self.total_steps, int(self.scheduler_cfg.get("warmup_steps", 0)))
        warmup_momentum = self.scheduler_cfg.get("warmup_momentum")
        current_lr = 0.0
        for idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = float(group["initial_lr"]) * multiplier
            current_lr = max(current_lr, float(group["lr"]))
            if warmup_momentum is not None and self.current_step < warmup_steps and "betas" in group:
                beta1 = float(warmup_momentum) + (self.base_beta1[idx] - float(warmup_momentum)) * multiplier
                group["betas"] = (beta1, group["betas"][1])
        return current_lr


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    stage: StageSpec,
    *,
    steps_per_epoch: int,
) -> WarmupLRScheduler:
    scheduler_cfg = dict(stage.scheduler)
    warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0.0))
    scheduler_cfg["warmup_steps"] = min(max(1, stage.epochs * steps_per_epoch), int(round(warmup_epochs * steps_per_epoch))) if warmup_epochs > 0 else 0
    total_steps = max(1, stage.epochs * max(1, steps_per_epoch))
    return WarmupLRScheduler(optimizer, scheduler_cfg, total_steps=total_steps)


def extract_monitor_value(val_result, monitor: MonitorSpec) -> float:
    if monitor.metric == "val_loss":
        return float(val_result.loss)
    metrics = val_result.metrics or {}
    if monitor.metric not in metrics:
        raise KeyError(f"Validation metrics do not include monitor metric '{monitor.metric}'")
    return float(metrics[monitor.metric])


def is_improvement(value: float, best: float | None, monitor: MonitorSpec, *, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if monitor.mode == "min":
        return value < (best - float(min_delta))
    return value > (best + float(min_delta))


def update_early_stopping(
    state: EarlyStoppingState,
    *,
    value: float,
    stage: StageSpec,
    completed_stage_epochs: int,
) -> EarlyStoppingState:
    if not stage.early_stopping.enabled:
        return state
    improved = is_improvement(value, state.best, stage.monitor, min_delta=stage.early_stopping.min_delta)
    if improved:
        state.best = value
        state.bad_epochs = 0
        return state
    bad_epochs = state.bad_epochs + 1
    stopped = completed_stage_epochs >= stage.early_stopping.min_epochs and bad_epochs > stage.early_stopping.patience
    return EarlyStoppingState(best=state.best, bad_epochs=bad_epochs, stopped=stopped)
