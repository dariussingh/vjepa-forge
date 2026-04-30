from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from faster_coco_eval import COCO, COCOeval_faster
from tqdm.auto import tqdm

from .config import resolve_path
from .data import load_dataset_info, make_detection_dataloaders
from .rf_detr import FrozenVJEPARFDETR, HungarianMatcher, SetCriterion, build_rf_detr_config, box_cxcywh_to_xyxy, prepare_targets


def _cfg(config: dict[str, Any], stage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path, int, int, float]:
    ultra_cfg = dict(config.get("ultralytics", {}))
    model_cfg = dict(config["model"])
    imgsz = int(stage.get("imgsz", model_cfg.get("imgsz", 640)))
    batch = int(stage.get("batch", ultra_cfg.get("batch", 16)))
    workers = int(ultra_cfg.get("workers", 8))
    fraction = float(stage.get("fraction", ultra_cfg.get("fraction", 1.0)))
    project_dir = Path(resolve_path(config["run"].get("project", "runs/vjepa-tune"), config["_config_dir"]))
    return ultra_cfg, model_cfg, project_dir, imgsz, batch, fraction


def _iter_named_params(module: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def _should_skip_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "embed" in name.lower()


def _optimizer_group(
    *,
    params: list[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    name: str,
) -> dict[str, Any]:
    return {
        "params": params,
        "lr": lr,
        "initial_lr": lr,
        "weight_decay": weight_decay,
        "group_name": name,
    }


def _build_optimizer(
    model: FrozenVJEPARFDETR,
    config: dict[str, Any],
    stage: dict[str, Any],
) -> torch.optim.Optimizer:
    detector_cfg = dict(config.get("detector", {}))
    optimizer_cfg = dict(detector_cfg.get("optimizer", {}))
    base_lr = float(stage["lr0"])
    weight_decay = float(stage.get("weight_decay", optimizer_cfg.get("weight_decay", 0.001)))
    backbone_lr_scale = float(optimizer_cfg.get("backbone_lr_scale", 0.1))
    neck_lr_scale = float(optimizer_cfg.get("neck_lr_scale", 0.5))
    decoder_lr_scale = float(optimizer_cfg.get("decoder_lr_scale", 1.0))
    head_lr_scale = float(optimizer_cfg.get("head_lr_scale", 1.0))

    backbone_decay: list[torch.nn.Parameter] = []
    backbone_no_decay: list[torch.nn.Parameter] = []
    for name, parameter in _iter_named_params(model.backbone):
        (backbone_no_decay if _should_skip_weight_decay(name, parameter) else backbone_decay).append(parameter)

    neck_decay: list[torch.nn.Parameter] = []
    neck_no_decay: list[torch.nn.Parameter] = []
    for module_name, module in (("neck", model.neck), ("input_proj", model.input_proj)):
        for name, parameter in _iter_named_params(module):
            full_name = f"{module_name}.{name}"
            (neck_no_decay if _should_skip_weight_decay(full_name, parameter) else neck_decay).append(parameter)
    neck_no_decay.extend([model.level_embed])
    neck_no_decay.extend(list(model.query_embed.parameters()))

    decoder_decay: list[torch.nn.Parameter] = []
    decoder_no_decay: list[torch.nn.Parameter] = []
    for name, parameter in _iter_named_params(model.decoder):
        (decoder_no_decay if _should_skip_weight_decay(name, parameter) else decoder_decay).append(parameter)

    head_decay: list[torch.nn.Parameter] = []
    head_no_decay: list[torch.nn.Parameter] = []
    for module_name, module in (("class_head", model.class_head), ("box_head", model.box_head)):
        for name, parameter in _iter_named_params(module):
            full_name = f"{module_name}.{name}"
            (head_no_decay if _should_skip_weight_decay(full_name, parameter) else head_decay).append(parameter)

    param_groups: list[dict[str, Any]] = []
    if backbone_decay:
        param_groups.append(_optimizer_group(params=backbone_decay, lr=base_lr * backbone_lr_scale, weight_decay=weight_decay, name="backbone_decay"))
    if backbone_no_decay:
        param_groups.append(_optimizer_group(params=backbone_no_decay, lr=base_lr * backbone_lr_scale, weight_decay=0.0, name="backbone_no_decay"))
    if neck_decay:
        param_groups.append(_optimizer_group(params=neck_decay, lr=base_lr * neck_lr_scale, weight_decay=weight_decay, name="neck_decay"))
    if neck_no_decay:
        param_groups.append(_optimizer_group(params=neck_no_decay, lr=base_lr * neck_lr_scale, weight_decay=0.0, name="neck_no_decay"))
    if decoder_decay:
        param_groups.append(_optimizer_group(params=decoder_decay, lr=base_lr * decoder_lr_scale, weight_decay=weight_decay, name="decoder_decay"))
    if decoder_no_decay:
        param_groups.append(_optimizer_group(params=decoder_no_decay, lr=base_lr * decoder_lr_scale, weight_decay=0.0, name="decoder_no_decay"))
    if head_decay:
        param_groups.append(_optimizer_group(params=head_decay, lr=base_lr * head_lr_scale, weight_decay=weight_decay, name="head_decay"))
    if head_no_decay:
        param_groups.append(_optimizer_group(params=head_no_decay, lr=base_lr * head_lr_scale, weight_decay=0.0, name="head_no_decay"))

    beta1 = float(optimizer_cfg.get("beta1", 0.9))
    beta2 = float(optimizer_cfg.get("beta2", 0.999))
    eps = float(optimizer_cfg.get("eps", 1e-8))
    return torch.optim.AdamW(param_groups, betas=(beta1, beta2), eps=eps)


def _lr_multiplier(
    *,
    step_index: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    schedule_type: str,
    warmup_start_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step_index < warmup_steps:
        warmup_progress = float(step_index + 1) / float(warmup_steps)
        return warmup_start_ratio + (1.0 - warmup_start_ratio) * warmup_progress
    if total_steps <= warmup_steps:
        return 1.0
    progress = float(step_index - warmup_steps) / float(max(total_steps - warmup_steps - 1, 1))
    progress = min(max(progress, 0.0), 1.0)
    if schedule_type == "linear":
        return 1.0 + (min_lr_ratio - 1.0) * progress
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _set_step_lrs(
    *,
    optimizer: torch.optim.Optimizer,
    step_index: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    schedule_type: str,
    warmup_start_ratio: float,
) -> float:
    mult = _lr_multiplier(
        step_index=step_index,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        schedule_type=schedule_type,
        warmup_start_ratio=warmup_start_ratio,
    )
    current_lr = 0.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(param_group["initial_lr"]) * mult
        current_lr = max(current_lr, float(param_group["lr"]))
    return current_lr


def _build_model(config: dict[str, Any], dataset_info, imgsz: int) -> FrozenVJEPARFDETR:
    data = {"nc": len(dataset_info.names), "names": dataset_info.names}
    model_cfg = build_rf_detr_config(config, data, imgsz=imgsz)
    return FrozenVJEPARFDETR(model_cfg)


def _save_checkpoint(
    *,
    save_path: Path,
    model: FrozenVJEPARFDETR,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": deepcopy(config),
    }
    torch.save(payload, save_path)


def _load_checkpoint(model: FrozenVJEPARFDETR, weight_path: str, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _evaluate_coco(
    *,
    model: FrozenVJEPARFDETR,
    data_loader,
    dataset_info,
    device: torch.device,
    imgsz: int,
) -> dict[str, float]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    category_ids = dataset_info.category_ids or list(range(len(dataset_info.names)))
    progress_bar = tqdm(
        data_loader,
        total=len(data_loader),
        desc="Validation",
        leave=True,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for step, (images, targets) in enumerate(progress_bar, start=1):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs["pred_logits"].softmax(-1)
            boxes = outputs["pred_boxes"]
            scores, labels = logits[..., :-1].max(dim=-1)
            for batch_idx, target in enumerate(targets):
                orig_h, orig_w = target["orig_hw"].tolist()
                image_id = int(Path(target["path"]).stem)
                boxes_xyxy = box_cxcywh_to_xyxy(boxes[batch_idx]).clone()
                boxes_xyxy[:, [0, 2]] *= orig_w
                boxes_xyxy[:, [1, 3]] *= orig_h
                topk = min(100, scores.shape[1])
                top_scores, top_idx = scores[batch_idx].topk(topk)
                top_labels = labels[batch_idx][top_idx]
                top_boxes = boxes_xyxy[top_idx]
                keep = top_scores > 0.05
                for score, cls_idx, box in zip(top_scores[keep], top_labels[keep], top_boxes[keep], strict=True):
                    x0, y0, x1, y1 = box.tolist()
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(category_ids[int(cls_idx)]),
                            "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                            "score": float(score),
                        }
                    )
            progress_bar.set_postfix(
                batches=f"{step}/{len(data_loader)}",
                preds=len(predictions),
            )
    progress_bar.close()

    prediction_json = dataset_info.root / "rf_detr_predictions.json"
    prediction_json.write_text(json.dumps(predictions), encoding="utf-8")
    coco_gt = COCO(str(dataset_info.annotation_json))
    coco_dt = coco_gt.loadRes(str(prediction_json))
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "map50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
    }


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _append_results_row(save_dir: Path, row: dict[str, float | int]) -> None:
    results_path = save_dir / "results.csv"
    header = [
        "epoch",
        "lr",
        "train/loss",
        "train/loss_ce",
        "train/loss_bbox",
        "train/loss_giou",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP75(B)",
        "metrics/mAP_small(B)",
        "metrics/mAP_medium(B)",
        "metrics/mAP_large(B)",
    ]
    if not results_path.exists():
        results_path.write_text(",".join(header) + "\n", encoding="utf-8")
    values = [str(row[key]) for key in header]
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(",".join(values) + "\n")


def run_rf_detr_stage(config: dict[str, Any], stage_name: str, stage: dict[str, Any], init_weights: str | None = None) -> str:
    ultra_cfg, _, project_dir, imgsz, batch_size, fraction = _cfg(config, stage)
    detector_cfg = dict(config.get("detector", {}))
    optimizer_cfg = dict(detector_cfg.get("optimizer", {}))
    scheduler_cfg = dict(detector_cfg.get("scheduler", {}))
    dataset_info = load_dataset_info(config["data"]["dataset_yaml"])
    device = torch.device(f"cuda:{config['run'].get('device', 0)}" if torch.cuda.is_available() else "cpu")
    model = _build_model(config, dataset_info, imgsz).to(device)
    model.configure_trainable(
        freeze_backbone=bool(stage.get("freeze_backbone", True)),
        unfreeze_last_n_blocks=int(stage.get("unfreeze_last_n_blocks", 0)),
    )
    train_loader, val_loader = make_detection_dataloaders(
        dataset_info=dataset_info,
        imgsz=imgsz,
        batch_size=batch_size,
        workers=int(ultra_cfg.get("workers", 8)),
        fraction=fraction,
        fliplr=float(ultra_cfg.get("fliplr", 0.5)),
    )
    optimizer = _build_optimizer(model, config, stage)
    if init_weights:
        _load_checkpoint(model, init_weights)

    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=model.nc,
        matcher=matcher,
        weight_dict={"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0},
    ).to(device)
    run_name = f"{config['run'].get('name', 'run')}-{stage.get('save_dir_name', stage_name)}"
    save_dir = project_dir / run_name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "train.log").write_text("", encoding="utf-8")

    epochs = int(stage["epochs"])
    warmup_epochs = max(0, int(stage.get("warmup_epochs", 0)))
    schedule_type = str(scheduler_cfg.get("type", "cosine")).lower()
    min_lr_ratio = float(stage.get("lrf", scheduler_cfg.get("min_lr_ratio", 0.01)))
    warmup_start_ratio = float(scheduler_cfg.get("warmup_start_ratio", 0.1))
    grad_clip_norm = float(optimizer_cfg.get("grad_clip_norm", 0.1))
    total_steps = max(epochs * len(train_loader), 1)
    warmup_steps = min(total_steps, warmup_epochs * len(train_loader))
    global_step = 0
    best_metric = -math.inf
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    for epoch in range(epochs):
        model.train()

        running = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for step, (images, targets) in enumerate(progress_bar, start=1):
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
            prepared_targets = prepare_targets(targets, imgsz=imgsz)
            lr = _set_step_lrs(
                optimizer=optimizer,
                step_index=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
                schedule_type=schedule_type,
                warmup_start_ratio=warmup_start_ratio,
            )
            outputs = model(images)
            losses = criterion(outputs, prepared_targets)
            loss_ce_total = sum(value for name, value in losses.items() if name.startswith("loss_ce"))
            loss_bbox_total = sum(value for name, value in losses.items() if name.startswith("loss_bbox"))
            loss_giou_total = sum(value for name, value in losses.items() if name.startswith("loss_giou"))
            loss = (
                2.0 * loss_ce_total
                + 5.0 * loss_bbox_total
                + 2.0 * loss_giou_total
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    grad_clip_norm,
                )
            optimizer.step()
            global_step += 1
            running["loss"] += float(loss.detach())
            running["loss_ce"] += float(loss_ce_total.detach())
            running["loss_bbox"] += float(loss_bbox_total.detach())
            running["loss_giou"] += float(loss_giou_total.detach())
            progress_bar.set_postfix(
                lr=f"{lr:.2e}",
                loss=f"{running['loss'] / step:.4f}",
                ce=f"{running['loss_ce'] / step:.4f}",
                bbox=f"{running['loss_bbox'] / step:.4f}",
                giou=f"{running['loss_giou'] / step:.4f}",
            )
        progress_bar.close()

        metrics = _evaluate_coco(model=model, data_loader=val_loader, dataset_info=dataset_info, device=device, imgsz=imgsz)
        current = metrics["map50_95"]
        epoch_row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train/loss": running["loss"] / max(len(train_loader), 1),
            "train/loss_ce": running["loss_ce"] / max(len(train_loader), 1),
            "train/loss_bbox": running["loss_bbox"] / max(len(train_loader), 1),
            "train/loss_giou": running["loss_giou"] / max(len(train_loader), 1),
            "metrics/mAP50(B)": metrics["map50"],
            "metrics/mAP50-95(B)": metrics["map50_95"],
            "metrics/mAP75(B)": metrics["map75"],
            "metrics/mAP_small(B)": metrics["map_small"],
            "metrics/mAP_medium(B)": metrics["map_medium"],
            "metrics/mAP_large(B)": metrics["map_large"],
        }
        _append_results_row(save_dir, epoch_row)
        next_best = current if best_metric == -math.inf else max(best_metric, current)
        summary = (
            f"[{datetime.utcnow().isoformat(timespec='seconds')}] "
            f"epoch {epoch + 1}/{epochs} "
            f"loss={_format_metric(epoch_row['train/loss'])} "
            f"ce={_format_metric(epoch_row['train/loss_ce'])} "
            f"bbox={_format_metric(epoch_row['train/loss_bbox'])} "
            f"giou={_format_metric(epoch_row['train/loss_giou'])} "
            f"mAP50={_format_metric(metrics['map50'])} "
            f"mAP50-95={_format_metric(metrics['map50_95'])} "
            f"mAP75={_format_metric(metrics['map75'])} "
            f"mAPs={_format_metric(metrics['map_small'])} "
            f"mAPm={_format_metric(metrics['map_medium'])} "
            f"mAPl={_format_metric(metrics['map_large'])} "
            f"best={_format_metric(next_best)}"
        )
        print(summary, flush=True)
        with (save_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
        _save_checkpoint(save_path=last_path, model=model, optimizer=optimizer, epoch=epoch, best_metric=next_best, config=config)
        if current > best_metric:
            best_metric = current
            _save_checkpoint(save_path=best_path, model=model, optimizer=optimizer, epoch=epoch, best_metric=best_metric, config=config)
    return str(best_path if best_path.exists() else last_path)


def run_rf_detr_validation(config: dict[str, Any], weights: str) -> dict[str, float]:
    stage = dict(config["schedule"]["stage2"])
    ultra_cfg, _, _, imgsz, batch_size, _ = _cfg(config, stage)
    dataset_info = load_dataset_info(config["data"]["dataset_yaml"])
    device = torch.device(f"cuda:{config['run'].get('device', 0)}" if torch.cuda.is_available() else "cpu")
    model = _build_model(config, dataset_info, imgsz).to(device)
    _load_checkpoint(model, weights)
    _, val_loader = make_detection_dataloaders(
        dataset_info=dataset_info,
        imgsz=imgsz,
        batch_size=batch_size,
        workers=int(ultra_cfg.get("workers", 8)),
        fraction=1.0,
        fliplr=0.0,
    )
    return _evaluate_coco(model=model, data_loader=val_loader, dataset_info=dataset_info, device=device, imgsz=imgsz)
