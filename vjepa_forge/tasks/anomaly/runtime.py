from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vjepa_forge.data.video import read_video_frames_uint8
from vjepa_forge.engine.checkpointing import checkpoint_paths, checkpoint_payload, load_checkpoint, resolve_resume_path, results_csv_rows, save_checkpoint, write_results_csv
from vjepa_forge.engine.optimization import build_scheduler, build_train_settings, normalize_stages, resolve_autoscaled_lr
from vjepa_forge.engine.runtime import setup_runtime
from vjepa_forge.heads.anomaly.modeling import ExtractedFeatures, build_feature_extractor, build_predictor
from vjepa_forge.losses.anomaly import anomaly_future_prediction_loss
from vjepa_forge.metrics.anomaly import roc_auc_score as _roc_auc_score
from vjepa_forge.tasks.anomaly.data import (
    AnomalyExportResult,
    AnomalyPredictResult,
    AnomalyTrainResult,
    AnomalyValidationResult,
    _build_source_record,
    _make_output_root,
    _predict_output_root,
    _repo_root,
    _seed_everything,
)
from vjepa_forge.tasks.anomaly.loaders import _build_eval_loader, _make_loaders
from vjepa_forge.tasks.anomaly.scoring import (
    _build_smoothed_summary,
    _clip_level_metrics,
    _finalize_video_summary,
    _flatten_metric_arrays,
    _threshold_clip_predictions,
    _thresholds_from_smoothed_summary,
    _timing_metrics,
    _write_csv,
    _write_json,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _make_loaders_compat(cfg: dict[str, Any], include_test: bool = True, *, feature_extractor=None, device=None, runtime=None) -> dict[str, Any]:
    try:
        return _make_loaders(cfg, include_test=include_test, feature_extractor=feature_extractor, device=device, runtime=runtime)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return _make_loaders(cfg, include_test=include_test)


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def _build_cfg(config: dict[str, Any], *, action: str) -> dict[str, Any]:
    model_cfg = dict(config["model"])
    if "name" not in model_cfg and "name_full" in model_cfg:
        model_cfg["name"] = model_cfg["name_full"]
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    data_cfg = dict(config["data"])
    dataset_yaml = data_cfg.get("_path") or data_cfg.get("dataset_yaml") or data_cfg.get("path")
    source = config.get("predict", {}).get("source")
    if not dataset_yaml and not (action == "predict" and source):
        raise ValueError("Anomaly runtime requires data._path or data.dataset_yaml")
    output_root = config.get("output", {}).get("root")
    default_workers = max(1, min(8, os.cpu_count() or 1))
    return {
        "action": action,
        "dataset": {
            "dataset_yaml": None if dataset_yaml is None else str(dataset_yaml),
            "image_size": int(data_cfg.get("image_size", 384)),
            "past_frames": int(data_cfg.get("past_frames", data_cfg.get("num_frames", 8))),
            "future_frames": int(data_cfg.get("future_frames", data_cfg.get("num_frames", 8))),
            "stride": int(data_cfg.get("stride", 1)),
            "video_backend": str(data_cfg.get("video_backend", "auto")),
            "feature_cache": str(data_cfg.get("feature_cache", "false")),
            "feature_cache_root": data_cfg.get("feature_cache_root"),
            "feature_cache_build_on_miss": bool(data_cfg.get("feature_cache_build_on_miss", True)),
            "feature_cache_readonly": bool(data_cfg.get("feature_cache_readonly", False)),
            "feature_cache_shard_size": int(data_cfg.get("feature_cache_shard_size", 64)),
            "feature_cache_dtype": str(data_cfg.get("feature_cache_dtype", "fp16")),
            "train_fraction": float(data_cfg.get("train_fraction", 1.0)),
        },
        "model": {
            "name": str(model_cfg.get("name", "vjepa2_1_vit_base_384")),
            "checkpoint": str(backbone_cfg.get("checkpoint")),
            "checkpoint_key": str(backbone_cfg.get("checkpoint_key", "ema_encoder")),
            "predictor_type": str(model_cfg.get("predictor_type", "vit_patch")),
            "hidden_dim": int(model_cfg.get("hidden_dim", 1024)),
            "dropout": float(model_cfg.get("dropout", 0.1)),
            "predictor_embed_dim": int(model_cfg.get("predictor_embed_dim", 768)),
            "predictor_depth": int(model_cfg.get("predictor_depth", 12)),
            "predictor_num_heads": int(model_cfg.get("predictor_num_heads", 12)),
            "predictor_use_rope": bool(model_cfg.get("predictor_use_rope", True)),
            "token_aggregation": str(model_cfg.get("token_aggregation", "topk_mean")),
            "token_topk_fraction": float(model_cfg.get("token_topk_fraction", 0.1)),
        },
        "train": {
            "batch_size": int(config["train"].get("batch_size", 1)),
            "epochs": int(config["train"].get("epochs", 10)),
            "save": bool(config["train"].get("save", True)),
            "save_period": int(config["train"].get("save_period", 0)),
            "resume": config["train"].get("resume", False),
            "project": config["train"].get("project"),
            "name": config["train"].get("name"),
            "exist_ok": bool(config["train"].get("exist_ok", False)),
            "lr_mode": str(config["train"].get("lr_mode", "manual")),
            "lr": float(config["train"].get("lr", 1.0e-4)),
            "reference_batch_size": int(config["train"].get("reference_batch_size", config["train"].get("batch_size", 1))),
            "reference_lr": float(config["train"].get("reference_lr", config["train"].get("lr", 1.0e-4))),
            "lr_scale_rule": str(config["train"].get("lr_scale_rule", "sqrt")),
            "weight_decay": float(config["train"].get("weight_decay", 1.0e-4)),
            "num_workers": int(config["train"].get("num_workers", default_workers)),
            "prefetch_factor": int(config["train"].get("prefetch_factor", 2)),
            "persistent_workers": bool(config["train"].get("persistent_workers", True)),
            "pin_memory": bool(config["train"].get("pin_memory", torch.cuda.is_available())),
            "reader_cache_size": int(config["train"].get("reader_cache_size", 4)),
            "device": str(config["train"].get("device", "cpu")),
            "seed": int(config["train"].get("seed", 7)),
            "save_latest_every_epoch": bool(config["train"].get("save_latest_every_epoch", True)),
            "save_epoch_checkpoints": bool(config["train"].get("save_epoch_checkpoints", False)),
            "scheduler": dict(config["train"].get("scheduler", {})),
            "early_stopping": dict(config["train"].get("early_stopping", {})),
        },
        "eval": {
            "batch_size": int(config["val"].get("batch_size", config["train"].get("batch_size", 1))),
            "num_workers": int(config["val"].get("num_workers", default_workers)),
            "prefetch_factor": int(config["val"].get("prefetch_factor", 2)),
            "persistent_workers": bool(config["val"].get("persistent_workers", True)),
            "pin_memory": bool(config["val"].get("pin_memory", torch.cuda.is_available())),
            "reader_cache_size": int(config["val"].get("reader_cache_size", 4)),
            "threshold_std_multiplier": float(config["val"].get("threshold_std_multiplier", 3.0)),
            "smoothing_window": int(config["val"].get("smoothing_window", 9)),
            "checkpoint_target": str(config["val"].get("checkpoint_target", "best")),
            "checkpoint_path": config["val"].get("checkpoint_path"),
            "split": str(config["val"].get("split", "val" if action == "val" else "test")),
        },
        "predict": {
            "batch_size": int(config.get("predict", {}).get("batch_size", config["val"].get("batch_size", 1))),
            "num_workers": int(config.get("predict", {}).get("num_workers", config["val"].get("num_workers", default_workers))),
            "split": str(config.get("predict", {}).get("split", "test")),
            "threshold": config.get("predict", {}).get("threshold"),
            "visualize": bool(config.get("predict", {}).get("visualize", False)),
            "source": source,
            "output_dir": config.get("predict", {}).get("output_dir"),
        },
        "export": {
            "format": str(config["export"].get("format", "onnx")),
            "output_path": str(config["export"].get("output_path", "anomaly.onnx")),
            "opset": int(config["export"].get("opset", 17)),
            "dynamic_axes": bool(config["export"].get("dynamic_axes", True)),
            "checkpoint_target": str(config["export"].get("checkpoint_target", config["val"].get("checkpoint_target", "best"))),
            "checkpoint_path": config["export"].get("checkpoint_path"),
        },
        "output": {"root": output_root},
        "distributed": dict(config.get("distributed", {})),
    }


def _extract_pair_features(feature_extractor: nn.Module, batch: dict[str, Any], runtime) -> tuple[ExtractedFeatures, ExtractedFeatures]:
    if "past_pooled" in batch:
        # Cached tensors may be stored as fp16/bf16 to save disk; cast to fp32 before
        # passing to the predictor so LayerNorm and other ops always receive full precision.
        # AMP autocast will downcast to bf16/fp16 again for the ops that benefit from it.
        def _load(t: torch.Tensor) -> torch.Tensor:
            return runtime.move_tensor(t).float()

        return (
            ExtractedFeatures(pooled=_load(batch["past_pooled"]), tokens=_load(batch["past_tokens"])),
            ExtractedFeatures(pooled=_load(batch["future_pooled"]), tokens=_load(batch["future_tokens"])),
        )
    past = runtime.move_tensor(batch["past"])
    future = runtime.move_tensor(batch["future"])
    with torch.no_grad():
        with runtime.autocast_context():
            past_feat = feature_extractor(past)
            future_feat = feature_extractor(future)
    return past_feat, future_feat


def _mse_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=1)


def _score_tokens(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, *, tubelet_size: int, aggregation: str, topk_fraction: float) -> torch.Tensor:
    token_errors = ((pred_tokens - target_tokens) ** 2).mean(dim=-1)
    if aggregation == "mean":
        temporal_scores = token_errors.mean(dim=-1)
    elif aggregation == "max":
        temporal_scores = token_errors.max(dim=-1).values
    elif aggregation == "topk_mean":
        k = max(1, int(round(token_errors.size(-1) * topk_fraction)))
        temporal_scores = torch.topk(token_errors, k=k, dim=-1).values.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported token aggregation: {aggregation}")
    return temporal_scores.repeat_interleave(tubelet_size, dim=1)


def _predict_sample_scores(predictor: nn.Module, past_features: ExtractedFeatures, future_features: ExtractedFeatures, model_cfg: dict[str, Any], *, tubelet_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    predictor_type = model_cfg["predictor_type"]
    if predictor_type == "global_mlp":
        pred_future = predictor(past_features.pooled)
        predictor_scores = _mse_score(pred_future, future_features.pooled)
        frozen_scores = _mse_score(past_features.pooled, future_features.pooled)
        frame_count = future_features.tokens.size(1) * tubelet_size
        return predictor_scores.unsqueeze(1).repeat(1, frame_count), frozen_scores.unsqueeze(1).repeat(1, frame_count)
    if predictor_type == "vit_patch":
        pred_tokens = predictor(past_features.tokens)
        predictor_scores = _score_tokens(
            pred_tokens,
            future_features.tokens,
            tubelet_size=tubelet_size,
            aggregation=str(model_cfg.get("token_aggregation", "topk_mean")),
            topk_fraction=float(model_cfg.get("token_topk_fraction", 0.1)),
        )
        past_reference = past_features.tokens.mean(dim=1, keepdim=True).expand_as(future_features.tokens)
        frozen_scores = _score_tokens(
            past_reference,
            future_features.tokens,
            tubelet_size=tubelet_size,
            aggregation=str(model_cfg.get("token_aggregation", "topk_mean")),
            topk_fraction=float(model_cfg.get("token_topk_fraction", 0.1)),
        )
        return predictor_scores, frozen_scores
    raise ValueError(f"Unsupported predictor_type: {predictor_type}")


def _aggregate_scores(loader: DataLoader, predictor: nn.Module, feature_extractor: nn.Module, runtime, desc: str, model_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    by_video: dict[str, dict[str, Any]] = {}
    predictor.eval()
    decode_times: list[float] = []
    model_times: list[float] = []
    for batch in _progress(loader, desc=desc, total=len(loader)):
        decode_times.append(float(batch.get("decode_time", 0.0)))
        model_start = time.perf_counter()
        past_feat, future_feat = _extract_pair_features(feature_extractor, batch, runtime)
        with runtime.autocast_context():
            predictor_scores_t, frozen_scores_t = _predict_sample_scores(
                predictor,
                past_feat,
                future_feat,
                model_cfg,
                tubelet_size=feature_extractor.tubelet_size,
            )
        model_times.append(float(time.perf_counter() - model_start))
        predictor_scores = predictor_scores_t.detach().cpu().numpy()
        frozen_scores = frozen_scores_t.detach().cpu().numpy()
        future_indices = batch["future_indices"].numpy()
        labels = batch.get("future_labels")
        labels_np = labels.numpy() if labels is not None else None
        video_names = batch["video_name"]
        for i, video_name in enumerate(video_names):
            state = by_video.setdefault(video_name, {"predictor_sum": {}, "predictor_count": {}, "frozen_sum": {}, "frozen_count": {}, "labels": {}, "has_labels": False})
            for local_idx, frame_idx in enumerate(future_indices[i]):
                idx = int(frame_idx)
                state["predictor_sum"][idx] = state["predictor_sum"].get(idx, 0.0) + float(predictor_scores[i, local_idx])
                state["predictor_count"][idx] = state["predictor_count"].get(idx, 0) + 1
                state["frozen_sum"][idx] = state["frozen_sum"].get(idx, 0.0) + float(frozen_scores[i, local_idx])
                state["frozen_count"][idx] = state["frozen_count"].get(idx, 0) + 1
            if labels_np is not None:
                state["has_labels"] = True
                for frame_idx, label in zip(future_indices[i], labels_np[i]):
                    state["labels"][int(frame_idx)] = int(label)
    summary = _finalize_video_summary(by_video)
    return summary, _timing_metrics(decode_times=decode_times, model_times=model_times)


def _checkpoint_payload(predictor: nn.Module, cfg: dict[str, Any], *, epoch: int, train_loss: float, val_loss: float, best_val_loss: float, effective_lr: float, checkpoint_kind: str) -> dict[str, Any]:
    return checkpoint_payload(
        model_state=predictor.state_dict(),
        optimizer_state=None,
        scheduler_state=None,
        epoch=epoch,
        global_step=0,
        best_fitness=best_val_loss,
        metrics={
            "train_loss": train_loss,
            "val_loss": val_loss,
            "effective_lr": effective_lr,
        },
        config=cfg,
        task="anomaly",
        media="video",
        checkpoint_kind=checkpoint_kind,
        component="predictor",
        extras={"effective_lr": effective_lr},
    )


def _predictor_state_dict_from_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    state_dict = checkpoint.get("model_state") or checkpoint.get("predictor_state") or checkpoint.get("extras", {}).get("predictor_state")
    if state_dict is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain predictor weights")
    return state_dict


def _resolve_effective_lr(train_cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    lr_mode = str(train_cfg.get("lr_mode", "manual"))
    batch_size = int(train_cfg["batch_size"])
    if lr_mode == "manual":
        effective_lr = float(train_cfg["lr"])
    elif lr_mode == "autoscale":
        reference_batch_size = int(train_cfg["reference_batch_size"])
        reference_lr = float(train_cfg["reference_lr"])
        ratio = batch_size / float(reference_batch_size)
        scale_rule = str(train_cfg.get("lr_scale_rule", "sqrt"))
        if scale_rule == "sqrt":
            effective_lr = reference_lr * math.sqrt(ratio)
        elif scale_rule == "linear":
            effective_lr = reference_lr * ratio
        else:
            raise ValueError(f"Unsupported lr_scale_rule: {scale_rule}")
    else:
        raise ValueError(f"Unsupported lr_mode: {lr_mode}")
    return float(effective_lr), {"lr_mode": lr_mode, "effective_lr": float(effective_lr)}


def _resolve_checkpoint_path(cfg: dict[str, Any], section: str) -> Path:
    explicit_path = cfg[section].get("checkpoint_path")
    output_root = _make_output_root(cfg)
    checkpoint_dir = checkpoint_paths(output_root).weights_dir
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        return path
    target = str(cfg[section].get("checkpoint_target", "best"))
    if target == "best":
        return checkpoint_dir / "best.pt"
    if target in {"latest", "last"}:
        return checkpoint_dir / "last.pt"
    raise ValueError(f"Unsupported checkpoint target: {target}")


class _InferenceWrapper(nn.Module):
    def __init__(self, feature_extractor: nn.Module, predictor: nn.Module, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.model_cfg = model_cfg
        self.tubelet_size = feature_extractor.tubelet_size

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        past_feat = self.feature_extractor(past)
        future_feat = self.feature_extractor(future)
        return _predict_sample_scores(
            self.predictor,
            past_feat,
            future_feat,
            self.model_cfg,
            tubelet_size=self.tubelet_size,
        )


def _render_timeline_strip(
    image: np.ndarray,
    *,
    scores: np.ndarray,
    current_index: int,
    threshold: float,
    labels: np.ndarray | None,
) -> np.ndarray:
    height, width = image.shape[:2]
    strip_h = max(28, height // 10)
    y0 = height - strip_h - 8
    x0 = 8
    x1 = width - 8
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y0 + strip_h), (15, 15, 15), -1)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)
    if labels is not None and labels.size:
        for idx, value in enumerate(labels):
            if int(value) <= 0:
                continue
            px0 = int(x0 + idx * (x1 - x0) / max(labels.size, 1))
            px1 = int(x0 + (idx + 1) * (x1 - x0) / max(labels.size, 1))
            cv2.rectangle(image, (px0, y0), (px1, y0 + strip_h), (40, 60, 180), -1)
    if scores.size:
        vmax = max(float(scores.max()), threshold, 1e-6)
        points = []
        for idx, score in enumerate(scores):
            px = int(x0 + idx * (x1 - x0) / max(scores.size - 1, 1))
            py = int(y0 + strip_h - 4 - (float(score) / vmax) * max(strip_h - 8, 1))
            points.append((px, py))
        if len(points) > 1:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (255, 220, 0), 2)
        tx_y = int(y0 + strip_h - 4 - (float(threshold) / vmax) * max(strip_h - 8, 1))
        cv2.line(image, (x0, tx_y), (x1, tx_y), (0, 0, 255), 1)
        cx = int(x0 + current_index * (x1 - x0) / max(scores.size - 1, 1))
        cv2.line(image, (cx, y0), (cx, y0 + strip_h), (255, 255, 255), 1)
    return image


def _render_prediction_video(
    *,
    source_path: str,
    output_path: Path,
    predictor_scores: list[float],
    threshold: float,
    labels: list[int] | None,
) -> None:
    frames = read_video_frames_uint8(source_path)
    if frames.size == 0:
        raise RuntimeError(f"No frames available for visualization: {source_path}")
    frame_count = min(len(frames), len(predictor_scores))
    frames = frames[:frame_count]
    scores = np.asarray(predictor_scores[:frame_count], dtype=np.float32)
    label_arr = None if labels is None else np.asarray(labels[:frame_count], dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for idx in range(frame_count):
            rgb = frames[idx].copy()
            score = float(scores[idx])
            predicted = score > float(threshold)
            if label_arr is not None and int(label_arr[idx]) > 0:
                overlay = rgb.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (255, 96, 96), -1)
                rgb = cv2.addWeighted(overlay, 0.12, rgb, 0.88, 0.0)
            status = "ANOMALY" if predicted else "NORMAL"
            color = (0, 0, 255) if predicted else (0, 200, 0)
            cv2.putText(rgb, f"score={score:.4f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb, f"threshold={float(threshold):.4f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb, status, (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            rgb = _render_timeline_strip(rgb, scores=scores, current_index=idx, threshold=float(threshold), labels=label_arr)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def train_from_runtime_config(config: dict[str, Any]) -> AnomalyTrainResult:
    cfg = _build_cfg(config, action="train")
    if int(cfg["train"]["num_workers"]) > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    _seed_everything(cfg["train"]["seed"])
    output_root = _make_output_root(cfg)
    paths = checkpoint_paths(output_root)
    reports = output_root / "reports"
    paths.weights_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    feature_extractor = runtime.prepare_module(feature_extractor.eval(), training=False)
    loaders = _make_loaders_compat(cfg, include_test=False, feature_extractor=feature_extractor, device=device, runtime=runtime)
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor), training=True)
    train_settings = build_train_settings(cfg["train"], epochs=int(cfg["train"]["epochs"]), batch_size=int(cfg["train"]["batch_size"]))
    backbone_ref = getattr(feature_extractor, "encoder", feature_extractor)
    optimization_host = type("AnomalyOptimizationHost", (), {"backbone": backbone_ref, "head": predictor})()
    stages = normalize_stages(task="anomaly", model=optimization_host, train_cfg=train_settings, default_epochs=int(cfg["train"]["epochs"]), batch_size=int(cfg["train"]["batch_size"]))
    stage0 = stages[0]
    effective_lr = resolve_autoscaled_lr(stage0.optimizer, batch_size=int(cfg["train"]["batch_size"]))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(predictor.parameters()),
                "lr": effective_lr,
                "initial_lr": effective_lr,
                "weight_decay": float(stage0.optimizer.get("weight_decay", cfg["train"]["weight_decay"])),
                "group_name": "predictor",
            }
        ],
        betas=tuple(float(value) for value in stage0.optimizer.get("betas", (0.9, 0.999))),
        eps=float(stage0.optimizer.get("eps", 1.0e-8)),
    )
    scheduler = build_scheduler(optimizer, stage0, steps_per_epoch=len(loaders["train_loader"]))
    train_rows = [
        {key: (int(value) if key == "epoch" else float(value) if key in {"train_loss", "val_loss", "lr", "best_fitness"} else value) for key, value in row.items()}
        for row in results_csv_rows(paths.results_csv)
    ]
    best_val = math.inf
    start_epoch = 1
    global_step = 0
    resume_path = resolve_resume_path(cfg["train"].get("resume", False), run_dir=output_root)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, resume_path))
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val = float(checkpoint.get("best_fitness", checkpoint.get("best_val_loss", math.inf)))
    best_path = paths.best
    last_path = paths.last
    epoch_timing_rows: list[dict[str, float]] = []
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        predictor.train()
        train_losses: list[float] = []
        train_decode_times: list[float] = []
        train_model_times: list[float] = []
        train_bar = _progress(loaders["train_loader"], desc=f"train {epoch}/{cfg['train']['epochs']}", total=len(loaders["train_loader"]))
        for batch in train_bar:
            train_decode_times.append(float(batch.get("decode_time", 0.0)))
            model_start = time.perf_counter()
            past_feat, future_feat = _extract_pair_features(feature_extractor, batch, runtime)
            with runtime.autocast_context():
                loss, _ = anomaly_future_prediction_loss(predictor, past_feat, future_feat, cfg["model"])
            optimizer.zero_grad(set_to_none=True)
            if runtime.scaler is not None:
                runtime.scaler.scale(loss).backward()
                runtime.scaler.step(optimizer)
                runtime.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step(global_step)
            train_model_times.append(float(time.perf_counter() - model_start))
            train_losses.append(float(loss.item()))
            global_step += 1
            if tqdm is not None:
                train_bar.set_postfix(loss=f"{train_losses[-1]:.5f}")
        predictor.eval()
        val_losses: list[float] = []
        val_decode_times: list[float] = []
        val_model_times: list[float] = []
        val_by_video: dict[str, dict[str, Any]] = {}
        with runtime.inference_context():
            val_bar = _progress(loaders["val_loader"], desc=f"val {epoch}/{cfg['train']['epochs']}", total=len(loaders["val_loader"]))
            for batch in val_bar:
                val_decode_times.append(float(batch.get("decode_time", 0.0)))
                model_start = time.perf_counter()
                past_feat, future_feat = _extract_pair_features(feature_extractor, batch, runtime)
                with runtime.autocast_context():
                    predictor_scores_t, frozen_scores_t = _predict_sample_scores(
                        predictor,
                        past_feat,
                        future_feat,
                        cfg["model"],
                        tubelet_size=feature_extractor.tubelet_size,
                    )
                    val_loss_value = float(anomaly_future_prediction_loss(predictor, past_feat, future_feat, cfg["model"])[0].item())
                val_model_times.append(float(time.perf_counter() - model_start))
                val_losses.append(val_loss_value)
                predictor_scores = predictor_scores_t.detach().cpu().numpy()
                frozen_scores = frozen_scores_t.detach().cpu().numpy()
                future_indices = batch["future_indices"].numpy()
                labels = batch.get("future_labels")
                labels_np = labels.numpy() if labels is not None else None
                for i, video_name in enumerate(batch["video_name"]):
                    state = val_by_video.setdefault(video_name, {"predictor_sum": {}, "predictor_count": {}, "frozen_sum": {}, "frozen_count": {}, "labels": {}, "has_labels": False})
                    for local_idx, frame_idx in enumerate(future_indices[i]):
                        idx = int(frame_idx)
                        state["predictor_sum"][idx] = state["predictor_sum"].get(idx, 0.0) + float(predictor_scores[i, local_idx])
                        state["predictor_count"][idx] = state["predictor_count"].get(idx, 0) + 1
                        state["frozen_sum"][idx] = state["frozen_sum"].get(idx, 0.0) + float(frozen_scores[i, local_idx])
                        state["frozen_count"][idx] = state["frozen_count"].get(idx, 0) + 1
                    if labels_np is not None:
                        state["has_labels"] = True
                        for frame_idx, label_value in zip(future_indices[i], labels_np[i]):
                            state["labels"][int(frame_idx)] = int(label_value)
                if tqdm is not None:
                    val_bar.set_postfix(loss=f"{val_loss_value:.5f}")
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_timings = _timing_metrics(decode_times=train_decode_times, model_times=train_model_times)
        val_timings = _timing_metrics(decode_times=val_decode_times, model_times=val_model_times)
        val_cfg = dict(cfg.get("eval", {}))
        val_summary = _finalize_video_summary(val_by_video)
        val_smoothed = _build_smoothed_summary(val_summary, int(val_cfg.get("smoothing_window", 1)))
        labels_raw, scores_pred_raw = _flatten_metric_arrays(val_summary, "predictor_scores")
        _, scores_frozen_raw = _flatten_metric_arrays(val_summary, "frozen_scores")
        labels_smooth, scores_pred_smooth = _flatten_metric_arrays(val_smoothed, "predictor_scores")
        _, scores_frozen_smooth = _flatten_metric_arrays(val_smoothed, "frozen_scores")
        predictor_threshold, frozen_threshold, calibration_labels, calibration_scores_pred, calibration_scores_frozen = _thresholds_from_smoothed_summary(val_smoothed, {"eval": val_cfg})
        calibration_normals_pred = calibration_scores_pred[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_pred
        calibration_normals_frozen = calibration_scores_frozen[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_frozen
        predictor_clip = _clip_level_metrics(val_smoothed, "predictor_scores", threshold=predictor_threshold, reduction="max")
        frozen_clip = _clip_level_metrics(val_smoothed, "frozen_scores", threshold=frozen_threshold, reduction="max")
        val_metrics = {
            "predictor_frame_auc_raw": _roc_auc_score(labels_raw, scores_pred_raw),
            "frozen_diff_frame_auc_raw": _roc_auc_score(labels_raw, scores_frozen_raw),
            "predictor_frame_auc": _roc_auc_score(labels_smooth, scores_pred_smooth),
            "frozen_diff_frame_auc": _roc_auc_score(labels_smooth, scores_frozen_smooth),
            "predictor_threshold": predictor_threshold,
            "frozen_threshold": frozen_threshold,
            "predictor_val_false_positive_rate": float(np.mean(calibration_normals_pred > predictor_threshold)) if len(calibration_normals_pred) else 0.0,
            "frozen_val_false_positive_rate": float(np.mean(calibration_normals_frozen > frozen_threshold)) if len(calibration_normals_frozen) else 0.0,
            "predictor_clip_auc": predictor_clip["auc"],
            "predictor_clip_accuracy": predictor_clip["accuracy"],
            "predictor_clip_precision": predictor_clip["precision"],
            "predictor_clip_recall": predictor_clip["recall"],
            "predictor_clip_specificity": predictor_clip["specificity"],
            "predictor_clip_f1": predictor_clip["f1"],
            "frozen_diff_clip_auc": frozen_clip["auc"],
            "frozen_diff_clip_accuracy": frozen_clip["accuracy"],
            "frozen_diff_clip_precision": frozen_clip["precision"],
            "frozen_diff_clip_recall": frozen_clip["recall"],
            "frozen_diff_clip_specificity": frozen_clip["specificity"],
            "frozen_diff_clip_f1": frozen_clip["f1"],
        }
        previous_best = best_val
        best_val = min(best_val, val_loss)
        train_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "best_fitness": best_val,
                "avg_train_decode_time": train_timings["avg_decode_time"],
                "avg_train_model_time": train_timings["avg_model_time"],
                "avg_val_decode_time": val_timings["avg_decode_time"],
                "avg_val_model_time": val_timings["avg_model_time"],
                **val_metrics,
            }
        )
        epoch_timing_rows.append({"epoch": epoch, **train_timings, **{f"val_{key}": value for key, value in val_timings.items()}})
        metric_summary = " ".join(
            [
                f"epoch={epoch}",
                f"train_loss={train_loss:.6f}",
                f"val_loss={val_loss:.6f}",
                f"predictor_frame_auc={val_metrics['predictor_frame_auc']:.4f}",
                f"predictor_clip_auc={val_metrics['predictor_clip_auc']:.4f}",
                f"predictor_clip_f1={val_metrics['predictor_clip_f1']:.4f}",
                f"frozen_diff_frame_auc={val_metrics['frozen_diff_frame_auc']:.4f}",
            ]
        )
        if tqdm is not None:
            tqdm.write(metric_summary)
        else:
            print(metric_summary)
        write_results_csv(paths.results_csv, train_rows)
        if cfg["train"]["save"]:
            latest_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val, effective_lr=effective_lr, checkpoint_kind="last")
            latest_payload["optimizer_state"] = optimizer.state_dict()
            latest_payload["scheduler_state"] = scheduler.state_dict()
            latest_payload["global_step"] = global_step
            latest_payload["best_fitness"] = best_val
            save_checkpoint(latest_payload, last_path)
            if val_loss <= previous_best:
                best_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val, effective_lr=effective_lr, checkpoint_kind="best")
                best_payload["optimizer_state"] = optimizer.state_dict()
                best_payload["scheduler_state"] = scheduler.state_dict()
                best_payload["global_step"] = global_step
                best_payload["best_fitness"] = best_val
                save_checkpoint(best_payload, best_path)
            if int(cfg["train"].get("save_period", 0)) > 0 and epoch % int(cfg["train"]["save_period"]) == 0:
                epoch_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val, effective_lr=effective_lr, checkpoint_kind="epoch")
                epoch_payload["optimizer_state"] = optimizer.state_dict()
                epoch_payload["scheduler_state"] = scheduler.state_dict()
                epoch_payload["global_step"] = global_step
                epoch_payload["best_fitness"] = best_val
                save_checkpoint(epoch_payload, paths.weights_dir / f"epoch_{epoch:03d}.pt")
    _write_csv(reports / "train_log.csv", train_rows)
    summary = {
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "effective_lr": effective_lr,
        "run_dir": str(output_root),
        "timings": epoch_timing_rows,
    }
    _write_json(reports / "train_summary.json", summary)
    return AnomalyTrainResult(best_val_loss=best_val, best_checkpoint=str(best_path), last_checkpoint=str(last_path), run_dir=str(output_root))


def _run_eval(config: dict[str, Any], *, split: str) -> tuple[dict[str, Any], Path]:
    cfg = _build_cfg(config, action="val")
    cfg["eval"]["split"] = split
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_root = _make_output_root(cfg)
    reports = output_root / "reports"
    plots = output_root / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    feature_extractor = runtime.prepare_module(feature_extractor.eval(), training=False)
    loaders = _make_loaders_compat(cfg, include_test=True, feature_extractor=feature_extractor, device=device)
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor).eval(), training=False)
    checkpoint_path = _resolve_checkpoint_path(cfg, "eval")
    checkpoint = load_checkpoint(checkpoint_path)
    predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, checkpoint_path))
    predictor.eval()
    summary, timings = _aggregate_scores(
        loaders["val_loader"] if split == "val" else loaders["test_loader"],
        predictor,
        feature_extractor,
        runtime,
        desc=f"score {split}",
        model_cfg=cfg["model"],
    )
    smoothed = _build_smoothed_summary(summary, int(cfg["eval"].get("smoothing_window", 1)))
    calibration_smoothed = smoothed
    if split != "val":
        calibration_summary, _ = _aggregate_scores(
            loaders["val_loader"],
            predictor,
            feature_extractor,
            runtime,
            desc="score val (threshold calibration)",
            model_cfg=cfg["model"],
        )
        calibration_smoothed = _build_smoothed_summary(calibration_summary, int(cfg["eval"].get("smoothing_window", 1)))
    labels_raw, scores_pred_raw = _flatten_metric_arrays(summary, "predictor_scores")
    _, scores_frozen_raw = _flatten_metric_arrays(summary, "frozen_scores")
    labels, scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
    _, scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
    predictor_threshold, frozen_threshold, calibration_labels, calibration_scores_pred, calibration_scores_frozen = _thresholds_from_smoothed_summary(calibration_smoothed, cfg)
    calibration_normals_pred = calibration_scores_pred[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_pred
    calibration_normals_frozen = calibration_scores_frozen[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_frozen
    predictor_clip = _clip_level_metrics(smoothed, "predictor_scores", threshold=predictor_threshold, reduction="max")
    frozen_clip = _clip_level_metrics(smoothed, "frozen_scores", threshold=frozen_threshold, reduction="max")
    metrics = {
        "split": split,
        "predictor_type": cfg["model"]["predictor_type"],
        "predictor_frame_auc_raw": _roc_auc_score(labels_raw, scores_pred_raw),
        "frozen_diff_frame_auc_raw": _roc_auc_score(labels_raw, scores_frozen_raw),
        "predictor_frame_auc": _roc_auc_score(labels, scores_pred),
        "frozen_diff_frame_auc": _roc_auc_score(labels, scores_frozen),
        "predictor_threshold": predictor_threshold,
        "frozen_threshold": frozen_threshold,
        "predictor_val_false_positive_rate": float(np.mean(calibration_normals_pred > predictor_threshold)) if len(calibration_normals_pred) else 0.0,
        "frozen_val_false_positive_rate": float(np.mean(calibration_normals_frozen > frozen_threshold)) if len(calibration_normals_frozen) else 0.0,
        "clip_score_reduction": "max",
        "predictor_clip_auc": predictor_clip["auc"],
        "frozen_diff_clip_auc": frozen_clip["auc"],
        "predictor_clip_accuracy": predictor_clip["accuracy"],
        "predictor_clip_precision": predictor_clip["precision"],
        "predictor_clip_recall": predictor_clip["recall"],
        "predictor_clip_specificity": predictor_clip["specificity"],
        "predictor_clip_f1": predictor_clip["f1"],
        "predictor_clip_confusion_matrix": predictor_clip["confusion_matrix"],
        "predictor_clip_counts": predictor_clip["counts"],
        "frozen_diff_clip_accuracy": frozen_clip["accuracy"],
        "frozen_diff_clip_precision": frozen_clip["precision"],
        "frozen_diff_clip_recall": frozen_clip["recall"],
        "frozen_diff_clip_specificity": frozen_clip["specificity"],
        "frozen_diff_clip_f1": frozen_clip["f1"],
        "frozen_diff_clip_confusion_matrix": frozen_clip["confusion_matrix"],
        "frozen_diff_clip_counts": frozen_clip["counts"],
        "smoothing_window": int(cfg["eval"].get("smoothing_window", 1)),
        "checkpoint_path": str(checkpoint_path),
        **timings,
    }
    report_path = reports / f"{split}_metrics.json"
    _write_json(report_path, metrics)
    _write_json(reports / f"{split}_scores.json", summary)
    _write_json(reports / f"{split}_scores_smoothed.json", smoothed)
    _write_json(
        reports / f"{split}_clip_scores.json",
        {
            "split": split,
            "predictor_threshold": predictor_threshold,
            "frozen_threshold": frozen_threshold,
            "clip_score_reduction": "max",
            "predictor_clips": predictor_clip["clips"],
            "frozen_clips": frozen_clip["clips"],
        },
    )
    return metrics, report_path


def _run_predict(config: dict[str, Any]) -> tuple[dict[str, Any], Path, list[str]]:
    cfg = _build_cfg(config, action="predict")
    source = cfg["predict"].get("source")
    split = None if source else str(cfg["predict"].get("split", "test"))
    if split not in {None, "val", "test"}:
        split = "test"
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_root = _make_output_root(cfg)
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    predict_out = _predict_output_root(cfg, split=split, source=source)
    print(f"Output will be saved to: {predict_out}")
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    feature_extractor = runtime.prepare_module(feature_extractor.eval(), training=False)
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor).eval(), training=False)
    checkpoint_path = _resolve_checkpoint_path(cfg, "eval")
    checkpoint = load_checkpoint(checkpoint_path)
    predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, checkpoint_path))
    predictor.eval()
    if source:
        videos = [_build_source_record(source, video_backend=str(cfg["dataset"].get("video_backend", "auto")))]
        loader = _build_eval_loader(
            videos,
            cfg,
            batch_size=int(cfg["predict"]["batch_size"]),
            num_workers=int(cfg["predict"]["num_workers"]),
            feature_extractor=feature_extractor,
            device=device,
            source=source,
        )
        desc = f"predict:{Path(source).name}"
    else:
        loaders = _make_loaders_compat(cfg, include_test=True, feature_extractor=feature_extractor, device=device)
        videos = loaders["val_videos"] if split == "val" else loaders["test_videos"]
        loader = loaders["val_loader"] if split == "val" else loaders["test_loader"]
        desc = f"predict:{split}"
    summary, timings = _aggregate_scores(loader, predictor, feature_extractor, runtime, desc=desc, model_cfg=cfg["model"])
    smoothed = _build_smoothed_summary(summary, int(cfg["eval"].get("smoothing_window", 1)))
    report_stem = "source" if source else str(split)
    _write_json(reports / f"{report_stem}_predict_scores.json", summary)
    _write_json(reports / f"{report_stem}_predict_scores_smoothed.json", smoothed)
    threshold = cfg["predict"].get("threshold")
    metrics: dict[str, Any] = {
        "split": split,
        "source": source,
        "predictor_type": cfg["model"]["predictor_type"],
        "smoothing_window": int(cfg["eval"].get("smoothing_window", 1)),
        "checkpoint_path": str(checkpoint_path),
        "threshold": None if threshold is None else float(threshold),
        **timings,
    }
    if not source:
        labels_raw, scores_pred_raw = _flatten_metric_arrays(summary, "predictor_scores")
        _, scores_frozen_raw = _flatten_metric_arrays(summary, "frozen_scores")
        labels, scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
        _, scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
        metrics.update(
            {
                "predictor_frame_auc_raw": _roc_auc_score(labels_raw, scores_pred_raw),
                "frozen_diff_frame_auc_raw": _roc_auc_score(labels_raw, scores_frozen_raw),
                "predictor_frame_auc": _roc_auc_score(labels, scores_pred),
                "frozen_diff_frame_auc": _roc_auc_score(labels, scores_frozen),
            }
        )
    if threshold is not None:
        threshold_value = float(threshold)
        predictor_clip = _threshold_clip_predictions(smoothed, "predictor_scores", threshold=threshold_value, reduction="max")
        metrics.update(
            {
                "clip_score_reduction": "max",
                "predictor_thresholded": predictor_clip,
            }
        )
    report_payload = {"summary": smoothed, "metrics": metrics}
    report_path = reports / f"{report_stem}_predict_summary.json"
    _write_json(report_path, report_payload)
    rendered_outputs: list[str] = []
    if bool(cfg["predict"].get("visualize", False)):
        if threshold is None:
            raise ValueError("predict.visualize=true requires predict.threshold=<float>")
        output_dir = cfg["predict"].get("output_dir")
        render_root = (_repo_root() / output_dir).resolve() if output_dir else _predict_output_root(cfg, split=split, source=source)
        for video in videos:
            payload = smoothed["videos"].get(video.name)
            if payload is None:
                continue
            render_path = render_root / f"{video.name}.mp4"
            _render_prediction_video(
                source_path=video.media_path,
                output_path=render_path,
                predictor_scores=payload["predictor_scores"],
                threshold=float(threshold),
                labels=payload.get("labels"),
            )
            rendered_outputs.append(str(render_path))
        metrics["rendered_outputs"] = rendered_outputs
        _write_json(reports / f"{report_stem}_predict_visualization.json", metrics)
    return metrics, report_path, rendered_outputs


def validate_from_runtime_config(config: dict[str, Any]) -> AnomalyValidationResult:
    split = str(config["val"].get("split", "val"))
    if split not in {"val", "test"}:
        split = "val"
    metrics, report_path = _run_eval(config, split=split)
    return AnomalyValidationResult(split=split, metrics=metrics, report_path=str(report_path))


def predict_from_runtime_config(config: dict[str, Any]) -> AnomalyPredictResult:
    split = config["predict"].get("split")
    metrics, report_path, rendered_outputs = _run_predict(config)
    if split not in {"val", "test"}:
        split = None
    return AnomalyPredictResult(split=split, metrics=metrics, report_path=str(report_path), rendered_outputs=rendered_outputs or None)


def export_from_runtime_config(config: dict[str, Any]) -> AnomalyExportResult:
    cfg = _build_cfg(config, action="export")
    if str(cfg["export"]["format"]).lower() != "onnx":
        raise ValueError("Active forge anomaly export currently supports format=onnx only")
    runtime = setup_runtime(device=cfg["train"]["device"], data_cfg={"distributed": cfg.get("distributed", {})})
    device = runtime.device
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = runtime.prepare_module(build_predictor(cfg["model"], feature_extractor).eval(), training=False)
    checkpoint_path = _resolve_checkpoint_path(cfg, "export")
    checkpoint = load_checkpoint(checkpoint_path)
    predictor.load_state_dict(_predictor_state_dict_from_checkpoint(checkpoint, checkpoint_path))
    predictor.eval()
    wrapper = runtime.prepare_module(_InferenceWrapper(feature_extractor, predictor, cfg["model"]).eval(), training=False)
    output_path = Path(cfg["export"]["output_path"])
    if not output_path.is_absolute():
        output_path = (_repo_root() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_past = torch.randn(1, 3, cfg["dataset"]["past_frames"], cfg["dataset"]["image_size"], cfg["dataset"]["image_size"], device=device)
    sample_future = torch.randn(1, 3, cfg["dataset"]["future_frames"], cfg["dataset"]["image_size"], cfg["dataset"]["image_size"], device=device)
    torch.onnx.export(
        wrapper,
        (sample_past, sample_future),
        str(output_path),
        input_names=["past", "future"],
        output_names=["predictor_scores", "frozen_scores"],
        dynamic_axes={
            "past": {0: "batch"},
            "future": {0: "batch"},
            "predictor_scores": {0: "batch"},
            "frozen_scores": {0: "batch"},
        } if cfg["export"]["dynamic_axes"] else None,
        opset_version=int(cfg["export"]["opset"]),
    )
    return AnomalyExportResult(output_path=str(output_path), checkpoint_path=str(checkpoint_path))
