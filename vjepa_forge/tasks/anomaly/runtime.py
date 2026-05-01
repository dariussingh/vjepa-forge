from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vjepa_forge.data.forge.dataset import ForgeDataset
from vjepa_forge.data.video import get_video_frame_count, read_video_clip
from vjepa_forge.heads.anomaly.modeling import ExtractedFeatures, build_feature_extractor, build_predictor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class VideoClipRecord:
    name: str
    media_path: str
    frame_count: int
    frame_labels: tuple[int, ...] | None


@dataclass(frozen=True)
class WindowRecord:
    video_name: str
    past_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    future_labels: tuple[int, ...] | None


@dataclass
class AnomalyTrainResult:
    best_val_loss: float
    best_checkpoint: str
    latest_checkpoint: str


@dataclass
class AnomalyValidationResult:
    split: str
    metrics: dict[str, Any]
    report_path: str


@dataclass
class AnomalyPredictResult:
    split: str
    metrics: dict[str, Any]
    report_path: str


@dataclass
class AnomalyExportResult:
    output_path: str
    checkpoint_path: str


class ForgeAnomalyWindowDataset(Dataset):
    def __init__(self, videos: list[VideoClipRecord], windows: list[WindowRecord], image_size: int) -> None:
        self.video_lookup = {video.name: video for video in videos}
        self.windows = windows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        record = self.video_lookup[window.video_name]
        clip = read_video_clip(record.media_path, clip_len=None, image_size=self.image_size)
        past = clip[list(window.past_indices)].permute(1, 0, 2, 3).contiguous()
        future = clip[list(window.future_indices)].permute(1, 0, 2, 3).contiguous()
        sample: dict[str, Any] = {
            "past": past,
            "future": future,
            "video_name": record.name,
            "future_indices": torch.tensor(window.future_indices, dtype=torch.long),
        }
        if window.future_labels is not None:
            sample["future_labels"] = torch.tensor(window.future_labels, dtype=torch.long)
        return sample


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_output_root(cfg: dict[str, Any]) -> Path:
    output_cfg = cfg.get("output", {})
    root = output_cfg.get("root")
    if root:
        path = Path(root)
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        return path
    data_path = Path(cfg["dataset"]["dataset_yaml"])
    return (_repo_root() / "outputs" / "vjepa-forge" / "anomaly" / data_path.parent.name).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_cfg(config: dict[str, Any], *, action: str) -> dict[str, Any]:
    model_cfg = dict(config["model"])
    if "name" not in model_cfg and "name_full" in model_cfg:
        model_cfg["name"] = model_cfg["name_full"]
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    data_cfg = dict(config["data"])
    dataset_yaml = data_cfg.get("_path") or data_cfg.get("dataset_yaml") or data_cfg.get("path")
    if not dataset_yaml:
        raise ValueError("Anomaly runtime requires data._path or data.dataset_yaml")
    output_root = config.get("output", {}).get("root")
    return {
        "dataset": {
            "dataset_yaml": str(dataset_yaml),
            "image_size": int(data_cfg.get("image_size", 384)),
            "past_frames": int(data_cfg.get("past_frames", data_cfg.get("num_frames", 8))),
            "future_frames": int(data_cfg.get("future_frames", data_cfg.get("num_frames", 8))),
            "stride": int(data_cfg.get("stride", 1)),
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
            "lr_mode": str(config["train"].get("lr_mode", "manual")),
            "lr": float(config["train"].get("lr", 1.0e-4)),
            "reference_batch_size": int(config["train"].get("reference_batch_size", config["train"].get("batch_size", 1))),
            "reference_lr": float(config["train"].get("reference_lr", config["train"].get("lr", 1.0e-4))),
            "lr_scale_rule": str(config["train"].get("lr_scale_rule", "sqrt")),
            "weight_decay": float(config["train"].get("weight_decay", 1.0e-4)),
            "num_workers": int(config["train"].get("num_workers", 0)),
            "device": str(config["train"].get("device", "cpu")),
            "seed": int(config["train"].get("seed", 7)),
            "save_latest_every_epoch": bool(config["train"].get("save_latest_every_epoch", True)),
            "save_epoch_checkpoints": bool(config["train"].get("save_epoch_checkpoints", False)),
        },
        "eval": {
            "batch_size": int(config["val"].get("batch_size", 1)),
            "num_workers": int(config["val"].get("num_workers", 0)),
            "threshold_std_multiplier": float(config["val"].get("threshold_std_multiplier", 3.0)),
            "smoothing_window": int(config["val"].get("smoothing_window", 9)),
            "checkpoint_target": str(config["val"].get("checkpoint_target", "best")),
            "checkpoint_path": config["val"].get("checkpoint_path"),
            "split": str(config["val"].get("split", "val" if action == "val" else "test")),
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
    }


def _build_video_records(dataset_yaml: str | Path, split: str) -> list[VideoClipRecord]:
    dataset = ForgeDataset(dataset_yaml, split=split)
    records: list[VideoClipRecord] = []
    for record in dataset.records:
        labels = [0] * get_video_frame_count(record.media_path)
        for annotation in record.annotations:
            if annotation.op != "ano":
                continue
            payload = annotation.payload
            if payload.get("status") != "abnormal":
                continue
            start = max(0, int(payload.get("start_frame", 0)))
            end = min(len(labels) - 1, int(payload.get("end_frame", -1)))
            for idx in range(start, end + 1):
                labels[idx] = 1
        records.append(
            VideoClipRecord(
                name=Path(record.media_path).stem,
                media_path=record.media_path,
                frame_count=len(labels),
                frame_labels=tuple(labels),
            )
        )
    return records


def _build_window_records(videos: list[VideoClipRecord], past_frames: int, future_frames: int, stride: int) -> list[WindowRecord]:
    if past_frames != future_frames:
        raise ValueError("Active anomaly runtime requires past_frames == future_frames")
    total = past_frames + future_frames
    windows: list[WindowRecord] = []
    for record in videos:
        if record.frame_count < total:
            continue
        for start in range(0, record.frame_count - total + 1, stride):
            past = tuple(range(start, start + past_frames))
            future = tuple(range(start + past_frames, start + total))
            future_labels = None if record.frame_labels is None else tuple(record.frame_labels[idx] for idx in future)
            windows.append(
                WindowRecord(
                    video_name=record.name,
                    past_indices=past,
                    future_indices=future,
                    future_labels=future_labels,
                )
            )
    if not windows:
        raise RuntimeError("No anomaly windows could be built from the Forge dataset")
    return windows


def _make_loaders(cfg: dict[str, Any], include_test: bool = True) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    train_videos = _build_video_records(dataset_cfg["dataset_yaml"], split="train")
    val_split = cfg["eval"].get("split", "val")
    val_videos = _build_video_records(dataset_cfg["dataset_yaml"], split=val_split)
    test_videos = _build_video_records(dataset_cfg["dataset_yaml"], split="test") if include_test else []
    common = {
        "past_frames": dataset_cfg["past_frames"],
        "future_frames": dataset_cfg["future_frames"],
        "stride": dataset_cfg["stride"],
    }
    train_windows = _build_window_records(train_videos, **common)
    val_windows = _build_window_records(val_videos, **common)
    test_windows = _build_window_records(test_videos, **common) if include_test else []
    image_size = dataset_cfg["image_size"]
    train_ds = ForgeAnomalyWindowDataset(train_videos, train_windows, image_size)
    val_ds = ForgeAnomalyWindowDataset(val_videos, val_windows, image_size)
    test_ds = ForgeAnomalyWindowDataset(test_videos, test_windows, image_size) if include_test else None
    loaders: dict[str, Any] = {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "train_loader": DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=torch.cuda.is_available()),
        "val_loader": DataLoader(val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=cfg["eval"]["num_workers"], pin_memory=torch.cuda.is_available()),
    }
    if test_ds is not None:
        loaders["test_loader"] = DataLoader(test_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=cfg["eval"]["num_workers"], pin_memory=torch.cuda.is_available())
    return loaders


def _extract_pair_features(feature_extractor: nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[ExtractedFeatures, ExtractedFeatures]:
    past = batch["past"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    with torch.no_grad():
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


def _normal_stats(scores: np.ndarray) -> dict[str, float]:
    return {"mean": float(scores.mean()) if len(scores) else 0.0, "std": float(scores.std()) if len(scores) else 0.0}


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(scores) == 0:
        return scores.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(scores.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _aggregate_scores(loader: DataLoader, predictor: nn.Module, feature_extractor: nn.Module, device: torch.device, desc: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    by_video: dict[str, dict[str, Any]] = {}
    predictor.eval()
    for batch in _progress(loader, desc=desc, total=len(loader)):
        past_feat, future_feat = _extract_pair_features(feature_extractor, batch, device)
        predictor_scores_t, frozen_scores_t = _predict_sample_scores(
            predictor,
            past_feat,
            future_feat,
            model_cfg,
            tubelet_size=feature_extractor.tubelet_size,
        )
        predictor_scores = predictor_scores_t.detach().cpu().numpy()
        frozen_scores = frozen_scores_t.detach().cpu().numpy()
        future_indices = batch["future_indices"].numpy()
        labels = batch.get("future_labels")
        labels_np = labels.numpy() if labels is not None else None
        video_names = batch["video_name"]
        for i, video_name in enumerate(video_names):
            state = by_video.setdefault(video_name, {"predictor_sum": {}, "predictor_count": {}, "frozen_sum": {}, "frozen_count": {}, "labels": {}})
            for local_idx, frame_idx in enumerate(future_indices[i]):
                idx = int(frame_idx)
                state["predictor_sum"][idx] = state["predictor_sum"].get(idx, 0.0) + float(predictor_scores[i, local_idx])
                state["predictor_count"][idx] = state["predictor_count"].get(idx, 0) + 1
                state["frozen_sum"][idx] = state["frozen_sum"].get(idx, 0.0) + float(frozen_scores[i, local_idx])
                state["frozen_count"][idx] = state["frozen_count"].get(idx, 0) + 1
            if labels_np is not None:
                for frame_idx, label in zip(future_indices[i], labels_np[i]):
                    state["labels"][int(frame_idx)] = int(label)
    summary: dict[str, Any] = {"videos": {}}
    for video_name, state in by_video.items():
        frame_ids = sorted(state["predictor_sum"].keys())
        predictor_series = np.asarray([state["predictor_sum"][idx] / state["predictor_count"][idx] for idx in frame_ids], dtype=np.float32)
        frozen_series = np.asarray([state["frozen_sum"][idx] / state["frozen_count"][idx] for idx in frame_ids], dtype=np.float32)
        labels = np.asarray([state["labels"].get(idx, 0) for idx in frame_ids], dtype=np.int64)
        summary["videos"][video_name] = {
            "frame_ids": frame_ids,
            "predictor_scores": predictor_series.tolist(),
            "frozen_scores": frozen_series.tolist(),
            "labels": labels.tolist(),
        }
    return summary


def _flatten_metric_arrays(video_summary: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for video in video_summary["videos"].values():
        scores.extend(video[key])
        labels.extend(video["labels"])
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def _build_smoothed_summary(video_summary: dict[str, Any], smoothing_window: int) -> dict[str, Any]:
    smoothed: dict[str, Any] = {"videos": {}}
    for video_name, payload in video_summary["videos"].items():
        predictor_scores = np.asarray(payload["predictor_scores"], dtype=np.float32)
        frozen_scores = np.asarray(payload["frozen_scores"], dtype=np.float32)
        smoothed["videos"][video_name] = {
            "frame_ids": payload["frame_ids"],
            "predictor_scores": _smooth_scores(predictor_scores, smoothing_window).tolist(),
            "frozen_scores": _smooth_scores(frozen_scores, smoothing_window).tolist(),
            "labels": payload["labels"],
        }
    return smoothed


def _checkpoint_payload(predictor: nn.Module, cfg: dict[str, Any], *, epoch: int, train_loss: float, val_loss: float, best_val_loss: float, effective_lr: float, checkpoint_kind: str) -> dict[str, Any]:
    return {
        "predictor_state": predictor.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "effective_lr": effective_lr,
        "checkpoint_kind": checkpoint_kind,
    }


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
    checkpoint_dir = output_root / "checkpoints"
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        return path
    target = str(cfg[section].get("checkpoint_target", "best"))
    if target == "best":
        return checkpoint_dir / "best_predictor.pt"
    if target == "latest":
        return checkpoint_dir / "latest_predictor.pt"
    raise ValueError(f"Unsupported checkpoint target: {target}")


def train_from_runtime_config(config: dict[str, Any]) -> AnomalyTrainResult:
    cfg = _build_cfg(config, action="train")
    device = _resolve_device(cfg["train"]["device"])
    _seed_everything(cfg["train"]["seed"])
    loaders = _make_loaders(cfg, include_test=False)
    output_root = _make_output_root(cfg)
    checkpoints = output_root / "checkpoints"
    reports = output_root / "reports"
    checkpoints.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = build_predictor(cfg["model"], feature_extractor).to(device)
    effective_lr, _ = _resolve_effective_lr(cfg["train"])
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=effective_lr, weight_decay=cfg["train"]["weight_decay"])
    train_rows: list[dict[str, Any]] = []
    best_val = math.inf
    best_path = checkpoints / "best_predictor.pt"
    latest_path = checkpoints / "latest_predictor.pt"
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        predictor.train()
        train_losses: list[float] = []
        train_bar = _progress(loaders["train_loader"], desc=f"train {epoch}/{cfg['train']['epochs']}", total=len(loaders["train_loader"]))
        for batch in train_bar:
            past_feat, future_feat = _extract_pair_features(feature_extractor, batch, device)
            predictor_scores, _ = _predict_sample_scores(predictor, past_feat, future_feat, cfg["model"], tubelet_size=feature_extractor.tubelet_size)
            loss = predictor_scores.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            if tqdm is not None:
                train_bar.set_postfix(loss=f"{train_losses[-1]:.5f}")
        predictor.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            val_bar = _progress(loaders["val_loader"], desc=f"val {epoch}/{cfg['train']['epochs']}", total=len(loaders["val_loader"]))
            for batch in val_bar:
                past_feat, future_feat = _extract_pair_features(feature_extractor, batch, device)
                predictor_scores, _ = _predict_sample_scores(predictor, past_feat, future_feat, cfg["model"], tubelet_size=feature_extractor.tubelet_size)
                val_loss_value = float(predictor_scores.mean().item())
                val_losses.append(val_loss_value)
                if tqdm is not None:
                    val_bar.set_postfix(loss=f"{val_loss_value:.5f}")
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "effective_lr": effective_lr})
        latest_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=min(best_val, val_loss), effective_lr=effective_lr, checkpoint_kind="latest")
        torch.save(latest_payload, latest_path)
        if val_loss < best_val:
            best_val = val_loss
            best_payload = _checkpoint_payload(predictor, cfg, epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val, effective_lr=effective_lr, checkpoint_kind="best")
            torch.save(best_payload, best_path)
    _write_csv(reports / "train_log.csv", train_rows)
    summary = {"best_val_loss": best_val, "best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path), "effective_lr": effective_lr}
    _write_json(reports / "train_summary.json", summary)
    return AnomalyTrainResult(best_val_loss=best_val, best_checkpoint=str(best_path), latest_checkpoint=str(latest_path))


def _run_eval(config: dict[str, Any], *, split: str) -> tuple[dict[str, Any], Path]:
    cfg = _build_cfg(config, action="val")
    cfg["eval"]["split"] = split
    device = _resolve_device(cfg["train"]["device"])
    loaders = _make_loaders(cfg, include_test=True)
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
    predictor = build_predictor(cfg["model"], feature_extractor).to(device)
    checkpoint_path = _resolve_checkpoint_path(cfg, "eval")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    predictor.load_state_dict(checkpoint["predictor_state"])
    predictor.eval()
    summary = _aggregate_scores(
        loaders["val_loader"] if split == "val" else loaders["test_loader"],
        predictor,
        feature_extractor,
        device,
        desc=f"score {split}",
        model_cfg=cfg["model"],
    )
    smoothed = _build_smoothed_summary(summary, int(cfg["eval"].get("smoothing_window", 1)))
    labels_raw, scores_pred_raw = _flatten_metric_arrays(summary, "predictor_scores")
    _, scores_frozen_raw = _flatten_metric_arrays(summary, "frozen_scores")
    labels, scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
    _, scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
    val_labels, val_scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
    _, val_scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
    pred_stats = _normal_stats(val_scores_pred[val_labels == 0] if np.any(val_labels == 0) else val_scores_pred)
    frozen_stats = _normal_stats(val_scores_frozen[val_labels == 0] if np.any(val_labels == 0) else val_scores_frozen)
    metrics = {
        "split": split,
        "predictor_type": cfg["model"]["predictor_type"],
        "predictor_frame_auc_raw": _roc_auc_score(labels_raw, scores_pred_raw),
        "frozen_diff_frame_auc_raw": _roc_auc_score(labels_raw, scores_frozen_raw),
        "predictor_frame_auc": _roc_auc_score(labels, scores_pred),
        "frozen_diff_frame_auc": _roc_auc_score(labels, scores_frozen),
        "predictor_threshold": float(pred_stats["mean"] + cfg["eval"]["threshold_std_multiplier"] * pred_stats["std"]),
        "frozen_threshold": float(frozen_stats["mean"] + cfg["eval"]["threshold_std_multiplier"] * frozen_stats["std"]),
        "predictor_val_false_positive_rate": float(np.mean(val_scores_pred > (pred_stats["mean"] + cfg["eval"]["threshold_std_multiplier"] * pred_stats["std"]))) if len(val_scores_pred) else 0.0,
        "frozen_val_false_positive_rate": float(np.mean(val_scores_frozen > (frozen_stats["mean"] + cfg["eval"]["threshold_std_multiplier"] * frozen_stats["std"]))) if len(val_scores_frozen) else 0.0,
        "smoothing_window": int(cfg["eval"].get("smoothing_window", 1)),
        "checkpoint_path": str(checkpoint_path),
    }
    report_path = reports / f"{split}_metrics.json"
    _write_json(report_path, metrics)
    _write_json(reports / f"{split}_scores.json", summary)
    _write_json(reports / f"{split}_scores_smoothed.json", smoothed)
    return metrics, report_path


def validate_from_runtime_config(config: dict[str, Any]) -> AnomalyValidationResult:
    split = str(config["val"].get("split", "val"))
    if split not in {"val", "test"}:
        split = "val"
    metrics, report_path = _run_eval(config, split=split)
    return AnomalyValidationResult(split=split, metrics=metrics, report_path=str(report_path))


def predict_from_runtime_config(config: dict[str, Any]) -> AnomalyPredictResult:
    split = str(config["predict"].get("split", "test"))
    if split not in {"val", "test"}:
        split = "test"
    metrics, report_path = _run_eval(config, split=split)
    return AnomalyPredictResult(split=split, metrics=metrics, report_path=str(report_path))


def export_from_runtime_config(config: dict[str, Any]) -> AnomalyExportResult:
    cfg = _build_cfg(config, action="export")
    if str(cfg["export"]["format"]).lower() != "onnx":
        raise ValueError("Active forge anomaly export currently supports format=onnx only")
    device = _resolve_device(cfg["train"]["device"])
    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=cfg["model"]["checkpoint"],
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = build_predictor(cfg["model"], feature_extractor).to(device)
    checkpoint_path = _resolve_checkpoint_path(cfg, "export")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    predictor.load_state_dict(checkpoint["predictor_state"])
    predictor.eval()
    wrapper = _InferenceWrapper(feature_extractor, predictor, cfg["model"]).to(device)
    wrapper.eval()
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
