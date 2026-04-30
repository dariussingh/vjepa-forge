from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from .config import load_config, resolve_path
from .dataset import (
    build_window_records,
    load_dataset_bundle,
    ClipDataset,
    write_manifest,
)
from .modeling import ExtractedFeatures, build_feature_extractor, build_predictor
from .viz import write_timeline_svg


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _load_cfg_from_cli(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args(argv)
    return load_config(resolve_path(args.config, _repo_root()))


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        dynamic_ncols=True,
        leave=False,
    )


def _make_loaders(cfg: dict[str, Any], include_test: bool = True) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg["root"] = resolve_path(dataset_cfg["root"], _repo_root())
    bundle = load_dataset_bundle(dataset_cfg)
    split_train = bundle.train_videos
    split_val = bundle.val_videos
    test_videos = bundle.test_videos if include_test else []

    common = dict(
        past_frames=dataset_cfg["past_frames"],
        future_frames=dataset_cfg["future_frames"],
        stride=dataset_cfg["stride"],
    )
    train_windows = build_window_records(split_train, **common)
    val_windows = build_window_records(split_val, **common)
    test_windows = build_window_records(test_videos, **common) if include_test else []

    image_size = dataset_cfg["image_size"]
    train_ds = ClipDataset(split_train, train_windows, image_size)
    val_ds = ClipDataset(split_val, val_windows, image_size)
    test_ds = ClipDataset(test_videos, test_windows, image_size) if include_test else None

    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    loaders: dict[str, Any] = {
        "dataset_name": bundle.dataset_name,
        "dataset_root": bundle.dataset_root,
        "train_videos": split_train,
        "val_videos": split_val,
        "test_videos": test_videos,
        "train_loader": DataLoader(
            train_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        ),
        "val_loader": DataLoader(
            val_ds,
            batch_size=eval_cfg["batch_size"],
            shuffle=False,
            num_workers=eval_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        ),
    }
    if test_ds is not None:
        loaders["test_loader"] = DataLoader(
            test_ds,
            batch_size=eval_cfg["batch_size"],
            shuffle=False,
            num_workers=eval_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _mse_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=1)


@torch.no_grad()
def _extract_pair_features(
    feature_extractor: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[ExtractedFeatures, ExtractedFeatures]:
    past = batch["past"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    past_feat = feature_extractor(past)
    future_feat = feature_extractor(future)
    return past_feat, future_feat


def _score_tokens(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    tubelet_size: int,
    aggregation: str,
    topk_fraction: float,
) -> torch.Tensor:
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


def _predict_sample_scores(
    predictor: nn.Module,
    past_features: ExtractedFeatures,
    future_features: ExtractedFeatures,
    model_cfg: dict[str, Any],
    *,
    tubelet_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictor_type = model_cfg["predictor_type"]
    if predictor_type == "global_mlp":
        pred_future = predictor(past_features.pooled)
        predictor_scores = _mse_score(pred_future, future_features.pooled)
        frozen_scores = _mse_score(past_features.pooled, future_features.pooled)
        frame_count = future_features.tokens.size(1) * tubelet_size
        return (
            predictor_scores.unsqueeze(1).repeat(1, frame_count),
            frozen_scores.unsqueeze(1).repeat(1, frame_count),
        )

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
    return {
        "mean": float(scores.mean()) if len(scores) else 0.0,
        "std": float(scores.std()) if len(scores) else 0.0,
    }


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
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(scores) == 0:
        return scores.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(scores.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _aggregate_scores(
    loader: DataLoader,
    predictor: nn.Module,
    feature_extractor: nn.Module,
    device: torch.device,
    desc: str,
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
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
            video_state = by_video.setdefault(
                video_name,
                {
                    "predictor_sum": {},
                    "predictor_count": {},
                    "frozen_sum": {},
                    "frozen_count": {},
                    "labels": {},
                },
            )
            for local_idx, frame_idx in enumerate(future_indices[i]):
                frame_idx_int = int(frame_idx)
                video_state["predictor_sum"][frame_idx_int] = (
                    video_state["predictor_sum"].get(frame_idx_int, 0.0) + float(predictor_scores[i, local_idx])
                )
                video_state["predictor_count"][frame_idx_int] = video_state["predictor_count"].get(frame_idx_int, 0) + 1
                video_state["frozen_sum"][frame_idx_int] = (
                    video_state["frozen_sum"].get(frame_idx_int, 0.0) + float(frozen_scores[i, local_idx])
                )
                video_state["frozen_count"][frame_idx_int] = video_state["frozen_count"].get(frame_idx_int, 0) + 1
            if labels_np is not None:
                for frame_idx, label in zip(future_indices[i], labels_np[i]):
                    video_state["labels"][int(frame_idx)] = int(label)

    summary: dict[str, Any] = {"videos": {}}
    for video_name, state in by_video.items():
        frame_ids = sorted(state["predictor_sum"].keys())
        predictor_series = np.asarray(
            [state["predictor_sum"][idx] / state["predictor_count"][idx] for idx in frame_ids],
            dtype=np.float32,
        )
        frozen_series = np.asarray(
            [state["frozen_sum"][idx] / state["frozen_count"][idx] for idx in frame_ids],
            dtype=np.float32,
        )
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
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


def _build_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    output_root = resolve_path(cfg["output"]["root"], _repo_root())
    return {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "reports": output_root / "reports",
        "plots": output_root / "plots",
    }


def _resolve_effective_lr(train_cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    lr_mode = str(train_cfg.get("lr_mode", "manual"))
    batch_size = int(train_cfg["batch_size"])
    if lr_mode == "manual":
        effective_lr = float(train_cfg["lr"])
    elif lr_mode == "autoscale":
        reference_batch_size = int(train_cfg["reference_batch_size"])
        reference_lr = float(train_cfg["reference_lr"])
        if reference_batch_size <= 0:
            raise ValueError("reference_batch_size must be > 0")
        scale_rule = str(train_cfg.get("lr_scale_rule", "sqrt"))
        ratio = batch_size / float(reference_batch_size)
        if scale_rule == "sqrt":
            effective_lr = reference_lr * math.sqrt(ratio)
        elif scale_rule == "linear":
            effective_lr = reference_lr * ratio
        else:
            raise ValueError(f"Unsupported lr_scale_rule: {scale_rule}")
    else:
        raise ValueError(f"Unsupported lr_mode: {lr_mode}")

    metadata = {
        "lr_mode": lr_mode,
        "effective_lr": float(effective_lr),
        "reference_batch_size": int(train_cfg.get("reference_batch_size", batch_size)),
        "reference_lr": float(train_cfg.get("reference_lr", train_cfg["lr"])),
        "lr_scale_rule": str(train_cfg.get("lr_scale_rule", "sqrt")),
    }
    return float(effective_lr), metadata


def _checkpoint_payload(
    predictor: nn.Module,
    cfg: dict[str, Any],
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    effective_lr: float,
    checkpoint_kind: str,
) -> dict[str, Any]:
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


def _resolve_eval_checkpoint_path(cfg: dict[str, Any]) -> Path:
    eval_cfg = cfg["eval"]
    explicit_path = eval_cfg.get("checkpoint_path")
    if explicit_path:
        return resolve_path(explicit_path, _repo_root())

    checkpoint_target = str(eval_cfg.get("checkpoint_target", "best"))
    checkpoint_dir = resolve_path(cfg["output"]["root"], _repo_root()) / "checkpoints"
    if checkpoint_target == "best":
        return checkpoint_dir / "best_predictor.pt"
    if checkpoint_target == "latest":
        return checkpoint_dir / "latest_predictor.pt"
    raise ValueError(f"Unsupported eval.checkpoint_target: {checkpoint_target}")


def main_prepare() -> None:
    cfg = _load_cfg_from_cli()
    loaders = _make_loaders(cfg, include_test=True)
    paths = _build_paths(cfg)
    manifest_path = paths["reports"] / "dataset_manifest.json"
    write_manifest(
        manifest_path,
        loaders["dataset_name"],
        loaders["dataset_root"],
        loaders["train_videos"],
        loaders["val_videos"],
        loaders["test_videos"],
    )
    print(f"Dataset: {loaders['dataset_name']}")
    print(f"Dataset root: {loaders['dataset_root']}")
    print(
        "Split summary: "
        f"train_videos={len(loaders['train_videos'])} "
        f"val_videos={len(loaders['val_videos'])} "
        f"test_videos={len(loaders['test_videos'])}"
    )
    print(
        "Window summary: "
        f"train_windows={len(loaders['train_loader'].dataset)} "
        f"val_windows={len(loaders['val_loader'].dataset)} "
        f"test_windows={len(loaders['test_loader'].dataset)}"
    )
    print(f"Wrote dataset manifest to {manifest_path}")


def main_train() -> None:
    cfg = _load_cfg_from_cli()
    train_cfg = cfg["train"]
    device = _resolve_device(train_cfg["device"])
    _seed_everything(train_cfg["seed"])
    loaders = _make_loaders(cfg, include_test=False)
    paths = _build_paths(cfg)
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=resolve_path(cfg["model"]["checkpoint"], _repo_root()),
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = build_predictor(cfg["model"], feature_extractor).to(device)
    effective_lr, lr_metadata = _resolve_effective_lr(train_cfg)
    print(
        f"Learning rate: mode={lr_metadata['lr_mode']} effective_lr={effective_lr:.6g} "
        f"batch_size={train_cfg['batch_size']}"
    )

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=effective_lr,
        weight_decay=train_cfg["weight_decay"],
    )

    train_rows: list[dict[str, Any]] = []
    best_val = math.inf
    best_path = paths["checkpoints"] / "best_predictor.pt"
    latest_path = paths["checkpoints"] / "latest_predictor.pt"

    for epoch in range(1, train_cfg["epochs"] + 1):
        predictor.train()
        train_losses: list[float] = []
        train_bar = _progress(
            loaders["train_loader"],
            desc=f"train epoch {epoch}/{train_cfg['epochs']}",
            total=len(loaders["train_loader"]),
        )
        for batch in train_bar:
            with torch.no_grad():
                past_feat, future_feat = _extract_pair_features(feature_extractor, batch, device)
            predictor_scores, _ = _predict_sample_scores(
                predictor,
                past_feat,
                future_feat,
                cfg["model"],
                tubelet_size=feature_extractor.tubelet_size,
            )
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
            val_bar = _progress(
                loaders["val_loader"],
                desc=f"val epoch {epoch}/{train_cfg['epochs']}",
                total=len(loaders["val_loader"]),
            )
            for batch in val_bar:
                past_feat, future_feat = _extract_pair_features(feature_extractor, batch, device)
                predictor_scores, _ = _predict_sample_scores(
                    predictor,
                    past_feat,
                    future_feat,
                    cfg["model"],
                    tubelet_size=feature_extractor.tubelet_size,
                )
                val_loss_value = float(predictor_scores.mean().item())
                val_losses.append(val_loss_value)
                if tqdm is not None:
                    val_bar.set_postfix(loss=f"{val_loss_value:.5f}")

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "effective_lr": effective_lr,
            }
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if train_cfg.get("save_latest_every_epoch", True):
            latest_payload = _checkpoint_payload(
                predictor,
                cfg,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=min(best_val, val_loss),
                effective_lr=effective_lr,
                checkpoint_kind="latest",
            )
            torch.save(latest_payload, latest_path)

        if val_loss < best_val:
            best_val = val_loss
            best_payload = _checkpoint_payload(
                predictor,
                cfg,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val,
                effective_lr=effective_lr,
                checkpoint_kind="best",
            )
            torch.save(best_payload, best_path)

        if train_cfg.get("save_epoch_checkpoints", False):
            epoch_path = paths["checkpoints"] / f"epoch_{epoch:03d}.pt"
            epoch_payload = _checkpoint_payload(
                predictor,
                cfg,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val,
                effective_lr=effective_lr,
                checkpoint_kind="epoch",
            )
            torch.save(epoch_payload, epoch_path)

    _write_csv(paths["reports"] / "train_log.csv", train_rows)
    summary = {
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "effective_lr": effective_lr,
        "lr_mode": lr_metadata["lr_mode"],
    }
    _save_json(paths["reports"] / "train_summary.json", summary)
    print(f"Saved best predictor checkpoint to {best_path}")
    print(f"Saved latest predictor checkpoint to {latest_path}")


def main_eval() -> None:
    cfg = _load_cfg_from_cli()
    eval_cfg = cfg["eval"]
    device = _resolve_device(cfg["train"]["device"])
    loaders = _make_loaders(cfg, include_test=True)
    paths = _build_paths(cfg)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    feature_extractor = build_feature_extractor(
        model_name=cfg["model"]["name"],
        checkpoint_path=resolve_path(cfg["model"]["checkpoint"], _repo_root()),
        checkpoint_key=cfg["model"]["checkpoint_key"],
        num_frames=cfg["dataset"]["past_frames"],
        image_size=cfg["dataset"]["image_size"],
        device=device,
    )
    predictor = build_predictor(cfg["model"], feature_extractor).to(device)
    predictor_ckpt = _resolve_eval_checkpoint_path(cfg)
    checkpoint = torch.load(predictor_ckpt, map_location="cpu")
    predictor.load_state_dict(checkpoint["predictor_state"])
    predictor.eval()
    print(
        f"Loaded checkpoint: {predictor_ckpt} "
        f"(kind={checkpoint.get('checkpoint_kind', 'unknown')} epoch={checkpoint.get('epoch', 'unknown')})"
    )

    print("Running validation scoring...")
    val_summary = _aggregate_scores(
        loaders["val_loader"],
        predictor,
        feature_extractor,
        device,
        desc="score val",
        model_cfg=cfg["model"],
    )
    print("Running test scoring...")
    test_summary = _aggregate_scores(
        loaders["test_loader"],
        predictor,
        feature_extractor,
        device,
        desc="score test",
        model_cfg=cfg["model"],
    )

    smoothing_window = int(eval_cfg.get("smoothing_window", 1))
    val_smoothed = _build_smoothed_summary(val_summary, smoothing_window)
    test_smoothed = _build_smoothed_summary(test_summary, smoothing_window)

    _, val_scores_pred = _flatten_metric_arrays(val_smoothed, "predictor_scores")
    _, val_scores_frozen = _flatten_metric_arrays(val_smoothed, "frozen_scores")
    pred_stats = _normal_stats(val_scores_pred)
    frozen_stats = _normal_stats(val_scores_frozen)
    pred_threshold = pred_stats["mean"] + eval_cfg["threshold_std_multiplier"] * pred_stats["std"]
    frozen_threshold = frozen_stats["mean"] + eval_cfg["threshold_std_multiplier"] * frozen_stats["std"]

    test_labels_raw, test_scores_pred_raw = _flatten_metric_arrays(test_summary, "predictor_scores")
    _, test_scores_frozen_raw = _flatten_metric_arrays(test_summary, "frozen_scores")
    test_labels, test_scores_pred = _flatten_metric_arrays(test_smoothed, "predictor_scores")
    _, test_scores_frozen = _flatten_metric_arrays(test_smoothed, "frozen_scores")

    metrics = {
        "dataset_name": loaders["dataset_name"],
        "predictor_type": cfg["model"]["predictor_type"],
        "predictor_frame_auc_raw": _roc_auc_score(test_labels_raw, test_scores_pred_raw),
        "frozen_diff_frame_auc_raw": _roc_auc_score(test_labels_raw, test_scores_frozen_raw),
        "predictor_frame_auc": _roc_auc_score(test_labels, test_scores_pred),
        "frozen_diff_frame_auc": _roc_auc_score(test_labels, test_scores_frozen),
        "predictor_threshold": float(pred_threshold),
        "frozen_threshold": float(frozen_threshold),
        "predictor_val_false_positive_rate": float(np.mean(val_scores_pred > pred_threshold)),
        "frozen_val_false_positive_rate": float(np.mean(val_scores_frozen > frozen_threshold)),
        "smoothing_window": smoothing_window,
    }
    _save_json(paths["reports"] / "metrics.json", metrics)
    _save_json(paths["reports"] / "val_scores.json", val_summary)
    _save_json(paths["reports"] / "test_scores.json", test_summary)
    _save_json(paths["reports"] / "val_scores_smoothed.json", val_smoothed)
    _save_json(paths["reports"] / "test_scores_smoothed.json", test_smoothed)

    timeline_rows: list[dict[str, Any]] = []
    for video_name, payload in test_smoothed["videos"].items():
        predictor_scores = np.asarray(payload["predictor_scores"], dtype=np.float32)
        frozen_scores = np.asarray(payload["frozen_scores"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        timeline_rows.append(
            {
                "video_name": video_name,
                "predictor_peak": float(predictor_scores.max()) if len(predictor_scores) else 0.0,
                "frozen_peak": float(frozen_scores.max()) if len(frozen_scores) else 0.0,
                "anomalous_frames": int(labels.sum()),
            }
        )
        write_timeline_svg(
            paths["plots"] / f"{video_name}_predictor.svg",
            f"{video_name} predictor score smoothed",
            predictor_scores,
            labels,
        )
        write_timeline_svg(
            paths["plots"] / f"{video_name}_frozen.svg",
            f"{video_name} frozen feature diff smoothed",
            frozen_scores,
            labels,
        )
    _write_csv(paths["reports"] / "timeline_summary.csv", timeline_rows)
    print(yaml.safe_dump(metrics, sort_keys=False))
