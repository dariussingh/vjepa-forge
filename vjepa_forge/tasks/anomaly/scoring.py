from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vjepa_forge.metrics.anomaly import roc_auc_score as _roc_auc_score


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


def _normal_stats(scores: np.ndarray) -> dict[str, float]:
    return {"mean": float(scores.mean()) if len(scores) else 0.0, "std": float(scores.std()) if len(scores) else 0.0}


def _timing_metrics(*, decode_times: list[float], model_times: list[float]) -> dict[str, float]:
    avg_decode = float(np.mean(decode_times)) if decode_times else 0.0
    avg_model = float(np.mean(model_times)) if model_times else 0.0
    return {
        "avg_decode_time": avg_decode,
        "avg_model_time": avg_model,
        "avg_step_time": avg_decode + avg_model,
    }


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(scores) == 0:
        return scores.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(scores.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _finalize_video_summary(by_video: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"videos": {}}
    for video_name, state in by_video.items():
        frame_ids = sorted(state["predictor_sum"].keys())
        predictor_series = np.asarray([state["predictor_sum"][idx] / state["predictor_count"][idx] for idx in frame_ids], dtype=np.float32)
        frozen_series = np.asarray([state["frozen_sum"][idx] / state["frozen_count"][idx] for idx in frame_ids], dtype=np.float32)
        labels = None
        if state["has_labels"]:
            labels = np.asarray([state["labels"].get(idx, 0) for idx in frame_ids], dtype=np.int64)
        summary["videos"][video_name] = {
            "frame_ids": frame_ids,
            "predictor_scores": predictor_series.tolist(),
            "frozen_scores": frozen_series.tolist(),
            "labels": None if labels is None else labels.tolist(),
        }
    return summary


def _flatten_metric_arrays(video_summary: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for video in video_summary["videos"].values():
        if video["labels"] is None:
            continue
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


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _clip_score_rows(video_summary: dict[str, Any], key: str, *, reduction: str = "max") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_name, payload in video_summary["videos"].items():
        scores = np.asarray(payload[key], dtype=np.float32)
        labels = None if payload["labels"] is None else np.asarray(payload["labels"], dtype=np.int64)
        if reduction == "max":
            clip_score = float(scores.max()) if scores.size else float("nan")
        elif reduction == "mean":
            clip_score = float(scores.mean()) if scores.size else float("nan")
        else:
            raise ValueError(f"Unsupported clip score reduction: {reduction}")
        clip_label = None if labels is None else int(np.any(labels != 0))
        rows.append(
            {
                "video_name": video_name,
                "clip_label": clip_label,
                "clip_score": clip_score,
            }
        )
    return rows


def _clip_level_metrics(video_summary: dict[str, Any], key: str, *, threshold: float, reduction: str = "max") -> dict[str, Any]:
    rows = _clip_score_rows(video_summary, key, reduction=reduction)
    labeled_rows = [row for row in rows if row["clip_label"] is not None]
    labels = np.asarray([row["clip_label"] for row in labeled_rows], dtype=np.int64)
    scores = np.asarray([row["clip_score"] for row in labeled_rows], dtype=np.float32)
    predictions = (scores > float(threshold)).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, len(rows))
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0.0 else 0.0
    return {
        "reduction": reduction,
        "auc": _roc_auc_score(labels, scores),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "counts": {
            "total": int(len(labeled_rows)),
            "normal": int(np.sum(labels == 0)),
            "anomaly": int(np.sum(labels == 1)),
        },
        "clips": rows,
    }


def _threshold_clip_predictions(video_summary: dict[str, Any], key: str, *, threshold: float, reduction: str = "max") -> dict[str, Any]:
    rows = _clip_score_rows(video_summary, key, reduction=reduction)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        label = row["clip_label"]
        clip_score = float(row["clip_score"])
        predicted_label = int(clip_score > float(threshold))
        enriched.append(
            {
                "video_name": row["video_name"],
                "clip_score": clip_score,
                "clip_label": label,
                "predicted_label": predicted_label,
                "threshold": float(threshold),
            }
        )
    return {
        "clip_score_reduction": reduction,
        "threshold": float(threshold),
        "clips": enriched,
    }


def _thresholds_from_smoothed_summary(smoothed: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    calibration_labels, calibration_scores_pred = _flatten_metric_arrays(smoothed, "predictor_scores")
    _, calibration_scores_frozen = _flatten_metric_arrays(smoothed, "frozen_scores")
    pred_reference = calibration_scores_pred[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_pred
    frozen_reference = calibration_scores_frozen[calibration_labels == 0] if np.any(calibration_labels == 0) else calibration_scores_frozen
    pred_stats = _normal_stats(pred_reference)
    frozen_stats = _normal_stats(frozen_reference)
    multiplier = cfg["eval"]["threshold_std_multiplier"]
    predictor_threshold = float(pred_stats["mean"] + multiplier * pred_stats["std"])
    frozen_threshold = float(frozen_stats["mean"] + multiplier * frozen_stats["std"])
    return predictor_threshold, frozen_threshold, calibration_labels, calibration_scores_pred, calibration_scores_frozen
