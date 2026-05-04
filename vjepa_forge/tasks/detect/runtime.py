from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vjepa_forge.engine.predictor import PredictResult
from vjepa_forge.engine.trainer import BaseTrainer, TrainResult
from vjepa_forge.engine.validator import ValidationResult
from vjepa_forge.heads.detection.box_ops import box_cxcywh_to_xyxy
from vjepa_forge.metrics.detection import summarize_detection_metrics

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def _frame_targets_from_labels(labels: dict[str, Any], *, media: str, num_frames: int | None = None) -> list[dict[str, torch.Tensor]]:
    targets: list[dict[str, torch.Tensor]] = []
    for item in labels["detections"]:
        detections = item["detections"]
        if media == "image":
            classes = torch.tensor([int(ann["class_id"]) for ann in detections], dtype=torch.int64) if detections else torch.empty(0, dtype=torch.int64)
            boxes = torch.tensor([ann["box"] for ann in detections], dtype=torch.float32) if detections else torch.empty(0, 4, dtype=torch.float32)
            targets.append({"labels": classes, "boxes": box_cxcywh_to_xyxy(boxes) if boxes.numel() else boxes})
            continue
        grouped: dict[int, list[dict[str, Any]]] = {}
        for det in detections:
            grouped.setdefault(int(det["frame_idx"]), []).append(det)
        max_frame = int(num_frames) if num_frames is not None else max(grouped, default=-1) + 1
        for frame_idx in range(max_frame):
            frame_dets = grouped.get(frame_idx, [])
            classes = torch.tensor([int(ann["class_id"]) for ann in frame_dets], dtype=torch.int64) if frame_dets else torch.empty(0, dtype=torch.int64)
            boxes = torch.tensor([ann["box"] for ann in frame_dets], dtype=torch.float32) if frame_dets else torch.empty(0, 4, dtype=torch.float32)
            targets.append({"labels": classes, "boxes": box_cxcywh_to_xyxy(boxes) if boxes.numel() else boxes})
    return targets


@dataclass
class DetectTrainResult(TrainResult):
    metrics: dict[str, float] | None = None


class DetectTrainer(BaseTrainer):
    def compute_loss(self, batch, outputs):
        loss, stats = self.model.head.compute_detection_loss(outputs, batch.labels)
        self._last_loss_stats = stats
        return loss


class DetectValidator(BaseTrainer):
    def run(self) -> ValidationResult:
        self.model.to(self.device)
        self.model.eval()
        split = str(getattr(self, "split", "val"))
        loader = self.build_loader(split=split)
        total = 0.0
        batches = 0
        decoded_predictions: list[dict[str, torch.Tensor]] = []
        decoded_targets: list[dict[str, torch.Tensor]] = []
        with torch.no_grad():
            progress = _progress(loader, desc=f"{split}", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                outputs = self.model(batch)
                loss, _ = self.model.head.compute_detection_loss(outputs, batch.labels)
                total += float(loss.detach().cpu().item())
                batches += 1
                decoded_predictions.extend(self.model.head.decode_predictions(outputs))
                frame_count = int(outputs["pred_logits"].shape[1]) if outputs["pred_logits"].ndim == 4 else None
                decoded_targets.extend(_frame_targets_from_labels(batch.labels, media=batch.media, num_frames=frame_count))
                if batches > 0 and progress is not loader:
                    progress.set_postfix(loss=f"{(total / batches):.4f}")
        metrics = summarize_detection_metrics(decoded_predictions, decoded_targets, num_classes=int(getattr(self.model.head.impl, "num_classes", self.model.model_cfg.get("num_classes", 1))))
        return ValidationResult(loss=total / max(batches, 1), batches=batches, metrics=metrics, split=split)


class DetectPredictor(BaseTrainer):
    def run(self) -> PredictResult:
        self.model.to(self.device)
        self.model.eval()
        split = str(getattr(self, "split", "val"))
        loader = self.build_loader(split=split)
        results: list[Any] = []
        summary_predictions: list[dict[str, torch.Tensor]] = []
        with torch.no_grad():
            progress = _progress(loader, desc=f"predict:{split}", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                outputs = self.model(batch)
                decoded = self.model.head.decode_predictions(outputs)
                results.append(decoded)
                summary_predictions.extend(decoded)
        summary = {"detections": int(sum(int(item["scores"].numel()) for item in summary_predictions))}
        return PredictResult(outputs=results, summary=summary, split=split)
