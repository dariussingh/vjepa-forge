from __future__ import annotations

from typing import Any

import torch

from vjepa_forge.engine.predictor import PredictResult
from vjepa_forge.engine.trainer import BaseTrainer
from vjepa_forge.engine.validator import ValidationResult
from vjepa_forge.heads.segmentation.losses import build_instance_targets, build_semantic_targets
from vjepa_forge.metrics.segmentation import instance_mask_iou, mean_iou

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


class SegmentTrainer(BaseTrainer):
    def compute_loss(self, batch, outputs):
        loss, stats = self.model.head.compute_segmentation_loss(outputs, batch.labels)
        self._last_loss_stats = stats
        return loss


class SegmentValidator(BaseTrainer):
    def run(self) -> ValidationResult:
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = self.model.to(self.device)
        self.model.eval()
        split = str(getattr(self, "split", "val"))
        loader = self.build_loader(split=split)
        total = 0.0
        batches = 0
        metric_values: list[float] = []
        with self.runtime.inference_context():
            progress = _progress(loader, desc=f"{split}", total=len(loader))
            for batch in progress:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_pass(self.model, batch)
                with self.runtime.autocast_context():
                    loss, _ = self.model.head.compute_segmentation_loss(outputs, batch.labels)
                total += float(loss.detach().cpu().item())
                batches += 1
                if self.model.head.strategy == "ultralytics":
                    if isinstance(outputs, dict):
                        logits = outputs["pred_logits"]
                        targets = build_semantic_targets(batch.labels["segments"], num_classes=self.model.head.num_classes, output_size=int(logits.shape[-1]), video_frames=int(logits.shape[1])).to(logits.device)
                        flat_logits = logits.reshape(-1, logits.shape[2], logits.shape[3], logits.shape[4])
                        metric_values.append(mean_iou(flat_logits, targets, self.model.head.num_classes))
                    elif outputs.ndim == 5:
                        targets = build_semantic_targets(batch.labels["segments"], num_classes=self.model.head.num_classes, output_size=int(outputs.shape[-1]), video_frames=int(outputs.shape[2])).to(outputs.device)
                        logits = outputs.permute(0, 2, 1, 3, 4).reshape(-1, outputs.shape[1], outputs.shape[3], outputs.shape[4])
                        metric_values.append(mean_iou(logits, targets, self.model.head.num_classes))
                    else:
                        targets = build_semantic_targets(batch.labels["segments"], num_classes=self.model.head.num_classes, output_size=int(outputs.shape[-1])).to(outputs.device)
                        metric_values.append(mean_iou(outputs, targets, self.model.head.num_classes))
                else:
                    decoded = self.model.head.decode_predictions(outputs)
                    if outputs["pred_logits"].ndim == 4:
                        target_items = build_instance_targets(
                            [{"segments": [ann for ann in item["segments"] if int(ann["frame_idx"]) == frame_idx]} for item in batch.labels["segments"] for frame_idx in range(outputs["pred_masks"].shape[1])],
                            output_size=int(outputs["pred_masks"].shape[-1]),
                        )
                    else:
                        target_items = build_instance_targets(batch.labels["segments"], output_size=int(outputs["pred_masks"].shape[-1]))
                    for pred, target in zip(decoded, target_items, strict=True):
                        metric_values.append(instance_mask_iou(list(pred["masks"].float()), list(target["masks"])))
                if batches > 0 and progress is not loader:
                    progress.set_postfix(loss=f"{(total / batches):.4f}")
        metrics = {"miou": float(sum(metric_values) / len(metric_values)) if metric_values else 0.0}
        return ValidationResult(loss=total / max(batches, 1), batches=batches, metrics=metrics, split=split)


class SegmentPredictor(BaseTrainer):
    def run(self) -> PredictResult:
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = self.model.to(self.device)
        self.model.eval()
        split = str(getattr(self, "split", "val"))
        loader = self.build_loader(split=split)
        results: list[Any] = []
        with self.runtime.inference_context():
            progress = _progress(loader, desc=f"predict:{split}", total=len(loader))
            for batch in progress:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_pass(self.model, batch)
                results.append(self.model.head.decode_predictions(outputs))
        return PredictResult(outputs=results, summary={"batches": len(results)}, split=split)
