from __future__ import annotations

"""Implements Forge-native detection metrics for decoded box predictions."""

from collections import defaultdict

import torch

from vjepa_forge.heads.detection.box_ops import box_iou


def _average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """Implements Pascal/COCO-style interpolated AP integration for Forge detection metrics."""
    if recalls.numel() == 0 or precisions.numel() == 0:
        return 0.0
    mrec = torch.cat([recalls.new_tensor([0.0]), recalls, recalls.new_tensor([1.0])])
    mpre = torch.cat([precisions.new_tensor([0.0]), precisions, precisions.new_tensor([0.0])])
    for idx in range(mpre.numel() - 2, -1, -1):
        mpre[idx] = torch.maximum(mpre[idx], mpre[idx + 1])
    changes = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).squeeze(1)
    if changes.numel() == 0:
        return 0.0
    ap = torch.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1])
    return float(ap.item())


def detection_average_precision(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    iou_threshold: float,
    num_classes: int,
) -> float:
    """Computes class-averaged AP for Forge-native decoded detections."""
    per_class_predictions: dict[int, list[tuple[int, float, torch.Tensor]]] = defaultdict(list)
    per_class_targets: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    for image_idx, target in enumerate(targets):
        for label, box in zip(target["labels"], target["boxes"], strict=True):
            per_class_targets[int(label.item())][image_idx].append(box)
    for image_idx, prediction in enumerate(predictions):
        for label, score, box in zip(prediction["labels"], prediction["scores"], prediction["boxes"], strict=True):
            per_class_predictions[int(label.item())].append((image_idx, float(score.item()), box))

    aps: list[float] = []
    for class_id in range(int(num_classes)):
        gt_by_image = per_class_targets.get(class_id, {})
        total_gts = sum(len(items) for items in gt_by_image.values())
        preds = sorted(per_class_predictions.get(class_id, []), key=lambda item: item[1], reverse=True)
        if total_gts == 0:
            continue
        if not preds:
            aps.append(0.0)
            continue
        matched: dict[int, set[int]] = defaultdict(set)
        tps = torch.zeros(len(preds), dtype=torch.float32)
        fps = torch.zeros(len(preds), dtype=torch.float32)
        for pred_idx, (image_idx, _, pred_box) in enumerate(preds):
            gt_boxes = gt_by_image.get(image_idx, [])
            if not gt_boxes:
                fps[pred_idx] = 1.0
                continue
            gt_tensor = torch.stack(gt_boxes, dim=0)
            ious = box_iou(pred_box.unsqueeze(0), gt_tensor).squeeze(0)
            best_iou, best_gt = torch.max(ious, dim=0)
            best_gt_idx = int(best_gt.item())
            if float(best_iou.item()) >= float(iou_threshold) and best_gt_idx not in matched[image_idx]:
                matched[image_idx].add(best_gt_idx)
                tps[pred_idx] = 1.0
            else:
                fps[pred_idx] = 1.0
        cum_tps = torch.cumsum(tps, dim=0)
        cum_fps = torch.cumsum(fps, dim=0)
        recalls = cum_tps / max(float(total_gts), 1.0)
        precisions = cum_tps / torch.clamp(cum_tps + cum_fps, min=1.0)
        aps.append(_average_precision(recalls, precisions))
    if not aps:
        return 0.0
    return float(sum(aps) / len(aps))


def summarize_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    num_classes: int,
) -> dict[str, float]:
    """Builds Forge-native detection metric summaries from decoded predictions."""
    ap50 = detection_average_precision(predictions, targets, iou_threshold=0.50, num_classes=num_classes)
    ap75 = detection_average_precision(predictions, targets, iou_threshold=0.75, num_classes=num_classes)
    return {"box_ap": (ap50 + ap75) * 0.5, "ap50": ap50, "ap75": ap75}
