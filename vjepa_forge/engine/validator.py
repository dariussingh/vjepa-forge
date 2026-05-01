from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .trainer import BaseTrainer


@dataclass
class ValidationResult:
    loss: float
    batches: int
    metrics: dict[str, Any] | None = None
    split: str = "val"


class BaseValidator(BaseTrainer):
    def run(self) -> ValidationResult:
        self.model.to(self.device)
        self.model.eval()
        split = str(getattr(self, "split", "val"))
        loader = self.build_loader(split=split)
        total = 0.0
        batches = 0
        with torch.no_grad():
            progress = self.progress(loader, desc=f"{split}", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                outputs = self.model(batch)
                total += float(self.compute_loss(batch, outputs).detach().cpu().item())
                batches += 1
                if batches > 0 and progress is not loader:
                    progress.set_postfix(loss=f"{(total / batches):.4f}")
        return ValidationResult(loss=total / max(batches, 1), batches=batches, split=split)


def binary_roc_auc(labels: list[float] | np.ndarray, scores: list[float] | np.ndarray) -> float | None:
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    pos = labels_np == 1
    neg = labels_np == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores_np, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores_np), dtype=np.float64) + 1.0
    unique_scores, inverse, counts = np.unique(scores_np, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
