from __future__ import annotations

from typing import Any

from vjepa_forge.metrics.classification import top1_accuracy


def evaluate_classification(logits, labels) -> dict[str, Any]:
    return {"top1": top1_accuracy(logits, labels)}
