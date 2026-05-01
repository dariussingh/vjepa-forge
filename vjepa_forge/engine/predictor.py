from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .trainer import BaseTrainer


@dataclass
class PredictResult:
    outputs: list
    summary: dict[str, Any] | None = None
    split: str = "val"


class BasePredictor(BaseTrainer):
    def run(self) -> PredictResult:
        self.model.to(self.device)
        self.model.eval()
        split = str(getattr(self, "split", "val"))
        loader = self.build_loader(split=split)
        results = []
        with torch.no_grad():
            progress = self.progress(loader, desc=f"predict:{split}", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                results.append(self.model(batch))
        return PredictResult(outputs=results, split=split)
