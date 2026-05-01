from __future__ import annotations

from dataclasses import dataclass

import torch

from .trainer import BaseTrainer


@dataclass
class PredictResult:
    outputs: list


class BasePredictor(BaseTrainer):
    def run(self) -> PredictResult:
        self.model.to(self.device)
        self.model.eval()
        loader = self.build_loader(split="val")
        results = []
        with torch.no_grad():
            for batch in loader:
                batch.x = batch.x.to(self.device)
                results.append(self.model(batch))
        return PredictResult(outputs=results)
