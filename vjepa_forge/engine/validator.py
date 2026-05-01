from __future__ import annotations

from dataclasses import dataclass

import torch

from .trainer import BaseTrainer


@dataclass
class ValidationResult:
    loss: float
    batches: int


class BaseValidator(BaseTrainer):
    def run(self) -> ValidationResult:
        self.model.to(self.device)
        self.model.eval()
        loader = self.build_loader(split="val")
        total = 0.0
        batches = 0
        with torch.no_grad():
            for batch in loader:
                batch.x = batch.x.to(self.device)
                outputs = self.model(batch)
                total += float(self.compute_loss(batch, outputs).detach().cpu().item())
                batches += 1
        return ValidationResult(loss=total / max(batches, 1), batches=batches)
