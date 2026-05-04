from __future__ import annotations

import torch

from vjepa_forge.engine.validator import BaseValidator, ValidationResult
from vjepa_forge.metrics.classification import top1_accuracy


class ClassifyValidator(BaseValidator):
    def run(self) -> ValidationResult:
        self.model.to(self.device)
        self.model.eval()
        split = str(getattr(self, "split", "val"))
        loader = self.build_loader(split=split)
        total = 0.0
        batches = 0
        logits_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        with torch.no_grad():
            progress = self.progress(loader, desc=f"{split}", total=len(loader))
            for batch in progress:
                batch.x = batch.x.to(self.device)
                outputs = self.model(batch)
                total += float(self.compute_loss(batch, outputs).detach().cpu().item())
                batches += 1
                logits_parts.append(outputs.detach().cpu())
                label_parts.append(batch.labels["class_ids"].detach().cpu())
                if batches > 0 and progress is not loader:
                    progress.set_postfix(loss=f"{(total / batches):.4f}")
        metrics = None
        if logits_parts:
            logits = torch.cat(logits_parts, dim=0)
            labels = torch.cat(label_parts, dim=0)
            metrics = {"top1": top1_accuracy(logits, labels)}
        return ValidationResult(loss=total / max(batches, 1), batches=batches, metrics=metrics, split=split)


__all__ = ["ClassifyValidator"]
