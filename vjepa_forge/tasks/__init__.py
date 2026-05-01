from .anomaly.predict import AnomalyPredictor
from .anomaly.train import AnomalyTrainer
from .anomaly.val import AnomalyValidator
from .classify.predict import ClassifyPredictor
from .classify.train import ClassifyTrainer
from .classify.val import ClassifyValidator
from .detect.predict import DetectPredictor
from .detect.train import DetectTrainer
from .detect.val import DetectValidator
from .segment.predict import SegmentPredictor
from .segment.train import SegmentTrainer
from .segment.val import SegmentValidator

TASK_REGISTRY = {
    "classify": {"train": ClassifyTrainer, "val": ClassifyValidator, "predict": ClassifyPredictor},
    "detect": {"train": DetectTrainer, "val": DetectValidator, "predict": DetectPredictor},
    "segment": {"train": SegmentTrainer, "val": SegmentValidator, "predict": SegmentPredictor},
    "anomaly": {"train": AnomalyTrainer, "val": AnomalyValidator, "predict": AnomalyPredictor},
}

__all__ = ["TASK_REGISTRY"]
