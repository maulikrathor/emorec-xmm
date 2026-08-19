"""Losses, metrics, and the training loop shared by every model in the repo."""

from emorec.training.losses import build_loss
from emorec.training.metrics import (
    classification_report_text,
    compute_metrics,
    save_confusion_matrix,
    trivial_baseline_metrics,
)
from emorec.training.trainer import evaluate, fit

__all__ = [
    "build_loss",
    "classification_report_text",
    "compute_metrics",
    "evaluate",
    "fit",
    "save_confusion_matrix",
    "trivial_baseline_metrics",
]
