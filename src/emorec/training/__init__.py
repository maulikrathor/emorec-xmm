"""Losses, metrics, and the training loop shared by every model in the repo."""

from emorec.training.losses import build_loss
from emorec.training.metrics import (
    align_predictions,
    bootstrap_ci,
    classification_report_text,
    compute_metrics,
    load_predictions,
    mcnemar,
    save_confusion_matrix,
    trivial_baseline_metrics,
)
from emorec.training.trainer import evaluate, fit, save_predictions

__all__ = [
    "align_predictions",
    "bootstrap_ci",
    "build_loss",
    "classification_report_text",
    "compute_metrics",
    "evaluate",
    "fit",
    "load_predictions",
    "mcnemar",
    "save_confusion_matrix",
    "save_predictions",
    "trivial_baseline_metrics",
]
