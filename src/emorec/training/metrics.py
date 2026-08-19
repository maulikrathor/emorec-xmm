"""Classification metrics, the trivial-baseline honesty anchor, and confusion-matrix plotting.

# TODO(post-review): bootstrap_ci + mcnemar per EVALUATION.md section 4
# (ED12) — additive, no refactor required.

CONTRACT: every metric here is computed with `labels=list(range(NUM_CLASSES))`
ALWAYS, so a class absent from a small/pilot run's `y_true`/`y_pred` still
appears in every dict with a defined (zero) value instead of silently
vanishing or raising.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (docs/TRIBAL_KNOWLEDGE.md §1.8) -- set BEFORE pyplot import, at module import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)

from emorec.data.labels import EMOTION_NAMES, EMOTION_TO_ID, ID_TO_EMOTION, NUM_CLASSES

ALL_LABELS: list[int] = list(range(NUM_CLASSES))


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Accuracy, macro/weighted F1, per-class F1/support, and a 7x7 confusion matrix.

    `labels=ALL_LABELS` is passed to every sklearn call so a class absent
    from `y_true`/`y_pred` (common on tiny pilot subsets) reports `0.0`
    rather than being silently dropped from the averages or crashing.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=ALL_LABELS, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=ALL_LABELS, average=None, zero_division=0)
    cm = sk_confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    support = cm.sum(axis=1)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": {ID_TO_EMOTION[i]: float(per_class_f1[i]) for i in ALL_LABELS},
        "per_class_support": {ID_TO_EMOTION[i]: int(support[i]) for i in ALL_LABELS},
        "confusion_matrix": cm.tolist(),
    }


def trivial_baseline_metrics(y_true: list[int]) -> dict:
    """`compute_metrics` for the "always predict neutral" baseline -- the honesty anchor.

    Every headline number must be reported alongside this: a model that
    doesn't clear it isn't doing anything (docs/CLAUDE.md commandment #9,
    "empty is honest, fake is fatal" — the trivial baseline is what makes a
    claimed improvement checkable, not just asserted).
    """
    y_true = list(y_true)
    neutral_id = EMOTION_TO_ID["neutral"]
    y_pred = [neutral_id] * len(y_true)
    return compute_metrics(y_true, y_pred)


def classification_report_text(y_true: list[int], y_pred: list[int]) -> str:
    """`sklearn.metrics.classification_report` rendered with MELD's frozen emotion names, all 7 classes always shown."""
    return classification_report(
        y_true, y_pred, labels=ALL_LABELS, target_names=EMOTION_NAMES, zero_division=0
    )


def save_confusion_matrix(cm, path, class_names: list[str], title: str) -> None:
    """Write a heatmap PNG of `cm` (list[list[int]] or array-like) to `path`.

    Uses the `Agg` backend set at module import (headless-safe: never tries
    to open a display). Creates parent directories.
    """
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(int(cm_arr[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
