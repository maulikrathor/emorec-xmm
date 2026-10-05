"""Classification metrics, the trivial-baseline honesty anchor, confusion-matrix plotting, and ED12 statistics.

CONTRACT: every metric here is computed with `labels=list(range(NUM_CLASSES))`
ALWAYS, so a class absent from a small/pilot run's `y_true`/`y_pred` still
appears in every dict with a defined (zero) value instead of silently
vanishing or raising.

Statistics (docs/EVALUATION.md section 4, ED12): `bootstrap_ci` (percentile
bootstrap over utterances) and `mcnemar` (exact paired test) operate on
per-utterance predictions persisted by `trainer.save_predictions`;
`load_predictions`/`align_predictions` read them back and guarantee paired
tests are never run on misaligned rows.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (docs/TRIBAL_KNOWLEDGE.md §1.8) -- set BEFORE pyplot import, at module import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
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


# ---------------------------------------------------------------------------
# ED12 statistics: bootstrap CI, McNemar, paired prediction loading
# ---------------------------------------------------------------------------

_BOOTSTRAP_METRICS = ("weighted_f1", "macro_f1", "accuracy")


def _metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """One scalar metric, computed exactly as `compute_metrics` does (`labels=ALL_LABELS`)."""
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    average = "weighted" if metric == "weighted_f1" else "macro"
    return float(f1_score(y_true, y_pred, labels=ALL_LABELS, average=average, zero_division=0))


def bootstrap_ci(
    y_true: list[int],
    y_pred: list[int],
    metric: str = "weighted_f1",
    n: int = 1000,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap 95% CI for `metric` (docs/EVALUATION.md section 4).

    Resamples utterance INDICES with replacement `n` times (each resample
    the same size as the data), recomputes `metric` on every resample, and
    reports the 2.5th/97.5th percentiles. Returns `{"point", "lo", "hi"}`
    where `point` is the metric on the original (un-resampled) predictions,
    identical to the value `compute_metrics` reports. Deterministic for a
    given `seed` (numpy `default_rng`). `metric` is one of `"weighted_f1"`,
    `"macro_f1"`, `"accuracy"`.
    """
    if metric not in _BOOTSTRAP_METRICS:
        raise ValueError(f"unknown metric {metric!r}; choices: {list(_BOOTSTRAP_METRICS)}")
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(f"length mismatch: y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}")
    size = len(y_true_arr)
    if size == 0:
        raise ValueError("bootstrap_ci requires at least one prediction")

    rng = np.random.default_rng(seed)
    scores = np.empty(n, dtype=np.float64)
    for b in range(n):
        idx = rng.integers(0, size, size=size)
        scores[b] = _metric_value(y_true_arr[idx], y_pred_arr[idx], metric)

    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {
        "point": _metric_value(y_true_arr, y_pred_arr, metric),
        "lo": float(lo),
        "hi": float(hi),
    }


def mcnemar(y_true: list[int], pred_a: list[int], pred_b: list[int]) -> dict:
    """Exact McNemar test on paired predictions of models A and B (ED12).

    Only DISCORDANT pairs carry information: `n01` = A correct & B wrong,
    `n10` = A wrong & B correct, `n_discordant = n01 + n10`. Under H0
    (equal error rates) `n01 ~ Binomial(n_discordant, 0.5)`; `p_value` is
    the two-sided exact binomial p-value from `scipy.stats.binomtest`
    (scipy is a declared dependency, also required by scikit-learn).
    `statistic` is `min(n01, n10)`, the exact test's statistic. With no
    discordant pairs (e.g. identical predictors) `p_value` is 1.0.
    `y_true`, `pred_a`, `pred_b` must be aligned row-for-row (see
    `align_predictions`).
    """
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, pred_a={len(pred_a)}, pred_b={len(pred_b)}"
        )
    t = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == t
    b_ok = np.asarray(pred_b) == t
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n_discordant = n01 + n10
    p_value = 1.0 if n_discordant == 0 else float(binomtest(n01, n_discordant, 0.5).pvalue)
    return {
        "n01": n01,
        "n10": n10,
        "statistic": min(n01, n10),
        "p_value": p_value,
        "n_discordant": n_discordant,
    }


def load_predictions(path: str | Path) -> tuple[list[str], list[int], list[int], list[list[float]]]:
    """Read a `trainer.save_predictions` JSONL file back as `(utt_ids, y_true, y_pred, y_prob)`, file order."""
    utt_ids: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            utt_ids.append(row["utt_id"])
            y_true.append(int(row["label"]))
            y_pred.append(int(row["pred"]))
            y_prob.append([float(p) for p in row["probs"]])
    return utt_ids, y_true, y_pred, y_prob


def align_predictions(
    path_a: str | Path, path_b: str | Path
) -> tuple[list[int], list[int], list[int]]:
    """Load two prediction files and return `(y_true, pred_a, pred_b)` in ONE common utt_id order.

    Asserts the two files contain exactly the same SET of `utt_id`s (and no
    duplicates, and agreeing gold labels), then reorders B to A's order.
    Raises `ValueError` naming the mismatch counts otherwise: a paired test
    on misaligned predictions is silently wrong, so this never guesses.
    """
    ids_a, true_a, pred_a, _ = load_predictions(path_a)
    ids_b, true_b, pred_b, _ = load_predictions(path_b)

    for name, ids in (("A", ids_a), ("B", ids_b)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"predictions {name} contain {len(ids) - len(set(ids))} duplicate utt_id(s)")

    set_a, set_b = set(ids_a), set(ids_b)
    if set_a != set_b:
        raise ValueError(
            f"utt_id sets differ: {len(set_a - set_b)} only in {path_a}, "
            f"{len(set_b - set_a)} only in {path_b} ({len(set_a & set_b)} shared)"
        )

    pos_b = {uid: i for i, uid in enumerate(ids_b)}
    order = [pos_b[uid] for uid in ids_a]
    true_b_aligned = [true_b[i] for i in order]
    pred_b_aligned = [pred_b[i] for i in order]

    n_label_mismatch = sum(1 for x, y in zip(true_a, true_b_aligned, strict=True) if x != y)
    if n_label_mismatch:
        raise ValueError(f"gold labels disagree on {n_label_mismatch} shared utt_id(s) between the two files")

    return true_a, pred_a, pred_b_aligned
