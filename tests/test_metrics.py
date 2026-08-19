"""Tests for emorec.training.metrics and (bundled per the CC-5 test-file list) emorec.training.losses."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from emorec.data.labels import EMOTION_TO_ID, ID_TO_EMOTION, NUM_CLASSES
from emorec.training.losses import build_loss
from emorec.training.metrics import (
    ALL_LABELS,
    classification_report_text,
    compute_metrics,
    save_confusion_matrix,
    trivial_baseline_metrics,
)

# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_matches_sklearn() -> None:
    y_true = [0, 1, 2, 0, 1, 3, 4, 5, 6, 0]
    y_pred = [0, 1, 1, 0, 2, 3, 4, 5, 5, 0]

    result = compute_metrics(y_true, y_pred)

    assert result["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert result["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)
    )
    assert result["weighted_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=ALL_LABELS, average="weighted", zero_division=0)
    )

    cm = sk_confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    assert result["confusion_matrix"] == cm.tolist()


def test_absent_class_from_predictions_yields_zero_f1_no_crash() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]  # class 1 (joy) never predicted

    result = compute_metrics(y_true, y_pred)

    assert result["per_class_f1"]["joy"] == 0.0
    assert result["per_class_f1"]["fear"] == 0.0  # never appears in y_true or y_pred at all


def test_confusion_matrix_is_7x7_and_row_sums_equal_support() -> None:
    y_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    y_pred = [0, 1, 1, 3, 4, 5, 6, 0, 0, 2]

    result = compute_metrics(y_true, y_pred)
    cm = np.array(result["confusion_matrix"])

    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    for i in range(NUM_CLASSES):
        emotion = ID_TO_EMOTION[i]
        assert int(cm[i].sum()) == result["per_class_support"][emotion]


# ---------------------------------------------------------------------------
# trivial_baseline_metrics
# ---------------------------------------------------------------------------


def test_trivial_baseline_metrics_matches_hand_calc() -> None:
    neutral = EMOTION_TO_ID["neutral"]
    joy = EMOTION_TO_ID["joy"]
    y_true = [neutral] * 7 + [joy] * 3  # 7 neutral, 3 joy, out of 10

    result = trivial_baseline_metrics(y_true)

    assert result["accuracy"] == pytest.approx(0.7)
    # neutral: precision = 7/10, recall = 7/7 = 1.0 -> F1 = 2*0.7*1/(0.7+1) = 14/17
    assert result["per_class_f1"]["neutral"] == pytest.approx(14 / 17)
    # joy: never predicted -> precision undefined (0 by zero_division), recall 0 -> F1 0
    assert result["per_class_f1"]["joy"] == pytest.approx(0.0)
    assert result["per_class_support"]["neutral"] == 7
    assert result["per_class_support"]["joy"] == 3


# ---------------------------------------------------------------------------
# classification_report_text / save_confusion_matrix
# ---------------------------------------------------------------------------


def test_classification_report_text_mentions_all_seven_classes() -> None:
    y_true = list(range(NUM_CLASSES))
    y_pred = list(range(NUM_CLASSES))
    text = classification_report_text(y_true, y_pred)
    for name in ID_TO_EMOTION.values():
        assert name in text


def test_save_confusion_matrix_writes_nonempty_png(tmp_path: Path) -> None:
    cm = [[3, 1], [0, 2]]
    path = tmp_path / "nested" / "cm.png"
    save_confusion_matrix(cm, path, class_names=["a", "b"], title="test title")
    assert path.exists()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# losses.py: weighted-CE weight math (bundled here per the CC-5 test-file list)
# ---------------------------------------------------------------------------


def test_weighted_ce_matches_hand_calculation() -> None:
    cfg = OmegaConf.create({"loss": {"name": "weighted_ce"}})
    counts = [100, 25, 4]  # hand-calculated 3-class example
    loss_fn = build_loss(cfg, counts)

    raw = [1 / (100**0.5), 1 / (25**0.5), 1 / (4**0.5)]  # [0.1, 0.2, 0.5]
    mean_raw = sum(raw) / 3
    expected = [w / mean_raw for w in raw]  # [0.375, 0.75, 1.875]

    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    torch.testing.assert_close(loss_fn.weight, torch.tensor(expected, dtype=torch.float32))
    assert loss_fn.weight.mean().item() == pytest.approx(1.0)


def test_weighted_ce_rejects_zero_count_class() -> None:
    cfg = OmegaConf.create({"loss": {"name": "weighted_ce"}})
    with pytest.raises(ValueError):
        build_loss(cfg, [10, 0, 5])


def test_build_loss_rejects_unknown_loss_name() -> None:
    cfg = OmegaConf.create({"loss": {"name": "focal"}})
    with pytest.raises(ValueError):
        build_loss(cfg, [10, 10, 10])
