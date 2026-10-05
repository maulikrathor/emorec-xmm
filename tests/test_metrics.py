"""Tests for emorec.training.metrics and (bundled per the CC-5 test-file list) emorec.training.losses."""

import json
from math import comb
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
    align_predictions,
    bootstrap_ci,
    classification_report_text,
    compute_metrics,
    load_predictions,
    mcnemar,
    save_confusion_matrix,
    trivial_baseline_metrics,
)
from emorec.training.trainer import save_predictions

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


# ---------------------------------------------------------------------------
# CC-7: persisted predictions, bootstrap CI, McNemar (ED12)
# ---------------------------------------------------------------------------


def _write_preds(path: Path, utt_ids: list[str], y_true: list[int], y_pred: list[int]) -> None:
    probs = [[1.0 if c == p else 0.0 for c in range(7)] for p in y_pred]
    save_predictions(path, utt_ids, y_true, y_pred, probs)


def test_save_predictions_exact_schema_and_round_trip(tmp_path: Path) -> None:
    utt_ids = ["dev/dia0_utt0", "dev/dia0_utt1", "dev/dia1_utt0"]
    y_true = [0, 3, 6]
    y_pred = [0, 4, 6]
    raw = np.random.default_rng(0).random((3, 7))
    y_prob = (raw / raw.sum(axis=1, keepdims=True)).tolist()

    path = tmp_path / "sub" / "predictions_dev.jsonl"
    save_predictions(path, utt_ids, y_true, y_pred, y_prob)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line, uid, label, pred, probs in zip(lines, utt_ids, y_true, y_pred, y_prob, strict=True):
        row = json.loads(line)
        assert list(row) == ["utt_id", "label", "pred", "probs"]  # fixed key order
        assert row["utt_id"] == uid and row["label"] == label and row["pred"] == pred
        assert len(row["probs"]) == 7
        assert row["probs"] == [round(p, 6) for p in probs]

    ids2, true2, pred2, prob2 = load_predictions(path)
    assert ids2 == utt_ids and true2 == y_true and pred2 == y_pred
    for got, want in zip(prob2, y_prob, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


def test_save_predictions_rejects_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        save_predictions(tmp_path / "p.jsonl", ["a", "b"], [0], [0, 1], [[0.0] * 7, [0.0] * 7])


def test_bootstrap_ci_point_matches_compute_metrics_and_brackets_it() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 7, size=300).tolist()
    y_pred = [t if rng.random() < 0.6 else int(rng.integers(0, 7)) for t in y_true]
    expected = compute_metrics(y_true, y_pred)

    for metric in ("weighted_f1", "macro_f1", "accuracy"):
        ci = bootstrap_ci(y_true, y_pred, metric=metric, n=200, seed=1)
        assert ci["point"] == pytest.approx(expected[metric])
        assert ci["lo"] <= ci["point"] <= ci["hi"]
        assert ci["lo"] < ci["hi"]


def test_bootstrap_ci_same_seed_identical_and_different_seed_differs() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 7, size=200).tolist()
    y_pred = [t if rng.random() < 0.5 else int(rng.integers(0, 7)) for t in y_true]

    a = bootstrap_ci(y_true, y_pred, n=200, seed=7)
    b = bootstrap_ci(y_true, y_pred, n=200, seed=7)
    c = bootstrap_ci(y_true, y_pred, n=200, seed=8)
    assert a == b
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])


def test_bootstrap_ci_perfect_predictions_collapse_to_one() -> None:
    y = [0, 1, 2, 3, 4, 5, 6] * 5
    ci = bootstrap_ci(y, y, metric="accuracy", n=50, seed=0)
    assert ci == {"point": 1.0, "lo": 1.0, "hi": 1.0}


def test_bootstrap_ci_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        bootstrap_ci([0, 1], [0, 1], metric="bleu")


def test_mcnemar_matches_hand_calculation() -> None:
    # 20 items. A-correct/B-wrong on 8 (n01), A-wrong/B-correct on 2 (n10),
    # both right on 6, both wrong on 4.
    y_true = [0] * 20
    pred_a = [0] * 8 + [1] * 2 + [0] * 6 + [1] * 4
    pred_b = [1] * 8 + [0] * 2 + [0] * 6 + [1] * 4

    result = mcnemar(y_true, pred_a, pred_b)

    assert result["n01"] == 8
    assert result["n10"] == 2
    assert result["n_discordant"] == 10
    assert result["statistic"] == 2  # min(n01, n10)
    # Exact two-sided binomial: 2 * P(X <= 2 | n=10, p=0.5) = 112 / 1024
    hand = 2 * sum(comb(10, k) for k in range(3)) / 2**10
    assert result["p_value"] == pytest.approx(hand)
    assert result["p_value"] == pytest.approx(0.109375)


def test_mcnemar_identical_predictors_give_p_one() -> None:
    y_true = [0, 1, 2, 3, 4, 5, 6, 0]
    pred = [0, 1, 2, 0, 0, 5, 6, 1]
    result = mcnemar(y_true, pred, list(pred))
    assert result["n_discordant"] == 0
    assert result["n01"] == result["n10"] == 0
    assert result["p_value"] == 1.0


def test_align_predictions_reorders_to_a_common_order(tmp_path: Path) -> None:
    ids = ["dev/dia0_utt0", "dev/dia0_utt1", "dev/dia1_utt0"]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_preds(path_a, ids, [0, 1, 2], [0, 1, 1])
    # B lists the same utterances in reverse order.
    _write_preds(path_b, ids[::-1], [2, 1, 0], [2, 0, 0])

    y_true, pred_a, pred_b = align_predictions(path_a, path_b)

    assert y_true == [0, 1, 2]
    assert pred_a == [0, 1, 1]
    assert pred_b == [0, 0, 2]  # B's predictions re-keyed to A's id order


def test_align_predictions_raises_when_utt_id_sets_differ(tmp_path: Path) -> None:
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_preds(path_a, ["dev/dia0_utt0", "dev/dia0_utt1", "dev/dia0_utt2"], [0, 1, 2], [0, 1, 2])
    _write_preds(path_b, ["dev/dia0_utt0", "dev/dia0_utt1", "dev/dia0_utt9"], [0, 1, 2], [0, 1, 2])

    with pytest.raises(ValueError, match=r"utt_id sets differ: 1 only in .* 1 only in"):
        align_predictions(path_a, path_b)
