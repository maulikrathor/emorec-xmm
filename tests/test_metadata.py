"""Tests for emorec.data.metadata: CSV loading, id construction, and hard validations."""

from pathlib import Path

import pandas as pd
import pytest
from conftest import MELD_COLUMNS, _write_csv

from emorec.data.labels import EMOTION_TO_ID, ID_TO_EMOTION
from emorec.data.metadata import (
    _check_all_emotions_in_train,
    _check_emotions_known,
    _check_global_utt_id_uniqueness,
    _check_id_types,
    _check_no_nulls,
    _check_row_counts,
    _check_unique_dialogue_utterance,
    load_split_csv,
    make_utt_id,
    validate_metadata,
)
from emorec.data.text_clean import normalize_text

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_fixes_smart_quotes_and_dashes() -> None:
    assert normalize_text("‘hi’") == "'hi'"
    assert normalize_text("“hi”") == '"hi"'
    assert normalize_text("wait—what") == "wait-what"
    assert normalize_text("well…") == "well..."


def test_normalize_text_strips_stray_windows1252_control_bytes() -> None:
    # \x92 is the byte that lands as U+0092 when a cp1252 file is mis-decoded
    # as latin-1 (rather than being correctly interpreted as a right single
    # quote). normalize_text strips it rather than guessing its meaning.
    assert normalize_text("don\x92t stop") == "dont stop"


def test_normalize_text_preserves_punctuation_case_and_apostrophes() -> None:
    s = "Don't. Wait -- really?! It's 3:00pm, y'all."
    assert normalize_text(s) == s


def test_normalize_text_collapses_whitespace_and_strips_ends() -> None:
    assert normalize_text("  a   b\t\tc  \n") == "a b c"


# ---------------------------------------------------------------------------
# make_utt_id
# ---------------------------------------------------------------------------


def test_make_utt_id_golden_strings() -> None:
    assert make_utt_id("train", 0, 0) == "train/dia0_utt0"
    assert make_utt_id("dev", 12, 3) == "dev/dia12_utt3"
    assert make_utt_id("test", 0, 0) == "test/dia0_utt0"  # same dia/utt, different split -> different id


# ---------------------------------------------------------------------------
# load_split_csv: encoding, normalization, deterministic sort
# ---------------------------------------------------------------------------


def _minimal_row(sr_no, text, speaker, emotion, dia, utt, season=1, episode=1, start="00:00:00,000", end="00:00:01,000"):
    return [sr_no, text, speaker, emotion, "neutral", dia, utt, season, episode, start, end]


def test_load_split_csv_sorts_deterministically_regardless_of_input_order(tmp_path: Path) -> None:
    rows = [
        _minimal_row(1, "c", "A", "joy", 0, 2),
        _minimal_row(2, "a", "A", "neutral", 0, 0),
        _minimal_row(3, "b", "A", "neutral", 0, 1),
    ]
    shuffled = [rows[2], rows[0], rows[1]]

    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    _write_csv(path_a, rows)
    _write_csv(path_b, shuffled)

    df_a = load_split_csv(path_a)
    df_b = load_split_csv(path_b)

    pd.testing.assert_frame_equal(df_a, df_b)
    assert list(df_a["Utterance"]) == ["a", "b", "c"]


def test_load_split_csv_label_roundtrips_through_id_to_emotion(tmp_path: Path) -> None:
    rows = [_minimal_row(1, "hi", "A", "Joy", 0, 0)]  # mixed case on purpose
    path = tmp_path / "x.csv"
    _write_csv(path, rows)

    df = load_split_csv(path)
    emotion = df.loc[0, "Emotion"]
    label_id = EMOTION_TO_ID[emotion]
    assert ID_TO_EMOTION[label_id] == emotion == "joy"


# ---------------------------------------------------------------------------
# Hard validation checks, exercised individually so a small df can isolate
# exactly one violation without needing to satisfy every other check too
# (see metadata.py's module docstring / MELD_EXPECTED_ROW_COUNTS comment).
# ---------------------------------------------------------------------------


def test_v2_fires_on_unknown_emotion_value() -> None:
    df = pd.DataFrame(
        [_minimal_row(1, "hi", "A", "confused", 0, 0)],
        columns=MELD_COLUMNS,
    )
    df["Emotion"] = df["Emotion"].str.strip().str.lower()

    with pytest.raises(ValueError, match="confused"):
        _check_emotions_known({"train": df})


def test_v3_fires_when_train_missing_an_emotion_class() -> None:
    df = pd.DataFrame(
        [_minimal_row(1, "hi", "A", "neutral", 0, 0)],
        columns=MELD_COLUMNS,
    )
    df["Emotion"] = df["Emotion"].str.strip().str.lower()

    with pytest.raises(ValueError, match="missing emotion class"):
        _check_all_emotions_in_train({"train": df})


def test_v4_fires_on_duplicate_dialogue_utterance_pair() -> None:
    df = pd.DataFrame(
        [
            _minimal_row(1, "a", "A", "neutral", 0, 0),
            _minimal_row(2, "b", "A", "joy", 0, 0),  # duplicate (Dialogue_ID, Utterance_ID)
        ],
        columns=MELD_COLUMNS,
    )

    with pytest.raises(ValueError, match="duplicate"):
        _check_unique_dialogue_utterance({"train": df})


def test_v7_fires_if_utt_id_namespacing_is_bypassed() -> None:
    # Correctly namespaced ids never collide even with identical (dia, utt)
    # across splits:
    namespaced = [make_utt_id("train", 0, 0), make_utt_id("dev", 0, 0)]
    _check_global_utt_id_uniqueness(namespaced)  # does not raise

    # Bypassing the split prefix (using the bare "diaX_uttY" form) is exactly
    # the mistake namespacing exists to prevent (TRIBAL_KNOWLEDGE.md §1.4):
    bypassed = ["dia0_utt0", "dia0_utt0"]
    with pytest.raises(ValueError, match="namespacing bypassed"):
        _check_global_utt_id_uniqueness(bypassed)


def test_v5_fires_on_negative_id() -> None:
    df = pd.DataFrame(
        [_minimal_row(1, "hi", "A", "neutral", -1, 0)],
        columns=MELD_COLUMNS,
    )
    with pytest.raises(ValueError, match="negative"):
        _check_id_types({"train": df})


def test_v6_fires_on_nan_utterance() -> None:
    df = pd.DataFrame(
        [_minimal_row(1, None, "A", "neutral", 0, 0)],
        columns=MELD_COLUMNS,
    )
    with pytest.raises(ValueError, match="NaN"):
        _check_no_nulls({"train": df})


def test_v1_fires_on_row_count_mismatch() -> None:
    df = pd.DataFrame([_minimal_row(1, "hi", "A", "neutral", 0, 0)], columns=MELD_COLUMNS)
    with pytest.raises(ValueError, match="row count mismatch"):
        _check_row_counts({"train": df}, expected={"train": 2})


# ---------------------------------------------------------------------------
# validate_metadata end-to-end happy path (R1-R6 + report writing), using
# expected_row_counts overridden to match small synthetic data — the real
# 9989/1109/2610 counts are only meaningful against the real mirror.
# ---------------------------------------------------------------------------


def test_validate_metadata_end_to_end_writes_reports(tmp_path: Path) -> None:
    train_rows = [
        _minimal_row(1, "a", "Ross", "neutral", 0, 0, season=1, episode=1),
        _minimal_row(2, "b", "Ross", "joy", 0, 1, season=1, episode=1),
        _minimal_row(3, "c", "Rachel", "sadness", 0, 2, season=1, episode=1),
        _minimal_row(4, "d", "Rachel", "anger", 1, 0, season=1, episode=2),
        _minimal_row(5, "e", "Ross", "surprise", 1, 1, season=1, episode=2),
        _minimal_row(6, "f", "Rachel", "fear", 1, 2, season=1, episode=2),
        _minimal_row(7, "g", "Ross", "disgust", 1, 3, season=1, episode=2),
    ]
    dev_rows = [_minimal_row(1, "x", "Ross", "neutral", 0, 0, season=1, episode=1)]  # overlapping episode w/ train
    test_rows = [_minimal_row(1, "y", "Rachel", "joy", 0, 0, season=2, episode=1)]

    dfs = {
        "train": pd.DataFrame(train_rows, columns=MELD_COLUMNS),
        "dev": pd.DataFrame(dev_rows, columns=MELD_COLUMNS),
        "test": pd.DataFrame(test_rows, columns=MELD_COLUMNS),
    }
    for df in dfs.values():
        df["Emotion"] = df["Emotion"].str.strip().str.lower()

    report_dir = tmp_path / "reports"
    report = validate_metadata(
        dfs,
        report_dir=report_dir,
        expected_row_counts={"train": 7, "dev": 1, "test": 1},
    )

    assert report.row_counts == {"train": 7, "dev": 1, "test": 1}
    assert report.emotion_counts["train"]["neutral"] == 1
    assert sum(report.emotion_counts["train"].values()) == 7
    assert report.episode_overlap["count"] == 1  # (Season=1, Episode=1) in both train and dev
    assert (report_dir / "meld_metadata.md").exists()
    assert (report_dir / "meld_metadata.json").exists()
