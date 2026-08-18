"""Tests for emorec.data.media: media indexing, utterance matching, and inventory."""

from pathlib import Path

import pandas as pd
import pytest
from conftest import MELD_COLUMNS, _touch_appledouble, _touch_mp4

from emorec.data.media import (
    MediaInventoryReport,
    build_media_index,
    inventory_media,
    is_appledouble,
    map_utterances_to_media,
)
from emorec.data.meld_paths import resolve_meld_layout
from emorec.data.metadata import load_split_csv


def test_is_appledouble_predicate() -> None:
    assert is_appledouble(Path("._dia0_utt0.mp4")) is True
    assert is_appledouble(Path("some/dir/._dia12_utt3.mp4")) is True
    assert is_appledouble(Path("dia0_utt0.mp4")) is False


def test_build_media_index_excludes_appledouble_and_reports_count(tmp_path: Path) -> None:
    video_dir = tmp_path / "video_dir"
    _touch_mp4(video_dir, 0, 0)
    _touch_mp4(video_dir, 0, 1)
    _touch_appledouble(video_dir, 0, 0)
    _touch_appledouble(video_dir, 9, 9)

    index, appledouble_skipped = build_media_index(video_dir)

    assert appledouble_skipped == 2
    assert set(index.keys()) == {"dia0_utt0", "dia0_utt1"}
    assert all(not is_appledouble(p) for p in index.values())


def test_missing_media_row_is_reported_and_never_matched(toy_meld_root: Path) -> None:
    layout = resolve_meld_layout(toy_meld_root)
    df = load_split_csv(layout.splits["train"].csv)

    matched, missing = map_utterances_to_media(df, "train", layout.splits["train"].video_dir)

    missing_pairs = {(m.dialogue_id, m.utterance_id) for m in missing}
    matched_pairs = {(m.dialogue_id, m.utterance_id) for m in matched}

    assert (1, 1) in missing_pairs  # deliberately unmatched row from the fixture
    assert (1, 1) not in matched_pairs
    assert len(matched) == 5
    assert len(missing) == 1


def test_orphan_mp4_is_counted_and_never_enters_a_match(toy_meld_root: Path) -> None:
    layout = resolve_meld_layout(toy_meld_root)
    dfs = {split: load_split_csv(sp.csv) for split, sp in layout.splits.items()}

    report = inventory_media(layout, dfs, report_dir=toy_meld_root / "reports")

    train_stats = report.per_split["train"]
    assert train_stats["orphan_media"] == 1
    assert "dia9_utt9" in train_stats["orphan_examples"]

    # The orphan must never appear as a matched utterance for any labeled row.
    matched, _missing = map_utterances_to_media(dfs["train"], "train", layout.splits["train"].video_dir)
    assert all((m.dialogue_id, m.utterance_id) != (9, 9) for m in matched)


def test_appledouble_shadowing_real_stem_does_not_satisfy_lookup_or_raise_duplicate(
    toy_meld_root: Path,
) -> None:
    layout = resolve_meld_layout(toy_meld_root)
    df = load_split_csv(layout.splits["test"].csv)

    # The fixture plants "._dia0_utt0.mp4" (AppleDouble) alongside the real
    # "dia0_utt0.mp4" in the test video dir. build_media_index must not
    # raise (that would be treating it as a stem collision) and the match
    # for dia0_utt0 must come from the real file, not the decoy.
    index, appledouble_skipped = build_media_index(layout.splits["test"].video_dir)
    assert appledouble_skipped == 2  # ._dia0_utt0.mp4 and ._dia50_utt50.mp4
    assert not is_appledouble(index["dia0_utt0"])

    matched, missing = map_utterances_to_media(df, "test", layout.splits["test"].video_dir)
    matched_by_pair = {(m.dialogue_id, m.utterance_id): m for m in matched}
    assert (0, 0) in matched_by_pair
    assert not is_appledouble(matched_by_pair[(0, 0)].mp4)
    assert not missing or all(m.dialogue_id != 0 or m.utterance_id != 0 for m in missing)


def test_appledouble_counted_separately_from_orphan_media(toy_meld_root: Path) -> None:
    layout = resolve_meld_layout(toy_meld_root)
    dfs = {split: load_split_csv(sp.csv) for split, sp in layout.splits.items()}

    report = inventory_media(layout, dfs, report_dir=toy_meld_root / "reports")

    test_stats = report.per_split["test"]
    assert test_stats["appledouble_skipped"] == 2
    # The decoy that shadows dia0_utt0 must not inflate orphan_media -- the
    # real dia0_utt0 file is matched, so orphan_media only reflects real,
    # unmatched media (none in this fixture's test split).
    assert test_stats["orphan_media"] == 0
    assert test_stats["mp4_files_total"] == test_stats["real_media"] + test_stats["appledouble_skipped"]


def test_per_split_arithmetic_holds_on_toy_fixture(toy_meld_root: Path) -> None:
    layout = resolve_meld_layout(toy_meld_root)
    dfs = {split: load_split_csv(sp.csv) for split, sp in layout.splits.items()}

    report = inventory_media(layout, dfs, report_dir=toy_meld_root / "reports")

    for split, stats in report.per_split.items():
        assert stats["unreconciled"] == [], f"{split}: {stats['unreconciled']}"
        assert stats["real_media"] == stats["matched"] + stats["orphan_media"]
        assert stats["labeled_rows"] == stats["matched"] + stats["missing_media"]
    assert "UNRECONCILED" not in report.to_markdown()


def test_injected_unreconcilable_count_produces_unreconciled_heading() -> None:
    report = MediaInventoryReport(
        generated_at="2026-01-01T00:00:00+00:00",
        total_mp4_count=10,
        mp4_dirs={"train_splits": 10},
        per_split={
            "train": {
                "mp4_files_total": 10,
                "appledouble_skipped": 0,
                "real_media": 10,
                "labeled_rows": 10,
                "matched": 9,
                "missing_media": 0,
                "orphan_media": 0,
                "orphan_examples": [],
                # Deliberately inconsistent: real_media (10) != matched (9) + orphan_media (0).
                "unreconciled": ["real_media (10) != matched (9) + orphan_media (0)"],
            }
        },
        cross_dir_duplicate_stems={},
        unaccounted_count=0,
        unaccounted_examples=[],
        conclusion="test conclusion",
    )

    markdown = report.to_markdown()
    assert "UNRECONCILED" in markdown
    assert "real_media (10) != matched (9) + orphan_media (0)" in markdown


def test_duplicate_stems_within_one_split_raise(tmp_path: Path) -> None:
    video_dir = tmp_path / "video_dir"
    # No mp4s directly in video_dir -> build_media_index falls back to
    # scanning one level deeper, where two different subdirectories produce
    # the same stem.
    (video_dir / "part_a").mkdir(parents=True)
    (video_dir / "part_b").mkdir(parents=True)
    (video_dir / "part_a" / "dia0_utt0.mp4").write_bytes(b"x")
    (video_dir / "part_b" / "dia0_utt0.mp4").write_bytes(b"y")

    with pytest.raises(ValueError, match="Duplicate media stem"):
        build_media_index(video_dir)


def test_same_stem_in_two_split_dirs_does_not_cross_match(tmp_path: Path) -> None:
    train_dir = tmp_path / "train_splits"
    dev_dir = tmp_path / "dev_splits_complete"
    _touch_mp4(train_dir, 0, 0)
    _touch_mp4(dev_dir, 0, 0)  # same stem, different split directory

    df = pd.DataFrame(
        [[1, "hi", "A", "neutral", "neutral", 0, 0, 1, 1, "00:00:00,000", "00:00:01,000"]],
        columns=MELD_COLUMNS,
    )

    train_matched, train_missing = map_utterances_to_media(df, "train", train_dir)
    dev_matched, dev_missing = map_utterances_to_media(df, "dev", dev_dir)

    assert not train_missing and not dev_missing
    assert train_matched[0].mp4 == train_dir / "dia0_utt0.mp4"
    assert dev_matched[0].mp4 == dev_dir / "dia0_utt0.mp4"
    assert train_matched[0].mp4 != dev_matched[0].mp4
