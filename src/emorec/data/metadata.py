"""MELD CSV loading and metadata validation/reporting.

CONTRACT: hard validations (V1-V7) raise `ValueError` and must never be
bypassed — they are the pipeline's first line of defense against silently
training on corrupted or mislabeled data. Report-only findings (R1-R6) never
raise; they are written to `artifacts/reports/meld_metadata.{md,json}` for a
human to read, not gates.

Label ids are ALWAYS taken from `emorec.data.labels.EMOTION_TO_ID` — this
module never derives or sorts its own mapping.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from emorec.data.labels import EMOTION_NAMES, EMOTION_TO_ID
from emorec.data.text_clean import normalize_text

logger = logging.getLogger(__name__)

# The real MELD mirror's exact per-split row counts (BACKLOG T-04 / the
# actual CSVs). This is the production default for V1; it is an explicit
# parameter (not a bare module constant baked into the check) purely so the
# check function remains unit-testable against small synthetic data — see
# TRIBAL_KNOWLEDGE.md Session Notes for the rationale.
MELD_EXPECTED_ROW_COUNTS: dict[str, int] = {"train": 9989, "dev": 1109, "test": 2610}


def load_split_csv(path: Path) -> pd.DataFrame:
    """Load one MELD split CSV, normalized and deterministically sorted.

    Reads with `encoding="utf-8"`; on `UnicodeDecodeError` retries with
    `"latin-1"` (MELD CSVs are known to contain stray Windows-1252 bytes —
    TRIBAL_KNOWLEDGE.md §1.3) and logs which encoding won. Applies
    `normalize_text` to `Utterance` and `.strip().lower()` to `Emotion`.
    Sorts rows by `(Dialogue_ID, Utterance_ID)` and resets the index, so two
    CSVs with the same rows in different order load to identical DataFrames.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
        logger.info("loaded %s with encoding=utf-8", path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
        logger.warning("loaded %s with encoding=latin-1 (utf-8 failed)", path)

    df["Utterance"] = df["Utterance"].map(normalize_text)
    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.lower()
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"], kind="stable").reset_index(drop=True)
    return df


def make_utt_id(split: str, dialogue_id: int, utterance_id: int) -> str:
    """Build the split-namespaced utterance id `f"{split}/dia{dialogue_id}_utt{utterance_id}"`.

    The split prefix exists because MELD re-indexes `Dialogue_ID` from 0
    within every split independently — `dia0_utt0` exists in train, dev, AND
    test. Without the prefix, ids collide across splits in every manifest,
    every audio-cache path, and every attribution artifact path
    (TRIBAL_KNOWLEDGE.md §1.4). Never construct an utt_id any other way.
    """
    return f"{split}/dia{dialogue_id}_utt{utterance_id}"


# --------------------------------------------------------------------------
# Hard validations (V1-V7). Each raises ValueError with the observed value.
# Exposed as separate functions (not just inlined in validate_metadata) so
# each can be unit-tested in isolation against small synthetic DataFrames
# without needing data that satisfies every other check too.
# --------------------------------------------------------------------------


def _check_row_counts(dfs: dict[str, pd.DataFrame], expected: dict[str, int]) -> None:
    """V1: per-split row counts match `expected` exactly."""
    mismatches = {
        split: (len(dfs[split]) if split in dfs else None, count)
        for split, count in expected.items()
        if split not in dfs or len(dfs[split]) != count
    }
    if mismatches:
        detail = ", ".join(
            f"{split}: observed={obs} expected={exp}" for split, (obs, exp) in mismatches.items()
        )
        raise ValueError(f"V1: row count mismatch — {detail}")


def _check_emotions_known(dfs: dict[str, pd.DataFrame]) -> None:
    """V2: every Emotion value is a key of EMOTION_TO_ID."""
    valid = set(EMOTION_TO_ID)
    for split, df in dfs.items():
        bad = sorted(set(df["Emotion"]) - valid)
        if bad:
            raise ValueError(
                f"V2: {split} contains emotion value(s) not in EMOTION_TO_ID: {bad} "
                f"(valid values: {sorted(valid)})"
            )


def _check_all_emotions_in_train(dfs: dict[str, pd.DataFrame]) -> None:
    """V3: all 7 emotions are present in the train split."""
    if "train" not in dfs:
        raise ValueError("V3: no 'train' split provided")
    present = set(dfs["train"]["Emotion"])
    missing = set(EMOTION_TO_ID) - present
    if missing:
        raise ValueError(f"V3: train split is missing emotion class(es): {sorted(missing)}")


def _check_unique_dialogue_utterance(dfs: dict[str, pd.DataFrame]) -> None:
    """V4: (Dialogue_ID, Utterance_ID) is unique within each split."""
    for split, df in dfs.items():
        pairs = list(zip(df["Dialogue_ID"], df["Utterance_ID"], strict=True))
        dupes = sorted({p for p, c in Counter(pairs).items() if c > 1})
        if dupes:
            raise ValueError(f"V4: {split} has duplicate (Dialogue_ID, Utterance_ID) pairs: {dupes[:10]}")


def _check_id_types(dfs: dict[str, pd.DataFrame]) -> None:
    """V5: Dialogue_ID and Utterance_ID are non-negative integers, no NaN."""
    for split, df in dfs.items():
        for col in ("Dialogue_ID", "Utterance_ID"):
            if df[col].isna().any():
                raise ValueError(f"V5: {split}.{col} contains NaN")
            is_int_like = pd.api.types.is_integer_dtype(df[col]) or (
                pd.api.types.is_float_dtype(df[col]) and (df[col] == df[col].astype(int)).all()
            )
            if not is_int_like:
                raise ValueError(f"V5: {split}.{col} is not integer-valued (dtype={df[col].dtype})")
            if (df[col].astype(int) < 0).any():
                bad = sorted(df.loc[df[col].astype(int) < 0, col].tolist())
                raise ValueError(f"V5: {split}.{col} contains negative value(s): {bad}")


def _check_no_nulls(dfs: dict[str, pd.DataFrame]) -> None:
    """V6: no NaN in Utterance, Emotion, Dialogue_ID, Utterance_ID."""
    for split, df in dfs.items():
        for col in ("Utterance", "Emotion", "Dialogue_ID", "Utterance_ID"):
            n_na = int(df[col].isna().sum())
            if n_na:
                raise ValueError(f"V6: {split}.{col} has {n_na} NaN value(s)")


def _check_global_utt_id_uniqueness(utt_ids: list[str]) -> None:
    """V7: every id in `utt_ids` is globally unique.

    Called by `validate_metadata` with ids built through `make_utt_id`
    (properly split-namespaced), where it should never fire under correct
    construction. Exposed standalone so a test can feed it deliberately
    NON-namespaced ids and demonstrate the exact collision namespacing
    exists to prevent (TRIBAL_KNOWLEDGE.md §1.4).
    """
    dupes = sorted({uid for uid, c in Counter(utt_ids).items() if c > 1})
    if dupes:
        raise ValueError(f"V7: duplicate utt_id(s) found (namespacing bypassed?): {dupes[:10]}")


# --------------------------------------------------------------------------
# Report-only findings (R1-R6) + MetadataReport
# --------------------------------------------------------------------------


def _summary_stats(s: pd.Series) -> dict[str, float]:
    return {
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "min": float(s.min()),
        "max": float(s.max()),
    }


@dataclass
class MetadataReport:
    """All V1-V7 pass/fail is implicit: this report is only built after every hard check passes."""

    generated_at: str
    row_counts: dict[str, int]
    emotion_counts: dict[str, dict[str, int]]  # R1
    emotion_pct: dict[str, dict[str, float]]  # R1
    utterance_length_stats: dict[str, dict[str, dict[str, float]]]  # R2
    turns_per_dialogue_stats: dict[str, dict[str, float]]  # R3
    split_integrity_duplicates: dict[str, Any]  # R4
    episode_overlap: dict[str, Any]  # R5
    speaker_stats: dict[str, dict[str, Any]]  # R6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# MELD metadata report",
            "",
            f"Generated: {self.generated_at}",
            "",
            "## Row counts",
            "",
        ]
        for split, n in self.row_counts.items():
            lines.append(f"- {split}: {n}")

        lines += ["", "## R1 — per-split emotion distribution", ""]
        for split in self.emotion_counts:
            lines.append(f"### {split}")
            for emotion in EMOTION_NAMES:
                c = self.emotion_counts[split].get(emotion, 0)
                p = self.emotion_pct[split].get(emotion, 0.0)
                lines.append(f"- {emotion}: {c} ({p}%)")
            lines.append("")

        lines += ["## R2 — utterance length statistics", ""]
        for split, stats in self.utterance_length_stats.items():
            lines.append(f"- {split}: chars={stats['chars']}, tokens={stats['tokens']}")

        lines += ["", "## R3 — turns per dialogue", ""]
        for split, stats in self.turns_per_dialogue_stats.items():
            lines.append(f"- {split}: {stats}")

        lines += ["", "## R4 — split integrity (Season, Episode, StartTime, EndTime) duplicates", ""]
        lines.append(f"Cross-split duplicate keys: {self.split_integrity_duplicates['count']}")
        for example in self.split_integrity_duplicates["examples"]:
            lines.append(f"- {example['key']}: {example['occurrences']}")

        lines += ["", "## R5 — episode overlap across splits", ""]
        lines.append(f"(Season, Episode) pairs appearing in more than one split: {self.episode_overlap['count']}")
        lines.append(self.episode_overlap["note"])

        lines += ["", "## R6 — speakers", ""]
        for split, stats in self.speaker_stats.items():
            lines.append(f"### {split}")
            lines.append(f"distinct speakers: {stats['distinct_speakers']}")
            for row in stats["top_10"]:
                lines.append(f"- {row['speaker']}: {row['count']}")
            lines.append("")

        return "\n".join(lines) + "\n"


def validate_metadata(
    dfs: dict[str, pd.DataFrame],
    report_dir: Path | str = Path("artifacts/reports"),
    expected_row_counts: dict[str, int] | None = None,
) -> MetadataReport:
    """Run hard validations V1-V7, then compute and write report-only findings R1-R6.

    `dfs` maps split name -> DataFrame as returned by `load_split_csv`.
    `expected_row_counts` defaults to `MELD_EXPECTED_ROW_COUNTS` (the real
    mirror's counts); pass a dict matching your data's actual sizes to
    exercise this function against synthetic/toy data. Raises `ValueError`
    on the first hard-check failure (V1..V7, in that order) — no report is
    written in that case. On success, writes
    `{report_dir}/meld_metadata.md` and `.json` and returns the report.
    """
    expected = expected_row_counts if expected_row_counts is not None else MELD_EXPECTED_ROW_COUNTS

    _check_row_counts(dfs, expected)
    _check_emotions_known(dfs)
    _check_all_emotions_in_train(dfs)
    _check_unique_dialogue_utterance(dfs)
    _check_id_types(dfs)
    _check_no_nulls(dfs)
    all_utt_ids = [
        make_utt_id(split, row.Dialogue_ID, row.Utterance_ID)
        for split, df in dfs.items()
        for row in df.itertuples()
    ]
    _check_global_utt_id_uniqueness(all_utt_ids)

    # R1
    emotion_counts: dict[str, dict[str, int]] = {}
    emotion_pct: dict[str, dict[str, float]] = {}
    for split, df in dfs.items():
        counts = df["Emotion"].value_counts().to_dict()
        counts = {e: int(counts.get(e, 0)) for e in EMOTION_NAMES}
        total = len(df)
        emotion_counts[split] = counts
        emotion_pct[split] = {e: round(100.0 * c / total, 2) if total else 0.0 for e, c in counts.items()}

    # R2
    utterance_length_stats: dict[str, dict[str, dict[str, float]]] = {}
    for split, df in dfs.items():
        chars = df["Utterance"].str.len()
        tokens = df["Utterance"].str.split().str.len()
        utterance_length_stats[split] = {"chars": _summary_stats(chars), "tokens": _summary_stats(tokens)}

    # R3
    turns_per_dialogue_stats: dict[str, dict[str, float]] = {}
    for split, df in dfs.items():
        turns_per_dialogue_stats[split] = _summary_stats(df.groupby("Dialogue_ID").size())

    # R4
    key_to_occurrences: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for split, df in dfs.items():
        for row in df.itertuples():
            key = (row.Season, row.Episode, row.StartTime, row.EndTime)
            key_to_occurrences[key].append(
                {"split": split, "utt_id": make_utt_id(split, row.Dialogue_ID, row.Utterance_ID)}
            )
    cross_split_dupes = {
        k: v for k, v in key_to_occurrences.items() if len({o["split"] for o in v}) > 1
    }
    split_integrity_duplicates = {
        "count": len(cross_split_dupes),
        "examples": [
            {"key": list(k), "occurrences": v} for k, v in list(cross_split_dupes.items())[:10]
        ],
    }

    # R5
    episode_to_splits: dict[tuple, set[str]] = defaultdict(set)
    for split, df in dfs.items():
        for season, episode in zip(df["Season"], df["Episode"], strict=True):
            episode_to_splits[(season, episode)].add(split)
    overlapping = {k: sorted(v) for k, v in episode_to_splits.items() if len(v) > 1}
    episode_overlap = {
        "count": len(overlapping),
        "examples": [list(k) for k in list(overlapping)[:10]],
        "note": (
            "MELD's official splits are drawn dialogue-wise from overlapping episodes, "
            "so this is expected dataset structure and NOT a bug introduced by this "
            "pipeline. Evidence for docs/KNOWN_LIMITATIONS.md #6 (speaker priors: the "
            "same episodes/speakers recur across splits)."
        ),
    }

    # R6
    speaker_stats: dict[str, dict[str, Any]] = {}
    for split, df in dfs.items():
        counts = df["Speaker"].value_counts()
        speaker_stats[split] = {
            "distinct_speakers": int(counts.shape[0]),
            "top_10": [{"speaker": s, "count": int(c)} for s, c in counts.head(10).items()],
        }

    report = MetadataReport(
        generated_at=datetime.now(UTC).isoformat(),
        row_counts={split: len(df) for split, df in dfs.items()},
        emotion_counts=emotion_counts,
        emotion_pct=emotion_pct,
        utterance_length_stats=utterance_length_stats,
        turns_per_dialogue_stats=turns_per_dialogue_stats,
        split_integrity_duplicates=split_integrity_duplicates,
        episode_overlap=episode_overlap,
        speaker_stats=speaker_stats,
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "meld_metadata.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "meld_metadata.md").write_text(report.to_markdown(), encoding="utf-8")

    return report
