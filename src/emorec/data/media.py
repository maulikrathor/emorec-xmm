"""Utterance <-> media (.mp4) resolution and mirror-wide inventory.

CONTRACT: media lookup for a split is ALWAYS scoped to that split's own
video directory. Never glob across the whole mirror — `dia0_utt0` exists in
every split (metadata.py's namespacing docstring explains why), so a global
glob would silently cross-match the wrong file. Orphan `.mp4` files (real
media with no labeled row) are counted and reported, never labeled, never
deleted. AppleDouble files ("._<name>.mp4" macOS resource-fork sidecars,
`is_appledouble`) are neither media nor orphans — they are their own
reported category and never participate in utterance matching.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from emorec.data.meld_paths import MeldLayout, is_appledouble

__all__ = [
    "MediaInventoryReport",
    "MediaMatch",
    "MediaMiss",
    "build_media_index",
    "inventory_media",
    "is_appledouble",
    "map_utterances_to_media",
]


def build_media_index(video_dir: Path) -> tuple[dict[str, Path], int]:
    """Index `video_dir`'s real (non-AppleDouble) `.mp4` files by file stem.

    Scans `video_dir` non-recursively first; if that finds zero `.mp4`
    files (real or AppleDouble), scans one level deeper (some mirror
    layouts nest videos one subdirectory down). AppleDouble files
    (`is_appledouble`) are excluded from the returned index entirely — they
    are never real media and must never satisfy an utterance's media
    lookup. Returns `(index, appledouble_skipped)`, where `index` keys are
    file stems exactly as they appear on disk (e.g. `"dia0_utt0"`) and
    `appledouble_skipped` is how many AppleDouble files were excluded.

    Raises `ValueError` if two REAL files anywhere in the scan produce the
    same stem — silently picking one would corrupt the utterance-to-media
    mapping without any visible symptom. An AppleDouble file that shares a
    stem with a real file is simply skipped, never treated as a collision.
    """
    video_dir = Path(video_dir)
    files = sorted(video_dir.glob("*.mp4"))
    if not files:
        files = sorted(video_dir.glob("*/*.mp4"))

    index: dict[str, Path] = {}
    appledouble_skipped = 0
    for f in files:
        if is_appledouble(f):
            appledouble_skipped += 1
            continue
        stem = f.stem
        if stem in index:
            raise ValueError(
                f"Duplicate media stem {stem!r} in {video_dir}: {index[stem]} and {f} "
                "both resolve to the same utterance — refusing to silently pick one."
            )
        index[stem] = f
    return index, appledouble_skipped


@dataclass(frozen=True)
class MediaMatch:
    dialogue_id: int
    utterance_id: int
    mp4: Path


@dataclass(frozen=True)
class MediaMiss:
    dialogue_id: int
    utterance_id: int
    expected_stem: str


def _match_with_index(
    df: pd.DataFrame, index: dict[str, Path]
) -> tuple[list[MediaMatch], list[MediaMiss]]:
    matched: list[MediaMatch] = []
    missing: list[MediaMiss] = []
    for row in df.itertuples():
        stem = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"
        path = index.get(stem)
        if path is None:
            missing.append(MediaMiss(row.Dialogue_ID, row.Utterance_ID, stem))
        else:
            matched.append(MediaMatch(row.Dialogue_ID, row.Utterance_ID, path))
    return matched, missing


def map_utterances_to_media(
    df: pd.DataFrame, split: str, video_dir: Path
) -> tuple[list[MediaMatch], list[MediaMiss]]:
    """Match each row of `df` to a media file in `video_dir`, that split's directory only.

    Expected stem is `f"dia{Dialogue_ID}_utt{Utterance_ID}"`. `split` is
    accepted for signature clarity/logging symmetry with the rest of the
    module but lookup itself never leaves `video_dir` — this is what
    prevents cross-split filename collisions. Media lookup goes through
    `build_media_index`, so AppleDouble files can never satisfy a match.
    Returns `(matched, missing)`; `missing` rows have no real media file and
    must be excluded (`MEDIA_MISSING`) downstream, never silently dropped.
    """
    del split  # lookup is scoped by video_dir alone; see docstring
    index, _appledouble_skipped = build_media_index(video_dir)
    return _match_with_index(df, index)


def _all_mp4_paths_by_dir(root: Path) -> dict[Path, list[Path]]:
    """Every directory anywhere under `root` that directly contains `.mp4` files, mapped to their full paths.

    A single `os.walk` pass so each physical file is attributed to exactly
    one (its containing) directory — summing the per-directory counts gives
    the mirror's true total `.mp4` count with no double counting from
    nested directories. Includes AppleDouble files; callers separate them
    out via `is_appledouble`.
    """
    result: dict[Path, list[Path]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        mp4s = sorted(Path(dirpath) / f for f in filenames if f.lower().endswith(".mp4"))
        if mp4s:
            result[Path(dirpath)] = mp4s
    return result


@dataclass
class MediaInventoryReport:
    generated_at: str
    total_mp4_count: int
    mp4_dirs: dict[str, int]  # relative-to-root path -> raw .mp4 count (junk included)
    per_split: dict[str, dict[str, Any]]
    cross_dir_duplicate_stems: dict[str, list[str]]  # stem -> dirs (relative to root) it appears in
    unaccounted_count: int  # mp4s outside the 3 resolved split dirs
    unaccounted_examples: list[str]  # up to 20 example paths, relative to root
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# MELD media inventory report",
            "",
            f"Generated: {self.generated_at}",
            "",
            f"Total .mp4 files found under root (junk included): {self.total_mp4_count}",
            "",
            "## Every .mp4-containing directory under root",
            "",
        ]
        for rel, count in sorted(self.mp4_dirs.items()):
            lines.append(f"- {rel}: {count}")

        lines += ["", "## Per split", ""]
        for split, stats in self.per_split.items():
            lines.append(f"### {split}")
            lines.append(f"- mp4 files total (raw glob, junk included): {stats['mp4_files_total']}")
            lines.append(f"- appledouble_skipped: {stats['appledouble_skipped']}")
            lines.append(f"- real_media: {stats['real_media']}")
            lines.append(f"- labeled_rows: {stats['labeled_rows']}")
            lines.append(f"- matched: {stats['matched']}")
            lines.append(f"- missing_media: {stats['missing_media']}")
            lines.append(f"- orphan_media: {stats['orphan_media']}")
            if stats["orphan_examples"]:
                lines.append(f"  orphan examples: {stats['orphan_examples']}")
            if stats["unreconciled"]:
                lines.append("")
                lines.append("#### UNRECONCILED")
                for problem in stats["unreconciled"]:
                    lines.append(f"- {problem}")
            lines.append("")

        lines += [
            "## Cross-directory duplicate stems",
            "",
            f"Stems appearing in more than one .mp4-containing directory: {len(self.cross_dir_duplicate_stems)}",
        ]
        for stem, dirs in list(self.cross_dir_duplicate_stems.items())[:20]:
            lines.append(f"- {stem}: {dirs}")

        lines += ["", "## Conclusion", "", self.conclusion]

        if self.unaccounted_count > 0:
            lines += [
                "",
                "### UNACCOUNTED",
                "",
                f"UNACCOUNTED: {self.unaccounted_count} files",
            ]
            for example in self.unaccounted_examples:
                lines.append(f"- {example}")

        return "\n".join(lines) + "\n"


def inventory_media(
    layout: MeldLayout,
    dfs: dict[str, pd.DataFrame],
    report_dir: Path | str = Path("artifacts/reports"),
) -> MediaInventoryReport:
    """Reconcile "how many .mp4 files exist" against "how many labeled utterances exist", empirically.

    Scans every `.mp4`-containing directory under `layout.root` (not just
    the three resolved split directories), matches each split's labeled
    rows against REAL media in ITS OWN resolved video directory only
    (AppleDouble files are their own category, never orphans, never
    matches), and reports cross-directory duplicate stems.

    Per split, reports `mp4_files_total` (raw, junk included),
    `appledouble_skipped`, `real_media`, `labeled_rows`, `matched`,
    `missing_media`, `orphan_media`. Two arithmetic identities are checked
    per split — `real_media == matched + orphan_media` and `labeled_rows ==
    matched + missing_media` — and any violation is listed under an
    `unreconciled` key and rendered under an "UNRECONCILED" heading; this
    never raises, the report itself is the deliverable.

    The whole-mirror conclusion states `total_mp4_count` against the sum of
    `real_media` + `appledouble_skipped` across the three resolved split
    dirs; any residual (mp4s living outside all three resolved dirs) is
    reported as `unaccounted_count`/`unaccounted_examples` and rendered
    under "UNACCOUNTED: <n> files" with up to 20 example paths — never
    explained away with a guess. Never assigns a label to an orphan, never
    deletes anything. Writes `{report_dir}/media_inventory.md` and `.json`.
    """
    root = Path(layout.root)
    mp4_by_dir = _all_mp4_paths_by_dir(root)
    total_mp4_count = sum(len(paths) for paths in mp4_by_dir.values())
    mp4_dirs_rel = {d.relative_to(root).as_posix(): len(paths) for d, paths in mp4_by_dir.items()}

    stem_to_dirs: dict[str, list[str]] = {}
    for d, paths in mp4_by_dir.items():
        rel = d.relative_to(root).as_posix()
        for p in paths:
            stem_to_dirs.setdefault(p.stem, []).append(rel)
    cross_dir_duplicate_stems = {s: dirs for s, dirs in stem_to_dirs.items() if len(dirs) > 1}

    per_split: dict[str, dict[str, Any]] = {}
    resolved_real_total = 0
    resolved_appledouble_total = 0
    for split, sp in layout.splits.items():
        df = dfs[split]
        index, appledouble_skipped = build_media_index(sp.video_dir)
        real_media = len(index)
        mp4_files_total = real_media + appledouble_skipped

        matched, missing = _match_with_index(df, index)
        matched_stems = {f"dia{m.dialogue_id}_utt{m.utterance_id}" for m in matched}
        orphan_stems = sorted(set(index) - matched_stems)

        labeled_rows = len(df)
        matched_n = len(matched)
        missing_n = len(missing)
        orphan_n = len(orphan_stems)

        unreconciled: list[str] = []
        if real_media != matched_n + orphan_n:
            unreconciled.append(
                f"real_media ({real_media}) != matched ({matched_n}) + orphan_media ({orphan_n})"
            )
        if labeled_rows != matched_n + missing_n:
            unreconciled.append(
                f"labeled_rows ({labeled_rows}) != matched ({matched_n}) + missing_media ({missing_n})"
            )

        resolved_real_total += real_media
        resolved_appledouble_total += appledouble_skipped

        per_split[split] = {
            "mp4_files_total": mp4_files_total,
            "appledouble_skipped": appledouble_skipped,
            "real_media": real_media,
            "labeled_rows": labeled_rows,
            "matched": matched_n,
            "missing_media": missing_n,
            "orphan_media": orphan_n,
            "orphan_examples": orphan_stems[:20],
            "unreconciled": unreconciled,
        }

    resolved_total = resolved_real_total + resolved_appledouble_total
    unaccounted_count = total_mp4_count - resolved_total

    resolved_dir_relpaths = {sp.video_dir.relative_to(root) for sp in layout.splits.values()}
    unaccounted_examples: list[str] = []
    if unaccounted_count > 0:
        outside_paths = sorted(
            p
            for d, paths in mp4_by_dir.items()
            if d.relative_to(root) not in resolved_dir_relpaths
            for p in paths
        )
        unaccounted_examples = [p.relative_to(root).as_posix() for p in outside_paths[:20]]

    labeled_total = sum(len(df) for df in dfs.values())
    matched_total = sum(stats["matched"] for stats in per_split.values())
    orphan_total = sum(stats["orphan_media"] for stats in per_split.values())
    missing_total = sum(stats["missing_media"] for stats in per_split.values())

    conclusion_lines = [
        f"Total .mp4 across mirror: {total_mp4_count}",
        "  = sum(real_media per split) ("
        + ", ".join(f"{split}={per_split[split]['real_media']}" for split in per_split)
        + f" -> {resolved_real_total}) + sum(appledouble per split) ("
        + ", ".join(f"{split}={per_split[split]['appledouble_skipped']}" for split in per_split)
        + f" -> {resolved_appledouble_total}) = {resolved_total}",
        f"Labeled utterances: {labeled_total}",
        "  = sum(labeled_rows per split) ("
        + ", ".join(f"{split}={per_split[split]['labeled_rows']}" for split in per_split)
        + f" -> {labeled_total})",
        (
            f"Of the {resolved_real_total} real media files, {matched_total} matched a labeled "
            f"row, {orphan_total} did not (orphan_media), and {missing_total} labeled row(s) had "
            "no real media (missing_media)."
        ),
        (
            f"gap = sum(orphan_media) ({orphan_total}) + sum(appledouble) "
            f"({resolved_appledouble_total}) = {orphan_total + resolved_appledouble_total}"
        ),
    ]
    if unaccounted_count > 0:
        conclusion_lines.append(
            f"Residual outside the three resolved split directories: {unaccounted_count} "
            f"file(s) ({total_mp4_count} total - {resolved_total} in resolved dirs). See "
            "UNACCOUNTED below — not explained here, only counted."
        )
    else:
        conclusion_lines.append(
            f"{resolved_total} (resolved split dirs) equals the mirror total ({total_mp4_count}); "
            "no residual outside the three resolved split directories."
        )
    conclusion = "\n".join(conclusion_lines)

    report = MediaInventoryReport(
        generated_at=datetime.now(UTC).isoformat(),
        total_mp4_count=total_mp4_count,
        mp4_dirs=mp4_dirs_rel,
        per_split=per_split,
        cross_dir_duplicate_stems=cross_dir_duplicate_stems,
        unaccounted_count=unaccounted_count,
        unaccounted_examples=unaccounted_examples,
        conclusion=conclusion,
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "media_inventory.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "media_inventory.md").write_text(report.to_markdown(), encoding="utf-8")

    return report
