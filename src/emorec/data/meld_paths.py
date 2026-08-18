"""MELD mirror discovery: root resolution and layout (CSV + video dir) discovery.

CONTRACT: no host- or platform-specific absolute path may appear here. The
mirror's exact directory layout is DISCOVERED at runtime by trying a fixed,
documented list of candidate relative paths/names — never assumed, never
hardcoded to one observed mirror. If the candidate lists stop matching a
future mirror, the fix is to extend the candidate lists, not to special-case
a path.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Candidate CSV paths, relative to the MELD root, tried in order. The first
# existing file wins. Note the real mirror's asymmetry: the train CSV sits
# under a `train/` subdirectory while dev/test CSVs sit at the root.
CSV_CANDIDATES: dict[str, list[str]] = {
    "train": [
        "train/train_sent_emo.csv",
        "train_sent_emo.csv",
        "train/train_sent_emo.csv.csv",
    ],
    "dev": [
        "dev_sent_emo.csv",
        "dev/dev_sent_emo.csv",
        "dev_splits_complete/dev_sent_emo.csv",
    ],
    "test": [
        "test_sent_emo.csv",
        "test/test_sent_emo.csv",
        "output_repeated_splits_test/test_sent_emo.csv",
    ],
}

# Candidate video-directory paths, relative to the MELD root, tried in
# order. The real mirror nests these one level deeper than their parent
# split directory (e.g. `train/train_splits`, not `train/`) — the parent
# directories hold zero .mp4 files directly, so the real nested path is
# listed FIRST. A candidate is accepted only if it is a directory that
# directly contains at least one real (non-AppleDouble) .mp4 file — see
# `_resolve_video_dir`. No recursive/one-level-deeper fallback is used for
# candidate acceptance; if none match, the fix is to extend this list.
VIDEO_DIR_CANDIDATES: dict[str, list[str]] = {
    "train": ["train/train_splits", "train_splits", "train"],
    "dev": ["dev/dev_splits_complete", "dev_splits_complete", "dev_splits", "dev"],
    "test": [
        "test/output_repeated_splits_test",
        "output_repeated_splits_test",
        "test_splits",
        "final_videos_test",
        "test",
    ],
}


def is_appledouble(path: Path) -> bool:
    """True for macOS AppleDouble resource forks ("._<name>").

    These are ~4 KB metadata sidecars written when a directory is archived
    on macOS. They match `*.mp4` globs but contain no video. The MELD
    mirror in use ships ~2,197 of them in the test split. They are counted
    and reported, never matched to a label, never deleted.
    """
    return path.name.startswith("._")


def count_mp4s(directory: Path) -> tuple[int, int, int]:
    """`(total, appledouble, real)` .mp4 file counts directly inside `directory` (non-recursive).

    `total == appledouble + real` always. Returns `(0, 0, 0)` if `directory`
    doesn't exist or isn't a directory (never raises on a missing path —
    candidate probing relies on this).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0, 0, 0
    files = list(directory.glob("*.mp4"))
    total = len(files)
    appledouble = sum(1 for f in files if is_appledouble(f))
    return total, appledouble, total - appledouble


@dataclass(frozen=True)
class SplitPaths:
    """Resolved paths for one MELD split."""

    split: str  # "train" | "dev" | "test"
    csv: Path
    video_dir: Path


@dataclass(frozen=True)
class MeldLayout:
    """Resolved MELD mirror layout: the root plus each split's paths."""

    root: Path
    splits: dict[str, SplitPaths]


_UNEXPANDED_VAR_HINT = (
    "This usually means an unexpanded shell variable (e.g. `--meld-root "
    "$MELD_ROOT` where MELD_ROOT was never set). Pass the absolute path "
    "literally."
)


def resolve_meld_root(cli_arg: str | None, cfg: DictConfig | None) -> Path:
    """Resolve the MELD root directory.

    Priority order: `cli_arg` -> `MELD_ROOT` environment variable ->
    `cfg.data.meld_root`. Each candidate is tried in that order; the first
    one that is an existing directory is returned immediately (so a
    higher-priority candidate that doesn't exist does NOT block a
    lower-priority one from being tried — only "none of them exist" is
    fatal).

    An explicitly-provided but empty/whitespace-only `cli_arg` or
    `MELD_ROOT` raises a DISTINCT `FileNotFoundError` naming the likely
    cause (an unexpanded shell variable) instead of falling through to the
    generic not-found message — a real run failed exactly this way and the
    generic message couldn't distinguish "not provided" from "provided as
    empty string".

    If nothing resolves, raises `FileNotFoundError` with a diagnostic block:
    the repr of all three inputs, `os.getcwd()`, the CLI argument's absolute
    resolved path (if any), and — if that path's parent exists — up to 20
    entries of the parent's contents, so a "the path IS right but resolution
    still failed" case is diagnosable from the error alone.
    """
    if cli_arg is not None and cli_arg.strip() == "":
        raise FileNotFoundError(
            f"--meld-root was provided but is empty (received: {cli_arg!r}). {_UNEXPANDED_VAR_HINT}"
        )

    env_val = os.environ.get("MELD_ROOT")
    if env_val is not None and env_val.strip() == "":
        raise FileNotFoundError(
            f"MELD_ROOT environment variable was provided but is empty (received: {env_val!r}). "
            f"{_UNEXPANDED_VAR_HINT}"
        )

    cfg_val = OmegaConf.select(cfg, "data.meld_root") if cfg is not None else None

    candidates: list[tuple[str, object]] = [
        ("--meld-root CLI argument", cli_arg),
        ("MELD_ROOT environment variable", env_val),
        ("cfg.data.meld_root", cfg_val),
    ]

    for _, value in candidates:
        if value and Path(str(value)).is_dir():
            return Path(str(value))

    lines = ["Could not resolve an existing MELD root directory. Checked, in priority order:"]
    lines += [f"  {name}: {value!r}" for name, value in candidates]
    lines.append(f"  cwd: {os.getcwd()!r}")

    if cli_arg:
        cli_abs = Path(cli_arg).resolve()
        lines.append(f"  absolute resolved path of --meld-root: {cli_abs}")
        parent = cli_abs.parent
        parent_exists = parent.exists()
        lines.append(f"  parent of that path exists: {parent_exists}")
        if parent_exists:
            try:
                entries = sorted(p.name for p in parent.iterdir())[:20]
            except OSError as exc:
                entries = [f"<could not list: {exc}>"]
            lines.append(f"  parent contents (up to 20): {entries}")

    lines.append(
        "Set one of: --meld-root <path> (CLI), MELD_ROOT=<path> (env var), "
        "or data.meld_root in configs/base.yaml."
    )
    raise FileNotFoundError("\n".join(lines))


def _mp4_dirs_with_counts(root: Path) -> list[tuple[Path, int, int, int]]:
    """Directories at depth 1 or 2 under `root` that directly contain >=1 .mp4 file, with `(total, appledouble, real)` counts."""
    found: list[tuple[Path, int, int, int]] = []
    depth1 = sorted(p for p in root.iterdir() if p.is_dir())
    candidate_dirs = list(depth1)
    for d in depth1:
        candidate_dirs.extend(sorted(p for p in d.iterdir() if p.is_dir()))
    for d in candidate_dirs:
        total, appledouble, real = count_mp4s(d)
        if total > 0:
            found.append((d, total, appledouble, real))
    return found


def _resolve_video_dir(root: Path, candidates: list[str]) -> Path | None:
    """First `candidates` entry (relative to root) that directly contains >=1 real (non-AppleDouble) .mp4 file.

    No recursive/one-level-deeper fallback: a candidate with zero direct
    real .mp4 files (empty, AppleDouble-only, or nonexistent) is rejected
    outright and the next candidate is tried.
    """
    for candidate in candidates:
        directory = root / candidate
        _total, _appledouble, real = count_mp4s(directory)
        if real > 0:
            return directory
    return None


def resolve_meld_layout(root: Path) -> MeldLayout:
    """Discover per-split CSV and video-directory paths under `root`.

    CSVs are located by trying `CSV_CANDIDATES[split]` in order and taking
    the first existing file. Video directories are located by trying
    `VIDEO_DIR_CANDIDATES[split]` in order and taking the first candidate
    that directly contains at least one real (non-AppleDouble) `.mp4` file
    — see `_resolve_video_dir`.

    Raises `FileNotFoundError` if a split's CSV can't be found among its
    candidates, or if a split's video directory can't be resolved — in the
    latter case the message enumerates every mp4-containing directory found
    under root, each with its (total, AppleDouble, real) counts, so the
    candidate list can be fixed. Never falls back to a global recursive glob
    for an unresolved split.
    """
    root = Path(root)
    splits: dict[str, SplitPaths] = {}

    for split in ("train", "dev", "test"):
        csv_path = None
        for candidate in CSV_CANDIDATES[split]:
            p = root / candidate
            if p.is_file():
                csv_path = p
                break
        if csv_path is None:
            tried = "\n".join(f"  {c}" for c in CSV_CANDIDATES[split])
            raise FileNotFoundError(
                f"Could not find a {split} CSV under {root}. Tried:\n{tried}"
            )

        video_dir = _resolve_video_dir(root, VIDEO_DIR_CANDIDATES[split])
        if video_dir is None:
            mp4_dirs = _mp4_dirs_with_counts(root)
            if mp4_dirs:
                found = "\n".join(
                    f"  {d.relative_to(root).as_posix()}: total={t} appledouble={a} real={r}"
                    for d, t, a, r in mp4_dirs
                )
            else:
                found = "  (no .mp4-containing directories found at depth <= 2 under root)"
            raise FileNotFoundError(
                f"Could not resolve a video directory for split {split!r} under {root}. "
                f"None of its candidate paths {VIDEO_DIR_CANDIDATES[split]} directly contained "
                f"a real (non-AppleDouble) .mp4 file. Directories that DO contain .mp4 files "
                f"(total/appledouble/real):\n{found}\n"
                "Fix VIDEO_DIR_CANDIDATES in meld_paths.py rather than guessing."
            )

        splits[split] = SplitPaths(split=split, csv=csv_path, video_dir=video_dir)

    return MeldLayout(root=root, splits=splits)
