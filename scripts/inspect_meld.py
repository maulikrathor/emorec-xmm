"""Stage 1 (metadata validation) + Stage 2 (media inventory) only — no audio is written.

Usage:
    python scripts/inspect_meld.py --meld-root PATH [--config configs/base.yaml]
    python scripts/inspect_meld.py --meld-root PATH --print-paths

`--print-paths` resolves the root and layout, prints each split's CSV path,
video directory, and (total/appledouble/real) .mp4 counts, then exits 0
WITHOUT reading any CSV or writing anything — use it to independently verify
path resolution on a new host before running any real work.

Otherwise exits non-zero if the MELD root can't be resolved, or if any hard
metadata validation (V1-V7) fails. Intended to run in well under a minute
even against the full real mirror — it only reads CSVs and lists
directories, it never touches audio/video content.
"""

import argparse
import sys
from pathlib import Path

from emorec.data.media import inventory_media
from emorec.data.meld_paths import count_mp4s, resolve_meld_layout, resolve_meld_root
from emorec.data.metadata import load_split_csv, validate_metadata
from emorec.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meld-root", default=None, help="Path to the MELD mirror root.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to a base config YAML.")
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Resolve and print paths + mp4 counts only; exit 0, read/write nothing.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        root = resolve_meld_root(args.meld_root, cfg)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Resolved MELD root: {root.resolve()}")

    try:
        layout = resolve_meld_layout(root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.print_paths:
        for split, sp in layout.splits.items():
            total, appledouble, real = count_mp4s(sp.video_dir)
            print(
                f"  {split}: csv={sp.csv} video_dir={sp.video_dir} "
                f"mp4(total={total}, appledouble={appledouble}, real={real})"
            )
        return 0

    for split, sp in layout.splits.items():
        print(f"  {split}: csv={sp.csv} video_dir={sp.video_dir}")

    dfs = {split: load_split_csv(sp.csv) for split, sp in layout.splits.items()}

    report_dir = Path(cfg.data.report_dir)

    print("\n--- Stage 1: metadata validation (V1-V7) ---")
    try:
        validate_metadata(dfs, report_dir=report_dir)
    except ValueError as exc:
        print(f"Metadata validation FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Metadata validation passed. Reports written to {report_dir}")

    print("\n--- Stage 2: media inventory ---")
    inventory_media(layout, dfs, report_dir=report_dir)
    print(f"Media inventory written to {report_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
