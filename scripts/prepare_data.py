"""Extract WAVs, loudness-normalize, and write JSONL manifests for one or more MELD splits.

Usage:
    python scripts/prepare_data.py --stage smoke [--split dev] [--limit 50]
    python scripts/prepare_data.py --stage full --split all [--force]
    python scripts/prepare_data.py --meld-root PATH --print-paths

`--print-paths` resolves the root and layout, prints each split's CSV path,
video directory, and (total/appledouble/real) .mp4 counts, then exits 0
WITHOUT reading any CSV or writing anything.

`--stage smoke` implies `--split dev` and `--limit 50` unless overridden, and
writes `{manifest_dir}/{split}_smoke.jsonl` — never `{split}.jsonl` — so a
partial smoke run can never clobber a full manifest.
`--stage full` processes the requested split(s) (default: all three) and
writes `{manifest_dir}/{split}.jsonl`.

Both stages call `seed_everything(cfg.run.seed)` at entry, log the resolved
config + git SHA via `emorec.utils.config.run_dir`, and append per-split
summary counts to `{report_dir}/excluded.json`. That summary keeps
NORMALISATION outcomes (`skipped_short`/`skipped_silent`/`clamped`/
`peak_limited` — the clip IS still written and IS in the manifest) in a
separate `normalisation` block from `exclusions_by_reason` (the clip is NOT
in the manifest) — the two must never be conflated.

After each split, `labeled_rows == written + excluded_total` is checked
explicitly; a mismatch prints a RECONCILIATION FAILURE block and exits
non-zero rather than silently drifting.
"""

import argparse
import json
import sys
from pathlib import Path

from emorec.data.meld_paths import count_mp4s, resolve_meld_layout, resolve_meld_root
from emorec.data.metadata import load_split_csv
from emorec.data.preprocess import process_split, write_manifest
from emorec.utils.config import load_config, run_dir
from emorec.utils.log import JsonlLogger
from emorec.utils.seed import seed_everything

ALL_SPLITS = ("train", "dev", "test")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], default=None)
    parser.add_argument("--split", choices=[*ALL_SPLITS, "all"], default=None)
    parser.add_argument("--meld-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Resolve and print paths + mp4 counts only; exit 0, read/write nothing.",
    )
    args = parser.parse_args(argv)
    if not args.print_paths and args.stage is None:
        parser.error("--stage is required unless --print-paths is given")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.run.seed)

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

    if args.stage == "smoke":
        if args.split == "all":
            raise SystemExit("--stage smoke does not support --split all; pick one split (default dev).")
        splits = [args.split or "dev"]
        limit = args.limit if args.limit is not None else 50
    else:
        splits = list(ALL_SPLITS) if args.split in (None, "all") else [args.split]
        limit = args.limit

    rdir = run_dir(cfg)
    run_logger = JsonlLogger(rdir / "prepare_data.log.jsonl")
    run_logger.log(
        {"event": "start", "stage": args.stage, "splits": splits, "limit": limit, "force": args.force, "root": str(root)}
    )

    report_dir = Path(cfg.data.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    excluded_path = report_dir / "excluded.json"
    summary: dict = json.loads(excluded_path.read_text(encoding="utf-8")) if excluded_path.exists() else {}

    manifest_dir = Path(cfg.data.manifest_dir)

    for split in splits:
        df = load_split_csv(layout.splits[split].csv)
        result = process_split(split, df, layout.splits[split].video_dir, cfg, limit=limit, force=args.force)

        manifest_name = f"{split}_smoke.jsonl" if args.stage == "smoke" else f"{split}.jsonl"
        write_manifest(result.rows, manifest_dir / manifest_name)

        by_reason: dict[str, int] = {}
        for exc in result.exclusions:
            by_reason[exc["reason"]] = by_reason.get(exc["reason"], 0) + 1

        labeled_count = len(df) if limit is None else min(len(df), limit)
        written = len(result.rows)
        excluded_total = len(result.exclusions)

        if labeled_count != written + excluded_total:
            print(
                "=== RECONCILIATION FAILURE ===\n"
                f"split={split!r} labeled_rows={labeled_count} written={written} "
                f"excluded_total={excluded_total} (written + excluded_total = "
                f"{written + excluded_total}, expected {labeled_count})\n"
                "===============================",
                file=sys.stderr,
            )
            run_logger.log(
                {
                    "event": "reconciliation_failure",
                    "split": split,
                    "labeled_rows": labeled_count,
                    "written": written,
                    "excluded_total": excluded_total,
                }
            )
            return 1

        # NORMALISATION: the clip IS written and IS in the manifest.
        # EXCLUSIONS: the clip is NOT in the manifest. Keep these visually
        # and structurally separate -- "skipped" must never read as "dropped".
        summary[split] = {
            "labeled_rows": labeled_count,
            "written": written,
            "excluded_total": excluded_total,
            "exclusions_by_reason": by_reason,
            "normalisation": dict(result.norm_counts),
        }
        run_logger.log({"event": "split_done", "split": split, **summary[split]})
        print(f"{split}: wrote {written} rows -> {manifest_dir / manifest_name} ({excluded_total} excluded)")

    excluded_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    run_logger.log({"event": "done"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
