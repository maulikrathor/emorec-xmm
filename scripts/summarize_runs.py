"""Turn artifacts/runs/* into one markdown results table for the README and review slides.

CONTRACT: every number printed comes from an actual file on disk (a run's
`run_meta.json`/`metrics.json`/`classification_report.txt`, or a
`metrics_*.json` written by `scripts/evaluate.py`, or a real dev manifest
referenced by a run's own `run_meta.json`). If a field can't be found, the
table prints "-" for it -- this script never invents, estimates, or
placeholders a number (docs/CLAUDE.md commandment #9).

Usage:
    python scripts/summarize_runs.py [--runs-dir artifacts/runs]
        [--out artifacts/reports/pilot_results.md]

A "run" is any subdirectory of `--runs-dir` containing BOTH `run_meta.json`
and `metrics.json` (both written by `emorec.training.trainer.fit`, via
`scripts/train.py`). The trivial all-neutral baseline row is computed live
from the first available run's dev manifest (`run_meta.json ->
split_names.dev`), not read from a file -- there is no trivial-baseline run
directory, only real trained-model runs.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from emorec.data.labels import EMOTION_NAMES
from emorec.training.metrics import trivial_baseline_metrics

_RUN_DIR_TIMESTAMP_RE = re.compile(r"-(\d{8}T\d{6}Z)$")
_REPORT_LINE_RE_TOKENS = 5  # name, precision, recall, f1-score, support


def _find_runs(runs_dir: Path) -> list[Path]:
    """Every subdirectory of `runs_dir` with BOTH run_meta.json and metrics.json, sorted by name."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        child
        for child in runs_dir.iterdir()
        if child.is_dir() and (child / "run_meta.json").exists() and (child / "metrics.json").exists()
    )


def _read_run_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))


def _read_epoch_rows(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "metrics.json").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _best_epoch_row(epoch_rows: list[dict]) -> dict | None:
    """The epoch row corresponding to `best.pt`.

    Mirrors `emorec.training.trainer.fit`'s own best-tracking exactly: it
    saves `best.pt` only on a STRICT dev-weighted-F1 improvement, so the
    surviving checkpoint is from the FIRST epoch reaching the eventual max.
    Python's `max()` already returns the first element on ties, so this is
    the correct (not just "a", the SAME) epoch `fit()` saved.
    """
    if not epoch_rows:
        return None
    return max(epoch_rows, key=lambda r: r.get("dev_weighted_f1", float("-inf")))


def _run_date_from_dirname(run_dir: Path) -> str:
    """The UTC timestamp `emorec.utils.config.run_dir` embedded in the run directory's own name.

    Never invented: `run_dir()` names directories `f"{cfg.run.name}-{UTC
    timestamp}"`; this just reads that timestamp back out of the name
    (`run_meta.json` itself carries no date field to read instead).
    """
    match = _RUN_DIR_TIMESTAMP_RE.search(run_dir.name)
    return match.group(1) if match else "-"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return str(value)


def _per_class_f1_from_json(run_dir: Path) -> dict[str, float] | None:
    """Prefer a structured `metrics_*.json` (written by scripts/evaluate.py) if one exists."""
    for path in sorted(run_dir.glob("metrics_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("per_class_f1"), dict):
            return data["per_class_f1"]
    return None


def _per_class_f1_from_report(run_dir: Path) -> dict[str, float] | None:
    """Fall back to parsing classification_report.txt (written by every scripts/train.py run)."""
    path = run_dir / "classification_report.txt"
    if not path.exists():
        return None
    result: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == _REPORT_LINE_RE_TOKENS and parts[0] in EMOTION_NAMES:
            try:
                result[parts[0]] = float(parts[3])
            except ValueError:
                continue
    return result or None


def _per_class_f1(run_dir: Path) -> dict[str, float] | None:
    return _per_class_f1_from_json(run_dir) or _per_class_f1_from_report(run_dir)


def _count_manifest_rows(manifest_path: Path) -> int:
    return sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _trivial_baseline(run_dirs: list[Path], metas: list[dict]) -> tuple[dict | None, str | None]:
    """Compute the trivial all-neutral baseline on the first run's dev manifest actually found on disk."""
    for meta in metas:
        dev_path = meta.get("split_names", {}).get("dev")
        if not dev_path:
            continue
        p = Path(dev_path)
        if not p.is_file():
            continue
        y_true = [
            json.loads(line)["label"]
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if y_true:
            return trivial_baseline_metrics(y_true), str(p)
    return None, None


def build_markdown(runs_dir: Path) -> str:
    run_dirs = _find_runs(runs_dir)
    lines: list[str] = ["# Pilot results", ""]

    if not run_dirs:
        lines.append(
            f"_No runs found under `{runs_dir}` (looked for subdirectories containing both "
            "`run_meta.json` and `metrics.json`)._"
        )
        return "\n".join(lines) + "\n"

    metas = [_read_run_meta(d) for d in run_dirs]

    seeds = sorted({m.get("seed") for m in metas if m.get("seed") is not None})
    subset_sizes = sorted(
        {m.get("subset_size") for m in metas}, key=lambda x: (x is not None, x if x is not None else 0)
    )
    git_shas = sorted({m.get("git_sha", "unknown") for m in metas})
    dates = sorted({_run_date_from_dirname(d) for d in run_dirs})
    split_name_variants = sorted({json.dumps(m.get("split_names", {}), sort_keys=True) for m in metas})

    trivial, dev_manifest_used = _trivial_baseline(run_dirs, metas)
    dev_size = "-"
    if dev_manifest_used:
        try:
            dev_size = str(_count_manifest_rows(Path(dev_manifest_used)))
        except OSError:
            dev_size = "-"

    lines += ["## Run header", ""]
    if len(split_name_variants) == 1:
        lines.append(f"- split names: `{json.loads(split_name_variants[0])}`")
    else:
        lines.append("- split names: varies across runs (see each run's own `run_meta.json`)")
    dev_manifest_note = f" (from `{dev_manifest_used}`)" if dev_manifest_used else ""
    lines.append(f"- dev set size: {dev_size}{dev_manifest_note}")
    subset_str = ", ".join("full" if s is None else str(s) for s in subset_sizes) if subset_sizes else "-"
    lines.append(f"- subset size(s): {subset_str}")
    lines.append(f"- seed(s): {', '.join(str(s) for s in seeds) if seeds else '-'}")
    lines.append(f"- date(s) (UTC, from run directory names): {', '.join(dates) if dates else '-'}")
    lines.append(f"- git SHA(s): {', '.join(git_shas) if git_shas else '-'}")
    lines.append("")

    lines += [
        "## Results",
        "",
        "| model | train subset size | seed | epochs run | dev accuracy | dev macro-F1 | dev weighted-F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for run_dir, meta in zip(run_dirs, metas, strict=True):
        best = _best_epoch_row(_read_epoch_rows(run_dir))
        model_name = meta.get("model_name", "-")
        subset = meta.get("subset_size")
        subset_cell = "full" if subset is None else str(subset)
        seed = meta.get("seed", "-")
        epochs_run = meta.get("epoch_count", "-")
        acc = _fmt(best.get("dev_accuracy")) if best else "-"
        macro = _fmt(best.get("dev_macro_f1")) if best else "-"
        weighted = _fmt(best.get("dev_weighted_f1")) if best else "-"
        lines.append(f"| {model_name} | {subset_cell} | {seed} | {epochs_run} | {acc} | {macro} | {weighted} |")
    lines.append("")

    lines += ["### Trivial baseline (majority-class reference)", ""]
    if trivial is not None:
        lines.append("| model | train subset size | seed | epochs run | dev accuracy | dev macro-F1 | dev weighted-F1 |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(
            f"| all-neutral (majority class) | - | - | - | {_fmt(trivial['accuracy'])} | "
            f"{_fmt(trivial['macro_f1'])} | {_fmt(trivial['weighted_f1'])} |"
        )
        lines.append("")
        lines.append(f"(computed on `{dev_manifest_used}`, {dev_size} examples)")
    else:
        lines.append(
            "_Trivial baseline not computed: no run's dev manifest "
            "(`run_meta.json` -> `split_names.dev`) was found on this machine._"
        )
    lines.append("")

    lines += ["## Per-class F1 (best checkpoint, dev)", ""]
    header = "| model | " + " | ".join(EMOTION_NAMES) + " |"
    sep = "|---|" + "|".join(["---"] * len(EMOTION_NAMES)) + "|"
    lines += [header, sep]
    for run_dir, meta in zip(run_dirs, metas, strict=True):
        per_class = _per_class_f1(run_dir)
        model_name = meta.get("model_name", "-")
        cells = " | ".join(_fmt(per_class.get(name)) if per_class else "-" for name in EMOTION_NAMES)
        lines.append(f"| {model_name} | {cells} |")
    if trivial is not None:
        cells = " | ".join(_fmt(trivial["per_class_f1"].get(name)) for name in EMOTION_NAMES)
        lines.append(f"| all-neutral (trivial) | {cells} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--out", default="artifacts/reports/pilot_results.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_dir = Path(args.runs_dir)

    markdown = build_markdown(runs_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"written to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
