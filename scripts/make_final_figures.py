"""Regenerate the presentation figures from the COMMITTED result artifacts only (CC-10).

Usage:
    python scripts/make_final_figures.py --results-dir results [--out-dir results/reports/figures]

No model, no data, no GPU and no new metric computation: every number plotted
is read from a file under `--results-dir` (classification reports, statistics.md,
ablation.csv/.md, gates.csv, gate_analysis.md, final_results.md). If a required
input is missing the script fails loudly, naming the file. Writes four PNGs:

  final_per_class_f1.png       per-class F1, B1/B2/M, dev and test  (classification_report*.txt)
  final_dev_vs_test_wf1.png    weighted-F1 with bootstrap 95% CI, dev vs test  (statistics.md)
  final_ablation.png           modality ablation, control highlighted  (ablation.csv / ablation.md)
  final_gate_distribution.png  g_audio histogram (log axis) + per-class means  (gates.csv / gate_analysis.md)

The `final_` prefix keeps these from overwriting the figures the experiment
scripts already wrote (e.g. `ablation.png`, `gate_distribution.png`).
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emorec.data.labels import EMOTION_NAMES

MODELS = {"text_only": "B1 text-only", "audio_only": "B2 audio-only", "xattn": "M gated fusion"}
COLORS = {"text_only": "#4393c3", "audio_only": "#f4a582", "xattn": "#b2182b"}
_CI = re.compile(r"(\d\.\d+) \[(\d\.\d+), (\d\.\d+)\]")

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})


def _require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing required input: {path}")
    return path


def _run_dir(results: Path, model: str) -> Path:
    matches = sorted((results / "runs").glob(f"{model}-seed*"))
    if not matches:
        raise SystemExit(f"missing required input: no run directory {results / 'runs' / (model + '-seed*')}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous: several run directories for {model}: {[m.name for m in matches]}")
    return matches[0]


def _table_rows(path: Path, heading: str) -> list[list[str]]:
    """Cells of every row of the first markdown table after the line starting with `heading`."""
    lines = _require(path).read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    except StopIteration:
        raise SystemExit(f"{path}: heading {heading!r} not found") from None
    rows, in_table = [], False
    for ln in lines[start + 1 :]:
        if ln.startswith("|"):
            in_table = True
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):  # skip the |---| separator
                rows.append(cells)
        elif in_table:
            break
    return rows[1:]  # drop the header row


def _report_f1(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Per-class F1 and support from a sklearn classification_report text file (2 decimals, as written)."""
    f1, support = {}, {}
    for ln in _require(path).read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) == 5 and parts[0] in EMOTION_NAMES:
            f1[parts[0]], support[parts[0]] = float(parts[3]), int(parts[4])
    if set(f1) != set(EMOTION_NAMES):
        raise SystemExit(f"{path}: expected all 7 classes, found {sorted(f1)}")
    return f1, support


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- figures


def fig_per_class(results: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    width = 0.27
    x = np.arange(len(EMOTION_NAMES))
    for ax, (split, fname) in zip(axes, (("dev", "classification_report.txt"), ("test", "classification_report_test.txt")), strict=True):
        support = {}
        for k, (model, label) in enumerate(MODELS.items()):
            f1, support = _report_f1(_run_dir(results, model) / fname)
            ax.bar(x + (k - 1) * width, [f1[e] for e in EMOTION_NAMES], width, label=label, color=COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e}\n(n={support[e]})" for e in EMOTION_NAMES])
        ax.set_title(f"Per-class F1 — {split} (best checkpoint, seed 42)")
        ax.set_ylim(0, 0.9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1")
    axes[0].legend(frameon=False)
    fig.text(0.5, -0.02, "Source: runs/*/classification_report*.txt (two decimals).", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out / "final_per_class_f1.png")


def fig_dev_vs_test(results: Path, out: Path) -> None:
    stats = results / "reports" / "statistics.md"
    ci = {}
    for cells in _table_rows(stats, "## Bootstrap 95% CI"):
        m = _CI.search(cells[2])
        if m is None:
            raise SystemExit(f"{stats}: cannot parse weighted-F1 cell {cells[2]!r}")
        ci[(cells[0], cells[1])] = tuple(float(v) for v in m.groups())
    p = {}
    for cells in _table_rows(stats, "## McNemar"):
        p[(cells[0], cells[1])] = cells[4]
    trivial = _table_rows(results / "reports" / "final_results.md", "### Trivial baseline")[0]
    trivial_wf1 = float(trivial[-1])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.26
    for k, (model, label) in enumerate(MODELS.items()):
        for g, split in enumerate(("dev", "test")):
            point, lo, hi = ci[(model, split)]
            ax.bar(g + (k - 1) * width, point, width, color=COLORS[model], label=label if g == 0 else None,
                   yerr=[[point - lo], [hi - point]], capsize=4, ecolor="black")  # fmt: skip
            ax.text(g + (k - 1) * width, hi + 0.01, f"{point:.4f}", ha="center", fontsize=8)
    ax.hlines(trivial_wf1, -0.45, 0.45, colors="grey", linestyles="--", linewidth=1)
    ax.text(0.45, trivial_wf1 + 0.008, f"all-neutral, dev ({trivial_wf1:.4f})", ha="right", fontsize=8, color="dimgrey",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})  # fmt: skip
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["dev (n=1108)", "test (n=2610)"])
    ax.set_ylabel("weighted-F1 (bootstrap 95% CI)")
    ax.set_ylim(0, 0.75)
    ax.set_title("Weighted-F1, dev vs test (full train, seed 42)")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5, -0.03,
        f"Paired McNemar, M vs B1: dev p={p[('xattn vs text_only', 'dev')]}, test p={p[('xattn vs text_only', 'test')]}. "
        "Test CIs of M and B1 overlap; the paired test is the same-data instrument.",
        ha="center", fontsize=8,
    )  # fmt: skip
    fig.tight_layout()
    _save(fig, out / "final_dev_vs_test_wf1.png")


def fig_ablation(results: Path, out: Path) -> None:
    df = pd.read_csv(_require(results / "reports" / "ablation.csv"))
    md = results / "reports" / "ablation.md"
    ci = {}
    for cells in _table_rows(md, "## Results"):
        m = _CI.search(cells[3])
        if m is None:
            raise SystemExit(f"{md}: cannot parse weighted-F1 CI cell {cells[3]!r}")
        ci[cells[0]] = tuple(float(v) for v in m.groups())
    pval = {cells[0]: cells[4] for cells in _table_rows(md, "## Paired McNemar")}

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, row in df.iterrows():
        mode = row["mode"]
        point, lo, hi = ci[mode]
        control = mode == "none"
        ax.bar(i, point, 0.6, color="#b2182b" if control else "#92c5de", edgecolor="black" if control else "none",
               linewidth=2.5 if control else 0, yerr=[[point - lo], [hi - point]], capsize=5, ecolor="black")  # fmt: skip
        note = "control" if control else f"Δ {row['delta_wF1']:+.4f}\nMcNemar p={pval[mode]}"
        ax.text(i, hi + 0.012, f"{point:.4f}\n{note}", ha="center", fontsize=9)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["mode"])
    ax.set_ylabel("weighted-F1 (bootstrap 95% CI), dev")
    ax.set_ylim(0, 0.8)
    g = df.loc[df["mode"] == "none", "mean_g_audio"].iloc[0]
    ax.set_title(f"Inference-time modality ablation, same weights (control mean g_audio = {g:.4f})")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.03, "Zeroed inputs are out-of-distribution; McNemar tests per-utterance correctness vs the control.", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out / "final_ablation.png")


def fig_gate(results: Path, out: Path) -> None:
    gates = pd.read_csv(_require(results / "reports" / "gates.csv"))
    g = gates["g_audio"].to_numpy(dtype=float)
    floor = 1e-6
    n_floor = int((g < floor).sum())
    per_class = {c[0]: (int(c[1]), float(c[2])) for c in _table_rows(results / "reports" / "gate_analysis.md", "## g_audio per TRUE class")}

    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax_h.hist(np.log10(np.clip(g, floor, None)), bins=40, color="#4393c3", edgecolor="white")
    for val, lab in ((0.05, "0.05: 'saturated toward text' below this"), (0.5, "0.5: audio-dominant above this")):
        ax_h.axvline(np.log10(val), color="black", linestyle="--", linewidth=1)
        ax_h.text(np.log10(val), ax_h.get_ylim()[1] * 0.92, " " + lab, rotation=90, va="top", ha="right", fontsize=8)
    ax_h.set_xlabel("log10(g_audio)")
    ax_h.set_ylabel("dev utterances")
    ax_h.set_xlim(-6.2, 0.2)
    ax_h.set_title(f"Audio gate on dev (n={len(g)}; mean {g.mean():.4f}, median {np.median(g):.4f})")
    if n_floor:
        ax_h.text(0.02, 0.97, f"{n_floor} value(s) below 1e-6 plotted at 1e-6", transform=ax_h.transAxes, fontsize=8, va="top")

    xs = np.arange(len(EMOTION_NAMES))
    ax_b.bar(xs, [per_class[e][1] * 1e3 for e in EMOTION_NAMES], color="#92c5de", edgecolor="#2166ac")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([f"{e}\n(n={per_class[e][0]})" for e in EMOTION_NAMES], fontsize=9)
    ax_b.set_ylabel("mean g_audio  (×10⁻³)")
    ax_b.set_title("Mean audio gate by true class\n(all classes ≪ 0.5, i.e. text-saturated)")
    ax_b.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.03, "Source: gates.csv, gate_analysis.md. The gate is shown exactly as produced; nothing is rescaled.", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out / "final_gate_distribution.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None, help="Default: <results-dir>/reports/figures")
    args = parser.parse_args(argv)

    results = Path(args.results_dir)
    out = Path(args.out_dir) if args.out_dir else results / "reports" / "figures"
    for fn in (fig_per_class, fig_dev_vs_test, fig_ablation, fig_gate):
        fn(results, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
