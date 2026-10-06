"""Figures for the XAI layer (CC-8): token heatmap, audio timeline, gate distribution, per-class F1.

Every plot function writes a PNG to `path` (parents created) and closes its
figure. Display-only transforms (e.g. stripping RoBERTa's `Ġ` BPE prefix) never
touch the underlying scores.
"""

import matplotlib

matplotlib.use("Agg")  # headless (docs/TRIBAL_KNOWLEDGE.md 1.8) -- set BEFORE pyplot import, at module import time

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from emorec.data.labels import EMOTION_NAMES

AUDIO_BIN_SECONDS = 0.5
TOKENS_PER_ROW = 14


def _clean_token(token: str) -> str:
    """Display-only: drop RoBERTa's `Ġ` (space) / `Ċ` (newline) BPE markers."""
    return token.replace("Ġ", "").replace("Ċ", "") or "␣"


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bin_attribution(
    attr: np.ndarray, frame_times: np.ndarray, bin_seconds: float = AUDIO_BIN_SECONDS
) -> tuple[np.ndarray, np.ndarray]:
    """Mean attribution per `bin_seconds` bin: returns `(bin_start_seconds, bin_means)`, empty bins dropped."""
    attr = np.asarray(attr, dtype=np.float64)
    bin_idx = np.floor(np.asarray(frame_times, dtype=np.float64) / bin_seconds).astype(int)
    starts, means = [], []
    for b in np.unique(bin_idx):
        starts.append(b * bin_seconds)
        means.append(attr[bin_idx == b].mean())
    return np.asarray(starts), np.asarray(means)


def plot_token_heatmap(tokens, attr, cur_span, pred_name: str, true_name: str, path: str | Path) -> None:
    """Token strip coloured by signed attribution (diverging, centred at 0), tokens printed beneath.

    Tokens inside `cur_span` (`(start, end)`, end exclusive) are drawn with a
    heavy outline and bold labels so target-utterance attribution is
    distinguishable from dialogue-context attribution. Long sequences wrap
    onto several rows of `TOKENS_PER_ROW` tokens.
    """
    attr = np.asarray(attr, dtype=np.float64)
    n = len(tokens)
    if len(attr) != n:
        raise ValueError(f"tokens ({n}) and attr ({len(attr)}) must have equal length")
    vmax = max(float(np.abs(attr).max()), 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    n_rows = max(1, -(-n // TOKENS_PER_ROW))
    fig, ax = plt.subplots(figsize=(1.0 * min(n, TOKENS_PER_ROW) + 1.5, 1.1 * n_rows + 1.2))
    start, end = cur_span
    for i, (tok, a) in enumerate(zip(tokens, attr, strict=True)):
        row, col = divmod(i, TOKENS_PER_ROW)
        y = -row
        in_cur = start <= i < end
        ax.add_patch(
            Rectangle(
                (col, y),
                0.95,
                0.45,
                facecolor=cmap(norm(a)),
                edgecolor="black" if in_cur else "#bbbbbb",
                linewidth=2.2 if in_cur else 0.6,
            )
        )
        ax.text(
            col + 0.475,
            y - 0.08,
            _clean_token(tok),
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold" if in_cur else "normal",
        )
    ax.set_xlim(-0.1, min(n, TOKENS_PER_ROW) + 0.1)
    ax.set_ylim(-n_rows + 0.1 - 0.35, 0.6)
    ax.axis("off")
    ax.set_title(f"IG text attribution — predicted: {pred_name} | true: {true_name}\n(bold/boxed = current utterance)")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", fraction=0.04, pad=0.0, shrink=0.5)
    _save(fig, path)


def plot_audio_timeline(attr, frame_times, pred_name: str, path: str | Path) -> None:
    """Audio IG attribution vs seconds: mean within 0.5 s bins, with a zero line."""
    starts, means = bin_attribution(attr, frame_times)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#b2182b" if m >= 0 else "#2166ac" for m in means]
    ax.bar(starts, means, width=AUDIO_BIN_SECONDS * 0.9, align="edge", color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("time (s) from start of the (≤10 s) clip")
    ax.set_ylabel("mean IG attribution / 0.5 s bin")
    ax.set_title(f"IG audio attribution — predicted: {pred_name}")
    _save(fig, path)


def plot_gate_distribution(df: pd.DataFrame, path: str | Path) -> None:
    """(a) histogram of `g_audio`; (b) mean `g_audio` per TRUE class with std error bars."""
    gated = df.dropna(subset=["g_audio"])
    if gated.empty:
        raise ValueError("no gate values to plot (unimodal model?)")

    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(11, 4))
    ax_h.hist(gated["g_audio"], bins=np.linspace(0.0, 1.0, 21), color="#4393c3", edgecolor="white")
    ax_h.set_xlabel("g_audio")
    ax_h.set_ylabel("utterances")
    ax_h.set_title(f"Audio-gate distribution (n={len(gated)})")
    ax_h.set_xlim(0.0, 1.0)

    means, stds = [], []
    for cid in range(len(EMOTION_NAMES)):
        vals = gated.loc[gated["label"] == cid, "g_audio"].to_numpy(dtype=np.float64)
        means.append(vals.mean() if len(vals) else np.nan)
        stds.append(vals.std() if len(vals) else np.nan)
    x = np.arange(len(EMOTION_NAMES))
    ax_b.bar(x, np.nan_to_num(means), yerr=np.nan_to_num(stds), capsize=3, color="#92c5de", edgecolor="#2166ac")
    ax_b.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(EMOTION_NAMES, rotation=45, ha="right")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("mean g_audio (± std)")
    ax_b.set_title("Audio gate by true emotion")
    fig.tight_layout()
    _save(fig, path)


def plot_per_class_comparison(metrics_by_model: dict, path: str | Path) -> None:
    """Grouped bars of per-class F1 (7 classes on x), one bar per model.

    `metrics_by_model[name]` is either a `compute_metrics` dict (its
    `per_class_f1` is used) or a `{emotion: f1}` dict directly.
    """
    if not metrics_by_model:
        raise ValueError("metrics_by_model is empty")
    names = list(metrics_by_model)
    width = 0.8 / len(names)
    x = np.arange(len(EMOTION_NAMES))

    fig, ax = plt.subplots(figsize=(10, 4))
    for k, name in enumerate(names):
        m = metrics_by_model[name]
        per_class = m.get("per_class_f1", m)
        ax.bar(x + k * width - 0.4 + width / 2, [per_class[e] for e in EMOTION_NAMES], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_NAMES, rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-class F1 by model")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# CC-9 additions: faithfulness curves and ablation bars
# ---------------------------------------------------------------------------


def plot_faithfulness_curves(curves_by_method: dict, path: str | Path) -> None:
    """Comprehensiveness (left) and sufficiency (right) vs deleted/retained fraction.

    `curves_by_method[name]` is `{"comprehensiveness": curve, "sufficiency":
    curve, "random": {"comprehensiveness": curve, "sufficiency": curve}}`
    (`"random"` optional), where a `curve` holds `fractions` and `mean` (and
    optionally `lo`/`hi` for a CI band). The random-attribution control is drawn
    as a dashed line in the same colour as its method. Both panels plot the
    DROP in predicted-class probability: higher is better on the left, lower
    is better on the right.
    """
    if not curves_by_method:
        raise ValueError("curves_by_method is empty")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    titles = {
        "comprehensiveness": "Comprehensiveness: drop after DELETING top-k%\n(higher = more faithful)",
        "sufficiency": "Sufficiency: drop when RETAINING only top-k%\n(lower = more faithful)",
    }
    for ax, kind in zip(axes, ("comprehensiveness", "sufficiency"), strict=True):
        for idx, (name, curves) in enumerate(curves_by_method.items()):
            color = f"C{idx}"
            c = curves[kind]
            ax.plot(c["fractions"], c["mean"], marker="o", color=color, label=name)
            if "lo" in c and "hi" in c:
                ax.fill_between(c["fractions"], c["lo"], c["hi"], color=color, alpha=0.15)
            rand = curves.get("random")
            if rand is not None:
                r = rand[kind]
                ax.plot(r["fractions"], r["mean"], linestyle="--", marker="x", color=color, alpha=0.8, label=f"{name} — random")
        ax.set_xlabel("fraction of tokens / frames")
        ax.set_ylabel("mean drop in predicted-class probability")
        ax.set_title(titles[kind], fontsize=10)
        ax.axhline(0.0, color="grey", linewidth=0.6)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path)


def plot_ablation_bars(df: pd.DataFrame, path: str | Path) -> None:
    """Weighted-F1 and macro-F1 per ablation mode (grouped bars); the `"none"` control is highlighted.

    `df` is `emorec.xai.ablation.ablation_frame`'s output (columns `mode`,
    `weighted_f1`, `macro_f1`).
    """
    modes = df["mode"].tolist()
    x = np.arange(len(modes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, col, label, base in ((-width / 2, "weighted_f1", "weighted-F1", "#2166ac"), (width / 2, "macro_f1", "macro-F1", "#b2182b")):
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x + offset, vals, width, label=label, color=base, alpha=0.55)
        for bar, mode in zip(bars, modes, strict=True):
            if mode == "none":
                bar.set_alpha(1.0)
                bar.set_edgecolor("black")
                bar.set_linewidth(2.0)
        for xi, v in zip(x + offset, vals, strict=True):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(control)" if m == "none" else m for m in modes])
    ax.set_ylim(0.0, max(0.05, float(df[["weighted_f1", "macro_f1"]].to_numpy().max()) * 1.15))
    ax.set_ylabel("F1")
    ax.set_title("Inference-time modality ablation (same trained weights)")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
