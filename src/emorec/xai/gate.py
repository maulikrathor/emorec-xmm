"""Modality-gate collection and dominance summary (CC-8; intrinsic signal #3 in docs/EVALUATION.md section 6).

CONTRACT: the gate `g = (g_text, g_audio)` is reported EXACTLY as the model
produced it. It is never rescaled, re-centred or "corrected" — if it is
saturated toward one modality across a split, that is a reportable result
(docs/BACKLOG.md J-08 pre-authorises reporting it), not a bug to fix here.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from emorec.data.labels import ID_TO_EMOTION, NUM_CLASSES

GATE_COLUMNS = ["utt_id", "label", "pred", "g_text", "g_audio", "correct"]

# A gate this close to 0 or 1 is called "saturated" in the summary.
SATURATION_EPS = 0.05


def collect_gates(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> pd.DataFrame:
    """One row per utterance, in loader order: `utt_id, label, pred, g_text, g_audio, correct`.

    Runs `model` in `eval()` under `no_grad`. If the model has no gate
    (`Output.gate is None`, e.g. the unimodal baselines) the gate columns are
    NaN so callers never crash; `label`/`pred`/`correct` are still filled.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            preds = output.logits.argmax(dim=-1).cpu().tolist()
            labels = batch.label.cpu().tolist()
            if output.gate is None:
                gates = [(float("nan"), float("nan"))] * len(preds)
            else:
                gates = [(float(g[0]), float(g[1])) for g in output.gate.float().cpu()]
            for meta, label, pred, (g_text, g_audio) in zip(batch.meta, labels, preds, gates, strict=True):
                rows.append(
                    {
                        "utt_id": meta["utt_id"],
                        "label": int(label),
                        "pred": int(pred),
                        "g_text": g_text,
                        "g_audio": g_audio,
                        "correct": bool(pred == label),
                    }
                )
    return pd.DataFrame(rows, columns=GATE_COLUMNS)


def gate_summary(df: pd.DataFrame) -> dict:
    """Dominance summary of `collect_gates`' frame. Standard deviations are population (ddof=0).

    Keys:
      `n`, `has_gate`;
      `overall` — `{mean, median, std}` of `g_audio`;
      `quantiles` — `g_audio` at 5/25/50/75/95%;
      `frac_audio_dominant` — fraction with `g_audio > 0.5`;
      `frac_saturated_audio` / `frac_saturated_text` — fraction with
        `g_audio > 0.95` / `g_audio < 0.05`;
      `per_class` — for every TRUE class name, `{mean, std, n}` of `g_audio`
        (`mean`/`std` NaN when `n == 0`);
      `by_correctness` — `{"correct": {mean, n}, "incorrect": {mean, n}}`;
      `top_audio_dominant` / `top_text_dominant` — the 10 `utt_id`s with the
        highest / lowest `g_audio`.
    If every gate is NaN (unimodal model) only `n` and `has_gate=False` are returned.
    """
    n = len(df)
    gated = df.dropna(subset=["g_audio"])
    if gated.empty:
        return {"n": n, "has_gate": False}

    g = gated["g_audio"].to_numpy(dtype=np.float64)

    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        return (float(values.mean()), float(values.std())) if len(values) else (float("nan"), float("nan"))

    per_class: dict[str, dict] = {}
    for cid in range(NUM_CLASSES):
        vals = gated.loc[gated["label"] == cid, "g_audio"].to_numpy(dtype=np.float64)
        mean, std = _mean_std(vals)
        per_class[ID_TO_EMOTION[cid]] = {"mean": mean, "std": std, "n": len(vals)}

    by_correctness: dict[str, dict] = {}
    for name, flag in (("correct", True), ("incorrect", False)):
        vals = gated.loc[gated["correct"] == flag, "g_audio"].to_numpy(dtype=np.float64)
        by_correctness[name] = {"mean": _mean_std(vals)[0], "n": len(vals)}

    q5, q25, q50, q75, q95 = (float(q) for q in np.percentile(g, [5, 25, 50, 75, 95]))
    ranked = gated.sort_values(["g_audio", "utt_id"], ascending=[False, True])

    return {
        "n": n,
        "has_gate": True,
        "overall": {"mean": float(g.mean()), "median": float(np.median(g)), "std": float(g.std())},
        "quantiles": {"p05": q5, "p25": q25, "p50": q50, "p75": q75, "p95": q95},
        "frac_audio_dominant": float((g > 0.5).mean()),
        "frac_saturated_audio": float((g > 1.0 - SATURATION_EPS).mean()),
        "frac_saturated_text": float((g < SATURATION_EPS).mean()),
        "per_class": per_class,
        "by_correctness": by_correctness,
        "top_audio_dominant": ranked["utt_id"].head(10).tolist(),
        "top_text_dominant": ranked["utt_id"].tail(10).iloc[::-1].tolist(),
    }
