"""Inference-time modality ablation on a TRAINED model (CC-9): is audio load-bearing despite g_audio ~ 0?

CONTRACT: the same trained weights are evaluated with one modality removed or
corrupted at inference; the model is never modified and everything runs under
`no_grad`. All modes go through `MultimodalERC.forward_from_embeds` with the
substituted tensor, so `"none"` is also the control for the path itself (it
must reproduce `model(batch).logits`, which `tests/test_ablation.py` checks).

Modes
  none           unmodified inputs (the control)
  zero_audio     `audio_hidden` replaced by zeros, `frame_mask` unchanged
  zero_text      text embeddings replaced by the pad-token embedding at every
                 position, `text_mask` unchanged
  shuffle_audio  each example gets ANOTHER example's audio (its `audio_hidden`
                 AND that example's own `frame_mask`) from the same batch,
                 chosen by a seeded derangement. Separates "audio carries
                 utterance-specific signal" from "audio presence merely acts
                 as a bias term": zeroing destroys both, shuffling keeps the
                 presence and destroys only the signal.

The gate is NOT altered or rescaled anywhere here (docs/BACKLOG.md J-08).
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from emorec.data.labels import EMOTION_NAMES
from emorec.training.metrics import compute_metrics
from emorec.xai.ig import _audio_inputs, _embed_ids

MODES = ("none", "zero_audio", "zero_text", "shuffle_audio")


def derangement(n: int, rng: np.random.Generator) -> list[int]:
    """A permutation of `range(n)` with no fixed point (a random n-cycle) for `n >= 2`.

    For `n < 2` no derangement exists and the identity is returned. Built as a
    single cycle over a random ordering, so it always succeeds and is
    reproducible for a seeded `rng`.
    """
    if n < 2:
        return list(range(n))
    order = rng.permutation(n)
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.roll(order, -1)  # perm[order[i]] = order[i + 1]
    return perm.tolist()


def ablate_modality(
    model: nn.Module,
    loader: DataLoader,
    mode: str,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Evaluate `model` over `loader` under ablation `mode` (see the module docstring).

    Returns `compute_metrics(y_true, y_pred)`'s keys plus: `mode`,
    `mean_g_text` / `mean_g_audio` (mean gate over the split, NaN if the model
    has none), `n_unshuffled` (examples that could not be paired with another
    one, only for `shuffle_audio`: a batch of size 1), and the per-utterance
    `utt_ids`, `y_true`, `y_pred`, `y_prob` (lists, loader order, for paired
    tests) and `logits` (`(N, 7)` float32 array). `seed` drives only
    `shuffle_audio`, through one generator for the whole pass.
    """
    if mode not in MODES:
        raise ValueError(f"unknown ablation mode {mode!r}; choices: {list(MODES)}")
    if not hasattr(model, "forward_from_embeds"):
        raise NotImplementedError(
            f"{type(model).__name__} has no forward_from_embeds; ablation needs the fusion model (xattn)."
        )
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)
    pad_id = model.text_encoder.model.config.pad_token_id

    utt_ids: list[str] = []
    y_true: list[int] = []
    logits_all: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    n_unshuffled = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            text_embeds = _embed_ids(model, batch.text_ids)
            audio_hidden, frame_mask = _audio_inputs(model, batch)

            if mode == "zero_audio":
                audio_hidden = torch.zeros_like(audio_hidden)
            elif mode == "zero_text":
                text_embeds = _embed_ids(model, torch.full_like(batch.text_ids, pad_id))
            elif mode == "shuffle_audio":
                perm = derangement(audio_hidden.shape[0], rng)
                n_unshuffled += sum(1 for i, p in enumerate(perm) if i == p)
                perm_t = torch.as_tensor(perm, device=device)
                audio_hidden, frame_mask = audio_hidden[perm_t], frame_mask[perm_t]

            output = model.forward_from_embeds(text_embeds, audio_hidden, batch.text_mask, frame_mask)
            utt_ids.extend(m["utt_id"] for m in batch.meta)
            y_true.extend(batch.label.cpu().tolist())
            logits_all.append(output.logits.float().cpu().numpy())
            if output.gate is not None:
                gates.append(output.gate.float().cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)
    probs = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    y_pred = probs.argmax(axis=1).tolist()
    gate = np.concatenate(gates, axis=0) if gates else None

    return {
        **compute_metrics(y_true, y_pred),
        "mode": mode,
        "mean_g_text": float(gate[:, 0].mean()) if gate is not None else float("nan"),
        "mean_g_audio": float(gate[:, 1].mean()) if gate is not None else float("nan"),
        "n_unshuffled": n_unshuffled,
        "utt_ids": utt_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": probs.tolist(),
        "logits": logits,
    }


def run_ablation_results(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None, seed: int = 42
) -> dict[str, dict]:
    """`{mode: ablate_modality(...)}` for every mode in `MODES`, in order (the full per-utterance results)."""
    return {mode: ablate_modality(model, loader, mode, device=device, seed=seed) for mode in MODES}


def ablation_frame(results: dict[str, dict]) -> pd.DataFrame:
    """One row per mode: `mode, accuracy, macro_f1, weighted_f1, f1_<class>..., mean_g_audio, delta_wF1, delta_macro`.

    The deltas are relative to the `"none"` row (`results` must contain it).
    """
    base = results["none"]
    rows = []
    for mode, r in results.items():
        row = {
            "mode": mode,
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            **{f"f1_{name}": r["per_class_f1"][name] for name in EMOTION_NAMES},
            "mean_g_audio": r["mean_g_audio"],
            "delta_wF1": r["weighted_f1"] - base["weighted_f1"],
            "delta_macro": r["macro_f1"] - base["macro_f1"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation_suite(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None, seed: int = 42
) -> pd.DataFrame:
    """The ablation table: `ablation_frame(run_ablation_results(...))`."""
    return ablation_frame(run_ablation_results(model, loader, device=device, seed=seed))
