"""Faithfulness of IG attributions (CC-9; docs/EVALUATION.md section 5): deletion curves, random control, sanity check.

Curves
  comprehensiveness  DELETE the top-k% attributed positions; the value is the
                     mean DROP in the predicted class's probability
                     (p_full - p_deleted). Higher = more faithful.
  sufficiency        RETAIN only the top-k% positions (everything else deleted);
                     the value is the DROP (p_full - p_retained), i.e. what is
                     LOST by keeping only the top-k%. Smaller = more faithful;
                     the probability retained is `p_full - value`.

Positions are ranked by the SIGNED attribution, descending (the evidence FOR
the predicted class); only real (unmasked) tokens/frames are eligible and
`ceil(k * n_real)` (at least 1) are selected. The predicted class is the
argmax of the unmodified forward, the same target IG explains.

DELETION OPERATORS match the IG baselines BY CONSTRUCTION: text positions are
replaced by the pad-token embedding, audio frames by a zero post-encoder frame
(`audio_hidden`, pre-`proj_a`, the point CC-8 attributes at); masks are left
unchanged so a deleted position is still attended, exactly like the baseline.
This consistency mildly favours IG in any cross-method comparison
(docs/KNOWN_LIMITATIONS.md 14) and must be stated alongside any result.

A curve is a dict `{"fractions": (F,), "per_example": (B, F), "mean": (F,),
"kind", "modality"}`; `faithfulness_auc` integrates `mean` over `fractions`.
"""

import copy
import math

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from emorec.xai.ig import _audio_inputs, _embed_ids, _text_logits, attribute_audio, attribute_text

DEFAULT_FRACTIONS = (0.05, 0.1, 0.2, 0.5)
MODALITIES = ("text", "audio")
# A method is reported BROKEN for this model if, after randomising the head, the
# mean Spearman correlation with the original attributions stays at or above this.
BROKEN_SPEARMAN_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _inputs(model: nn.Module, batch) -> dict:
    """Everything a deletion forward needs, on the model's device, under no_grad."""
    model.eval()
    device = next(model.parameters()).device
    batch = batch.to(device)
    pad_id = model.text_encoder.model.config.pad_token_id
    with torch.no_grad():
        text_embeds = _embed_ids(model, batch.text_ids).detach()
        pad_embeds = _embed_ids(model, torch.full_like(batch.text_ids, pad_id)).detach()
        audio_hidden, frame_mask = _audio_inputs(model, batch)
    return {
        "text_embeds": text_embeds,
        "pad_embeds": pad_embeds,
        "audio_hidden": None if audio_hidden is None else audio_hidden.detach(),
        "text_mask": batch.text_mask,
        "frame_mask": frame_mask,
    }


def _probs(model: nn.Module, x: dict, text_embeds=None, audio_hidden=None) -> torch.Tensor:
    with torch.no_grad():
        logits = _text_logits(
            model,
            x["text_embeds"] if text_embeds is None else text_embeds,
            x["audio_hidden"] if audio_hidden is None else audio_hidden,
            x["text_mask"],
            x["frame_mask"],
        )
    return torch.softmax(logits, dim=-1)


def _valid_mask(x: dict, modality: str) -> torch.Tensor:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    if modality == "audio" and x["frame_mask"] is None:
        raise ValueError("this model has no audio path (text-only), so audio faithfulness is undefined")
    return x["text_mask"].bool() if modality == "text" else x["frame_mask"].bool()


def _top_fraction_mask(attr: torch.Tensor, valid: torch.Tensor, fraction: float) -> torch.Tensor:
    """`(B, T)` bool: the `ceil(fraction * n_real)` highest-attributed real positions per example."""
    masked = attr.masked_fill(~valid, float("-inf"))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(attr.shape[1], device=attr.device).expand_as(order))
    n_real = valid.sum(dim=1).double()
    k = torch.ceil((fraction * n_real).round(decimals=9)).clamp(min=1).long()
    return (ranks < k[:, None]) & valid


def _curve(model: nn.Module, batch, attr, modality: str, fractions, kind: str, target=None) -> dict:
    x = _inputs(model, batch)
    valid = _valid_mask(x, modality)
    attr_t = torch.as_tensor(np.asarray(attr), dtype=torch.float32, device=valid.device)
    if attr_t.shape != valid.shape:
        raise ValueError(f"attr shape {tuple(attr_t.shape)} does not match the {modality} input {tuple(valid.shape)}")

    p_full = _probs(model, x)
    cls = p_full.argmax(dim=-1) if target is None else torch.as_tensor(target, device=valid.device).long()
    base = p_full.gather(1, cls[:, None]).squeeze(1)

    drops = []
    for fraction in fractions:
        top = _top_fraction_mask(attr_t, valid, fraction)
        delete = top if kind == "comprehensiveness" else (valid & ~top)
        if modality == "text":
            p = _probs(model, x, text_embeds=torch.where(delete[..., None], x["pad_embeds"], x["text_embeds"]))
        else:
            p = _probs(model, x, audio_hidden=x["audio_hidden"].masked_fill(delete[..., None], 0.0))
        drops.append((base - p.gather(1, cls[:, None]).squeeze(1)).cpu().numpy())

    per_example = np.stack(drops, axis=1)  # (B, F)
    return {
        "fractions": np.asarray(fractions, dtype=np.float64),
        "per_example": per_example,
        "mean": per_example.mean(axis=0),
        "kind": kind,
        "modality": modality,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def comprehensiveness(model, batch, attr, modality: str, fractions=DEFAULT_FRACTIONS) -> dict:
    """Curve of the mean drop in predicted-class probability when the top-k% attributed positions are DELETED.

    `modality` is `"text"` (`attr` `(B, T_t)`) or `"audio"` (`attr` `(B, T_f)`).
    Higher = more faithful. Deletion operators match the IG baselines (module
    docstring), which mildly favours IG (docs/KNOWN_LIMITATIONS.md 14).
    """
    return _curve(model, batch, attr, modality, fractions, "comprehensiveness")


def sufficiency(model, batch, attr, modality: str, fractions=DEFAULT_FRACTIONS) -> dict:
    """Curve of the drop in predicted-class probability when ONLY the top-k% attributed positions are RETAINED.

    The value is the DROP relative to the full input, so smaller = better (the
    probability retained is `p_full - value`). Same operators and caveat as
    `comprehensiveness`.
    """
    return _curve(model, batch, attr, modality, fractions, "sufficiency")


def random_baseline_curves(
    model, batch, modality: str, fractions=DEFAULT_FRACTIONS, seed: int = 42, n_draws: int = 5
) -> dict:
    """`{"comprehensiveness": curve, "sufficiency": curve}` for RANDOM attributions of the real shape (the control).

    Each of `n_draws` draws is a seeded uniform-random attribution
    (one `torch.Generator` seeded with `seed`); the returned curves average
    their per-example values. Without this control a faithfulness curve means
    nothing: it is what shows IG beating chance.
    """
    x = _inputs(model, batch)
    shape = _valid_mask(x, modality).shape
    gen = torch.Generator().manual_seed(seed)
    draws = {"comprehensiveness": [], "sufficiency": []}
    for _ in range(n_draws):
        attr = torch.rand(shape, generator=gen)
        draws["comprehensiveness"].append(_curve(model, batch, attr, modality, fractions, "comprehensiveness"))
        draws["sufficiency"].append(_curve(model, batch, attr, modality, fractions, "sufficiency"))

    out = {}
    for kind, curves in draws.items():
        per_example = np.mean([c["per_example"] for c in curves], axis=0)
        out[kind] = {
            "fractions": curves[0]["fractions"],
            "per_example": per_example,
            "mean": per_example.mean(axis=0),
            "kind": kind,
            "modality": modality,
        }
    return out


def combine_curves(curves: list[dict]) -> dict:
    """Stack the examples of several curves (same fractions/kind/modality) into one."""
    per_example = np.concatenate([c["per_example"] for c in curves], axis=0)
    return {
        "fractions": curves[0]["fractions"],
        "per_example": per_example,
        "mean": per_example.mean(axis=0),
        "kind": curves[0]["kind"],
        "modality": curves[0]["modality"],
    }


def _trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return ((x[1:] - x[:-1]) * (y[..., 1:] + y[..., :-1]) / 2.0).sum(axis=-1)


def faithfulness_auc(curve: dict) -> float:
    """Trapezoid area of `curve["mean"]` over `curve["fractions"]` (unnormalised; spans 0.05-0.5 by default)."""
    return float(_trapezoid(curve["mean"], curve["fractions"]))


def per_example_auc(curve: dict) -> np.ndarray:
    """`(B,)` trapezoid area of each example's own curve (for paired comparisons and CIs)."""
    return _trapezoid(curve["per_example"], curve["fractions"])


def mean_ci(values, n: int = 1000, seed: int = 42) -> dict:
    """`{mean, lo, hi}`: percentile bootstrap 95% CI of the mean over examples (`n` resamples)."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), size=len(v))].mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def randomized_head_copy(model: nn.Module, seed: int) -> nn.Module:
    """A copy of `model` whose classifier head (`model.mlp`) has freshly re-initialised parameters.

    The head is deep-copied and randomised (seeded, on CPU, without touching
    the global RNG); the rest of the model is shared by reference through a
    shallow clone with its own module table, so the encoders are not duplicated
    (they are ~450 MB) and the input `model` is never modified.
    """
    head = copy.deepcopy(model.mlp).cpu()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    head = head.to(next(model.mlp.parameters()).device)

    clone = copy.copy(model)
    clone._modules = dict(model._modules)  # own table: swapping the head must not touch `model`
    clone._modules["mlp"] = head
    return clone


def sanity_check_randomization(
    model: nn.Module, batch, attr: dict, seed: int = 42, n_steps: int = 32, target=None
) -> dict:
    """Adebayo-style parameter-randomisation check on the classifier head (`model.mlp`).

    `attr` maps `"text"` and/or `"audio"` to the ORIGINAL IG attributions for
    `batch` (`(B, T)` arrays). A randomised-head copy (`randomized_head_copy`;
    the real model is never mutated) gets its IG recomputed for the SAME target
    (the original model's predicted class unless `target` is given), and the
    Spearman correlation between original and randomised attributions is taken
    per example over real positions (`spearman`, signed; `spearman_abs`, on
    magnitudes). A method whose attributions do NOT degrade (mean signed
    Spearman >= `BROKEN_SPEARMAN_THRESHOLD`) is REPORTED AS BROKEN for this
    model (`broken[modality] = True`), not repaired.

    Returns `{"modalities": {mod: {"spearman": [...], "spearman_abs": [...],
    "mean_spearman": float, "mean_spearman_abs": float}}, "broken": {mod: bool},
    "threshold", "head": "mlp", "seed"}`. NaN correlations (constant
    attributions) are excluded from the means.
    """
    x = _inputs(model, batch)
    if target is None:
        target = _probs(model, x).argmax(dim=-1)
    rand_model = randomized_head_copy(model, seed)

    result: dict = {"modalities": {}, "broken": {}, "threshold": BROKEN_SPEARMAN_THRESHOLD, "head": "mlp", "seed": seed}
    for modality, original in attr.items():
        valid = _valid_mask(x, modality).cpu().numpy()
        fn = attribute_text if modality == "text" else attribute_audio
        randomised = fn(rand_model, batch, target=target, n_steps=n_steps)["attr"]
        original = np.asarray(original)

        signed, absolute = [], []
        for i in range(original.shape[0]):
            a, b = original[i][valid[i]], randomised[i][valid[i]]
            signed.append(float(spearmanr(a, b).statistic) if len(a) > 1 else float("nan"))
            absolute.append(float(spearmanr(np.abs(a), np.abs(b)).statistic) if len(a) > 1 else float("nan"))
        mean_signed = float(np.nanmean(signed)) if not np.all(np.isnan(signed)) else float("nan")
        mean_abs = float(np.nanmean(absolute)) if not np.all(np.isnan(absolute)) else float("nan")
        result["modalities"][modality] = {
            "spearman": signed,
            "spearman_abs": absolute,
            "mean_spearman": mean_signed,
            "mean_spearman_abs": mean_abs,
        }
        result["broken"][modality] = bool(not math.isnan(mean_signed) and mean_signed >= BROKEN_SPEARMAN_THRESHOLD)
    return result
