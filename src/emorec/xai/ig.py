"""Integrated Gradients for the fusion model, one modality per pass (CC-8).

CONTRACT (docs/ARCHITECTURE.md "XAI attachment points", docs/MODULES.md):

* IG-text attributes at the RoBERTa input-embedding output (baseline: the
  pad-token embedding at every position); IG-audio attributes the audio
  encoder's frame-level output (baseline: zeros). ONE modality is attributed
  per captum call, the other is held fixed — never both at once
  (docs/CLAUDE.md dragon #3).
* The attribution target is the PREDICTED class's logit by default, not the
  gold label: the point is to explain what the model did.
* Returned scores are raw and signed; nothing is normalised here.

DOCUMENTED DEVIATION (project-lead decision, no ADR): ARCHITECTURE.md places
IG-audio at the post-`proj_a` frames; this module attributes at the EXISTING
pre-`proj_a` hook (`MultimodalERC.forward_from_embeds`' `audio_hidden`, the
768-d encoder output). Both points are frame-level and frame-aligned, so the
temporal interpretation is identical; only the feature basis differs.

`captum` is the optional `[xai]` extra, so it is imported lazily — importing
`emorec.xai` never requires it.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from emorec.models.baselines import TextOnly

logger = logging.getLogger(__name__)

# Reconvergence rule (docs/ARCHITECTURE.md): if the convergence delta exceeds
# this fraction of the attribution sum, the example is re-run with RERUN_STEPS.
DELTA_REL_THRESHOLD = 0.05
RERUN_STEPS = 64

# WavLM-base-plus frame rate at 16 kHz. The conv feature extractor's total
# stride is 5*2*2*2*2*2*2 = 320 samples = 20 ms (nominal 50 fps); with its
# 400-sample receptive field a 10 s clip yields floor((160000-400)/320)+1 = 499
# frames, i.e. 49.9 frames/s. Wav2Vec2-base shares the same conv stack.
WAVLM_FRAME_RATE = 49.9

# Keys `save_attribution` requires in a record (docs/MODULES.md schema).
_REQUIRED_RECORD_KEYS = (
    "text_attr",
    "text_tokens",
    "cur_span",
    "audio_attr",
    "frame_times",
    "pred",
    "label",
    "probs",
    "gate",
    "convergence_delta",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def frames_to_times(n_frames: int, frame_rate: float = WAVLM_FRAME_RATE) -> np.ndarray:
    """Frame index -> seconds: `arange(n_frames) / frame_rate`, starting at 0.

    `frame_rate` defaults to `WAVLM_FRAME_RATE` (49.9 fps, see its comment for
    the derivation). Times are relative to the START OF THE (ED5-truncated)
    clip the encoder saw, not to the original utterance, if it was over 10 s.
    """
    return np.arange(n_frames, dtype=np.float64) / frame_rate


def cur_span(text_ids, cur_id: int, end_cur_id: int) -> tuple[int, int]:
    """Locate the `<cur> ... </cur>` span in ONE example's token ids by id search.

    Returns `(start, end)` with `end` exclusive and BOTH marker tokens excluded:
    `text_ids[start:end]` is exactly the current utterance's tokens. Searching
    by token id (never by parsing the string) is what makes this robust to BPE.
    Raises `ValueError` if either marker is absent, appears more than once, or
    the closing marker precedes the opening one (e.g. a long context truncated
    away `</cur>` at `data.max_text_len`).
    """
    ids = [int(i) for i in (text_ids.tolist() if hasattr(text_ids, "tolist") else text_ids)]
    starts = [i for i, t in enumerate(ids) if t == cur_id]
    ends = [i for i, t in enumerate(ids) if t == end_cur_id]
    if len(starts) != 1:
        raise ValueError(f"expected exactly one <cur> marker (id {cur_id}), found {len(starts)}")
    if len(ends) != 1:
        raise ValueError(f"expected exactly one </cur> marker (id {end_cur_id}), found {len(ends)}")
    if ends[0] <= starts[0]:
        raise ValueError(f"</cur> at {ends[0]} does not follow <cur> at {starts[0]}")
    return starts[0] + 1, ends[0]


def attribution_path(out_dir: str | Path, utt_id: str) -> Path:
    """`{out_dir}/{split}_{stem}.npz` for a split-namespaced `utt_id` (`dev/dia0_utt0` -> `dev_dia0_utt0.npz`)."""
    return Path(out_dir) / f"{utt_id.replace('/', '_')}.npz"


def save_attribution(path: str | Path, record: dict) -> None:
    """Write one example's attribution record as `np.savez_compressed` (docs/MODULES.md schema).

    `record` must contain: `text_attr (T_t,)`, `text_tokens (T_t,) str`,
    `cur_span (2,)`, `audio_attr (T_f,)`, `frame_times (T_f,)`, `pred int`,
    `label int`, `probs (7,)`, `gate (2,) or None` (stored as an empty array
    when None, e.g. unimodal models), `convergence_delta (2,)` (text, audio;
    NaN for a modality that was not attributed). An optional `ig_n_steps (2,)`
    is stored too if present. Raises `KeyError` naming any missing key.
    """
    missing = [k for k in _REQUIRED_RECORD_KEYS if k not in record]
    if missing:
        raise KeyError(f"attribution record missing keys: {missing}")

    gate = record["gate"]
    arrays = {
        "text_attr": np.asarray(record["text_attr"], dtype=np.float32),
        "text_tokens": np.asarray(record["text_tokens"], dtype=str),
        "cur_span": np.asarray(record["cur_span"], dtype=np.int64),
        "audio_attr": np.asarray(record["audio_attr"], dtype=np.float32),
        "frame_times": np.asarray(record["frame_times"], dtype=np.float64),
        "pred": np.asarray(int(record["pred"])),
        "label": np.asarray(int(record["label"])),
        "probs": np.asarray(record["probs"], dtype=np.float32),
        "gate": np.zeros(0, dtype=np.float32) if gate is None else np.asarray(gate, dtype=np.float32),
        "convergence_delta": np.asarray(record["convergence_delta"], dtype=np.float64),
    }
    if "ig_n_steps" in record:
        arrays["ig_n_steps"] = np.asarray(record["ig_n_steps"], dtype=np.int64)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _text_logits(
    model: nn.Module,
    text_embeds: torch.Tensor,
    audio_hidden: torch.Tensor | None,
    text_mask: torch.Tensor,
    frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Logits from text embeddings, via `MultimodalERC.forward_from_embeds` (the XAI hook).

    `TextOnly` (B1) has no `forward_from_embeds`; it is dispatched to its own
    layers directly (text encoder from `inputs_embeds` -> `proj_t` -> `pool_t`
    -> `mlp`, exactly `TextOnly.forward` minus the embedding lookup). Any
    other model has no text-embedding entry point and raises.
    """
    if hasattr(model, "forward_from_embeds"):
        return model.forward_from_embeds(text_embeds, audio_hidden, text_mask, frame_mask).logits
    if isinstance(model, TextOnly):
        hidden = model.text_encoder.model(inputs_embeds=text_embeds, attention_mask=text_mask).last_hidden_state
        z_t = model.pool_t(model.proj_t(hidden), text_mask.bool())
        return model.mlp(z_t)
    raise NotImplementedError(
        f"{type(model).__name__} has no text-embedding entry point (only MultimodalERC.forward_from_embeds "
        "and TextOnly are supported by IG)."
    )


def _embed_ids(model: nn.Module, text_ids: torch.Tensor) -> torch.Tensor:
    """Input-embedding lookup: `model.embed_text_ids` (MultimodalERC) or the text encoder's own (TextOnly)."""
    if hasattr(model, "embed_text_ids"):
        return model.embed_text_ids(text_ids)
    return model.text_encoder.model.get_input_embeddings()(text_ids)


def _audio_inputs(model: nn.Module, batch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """`(audio_hidden, frame_mask)` for `batch`: cached fields, or the frozen encoder run once under no_grad.

    Returns `(None, None)` for a text-only model, which has no audio path.
    """
    if isinstance(model, TextOnly):
        return None, None
    if hasattr(batch, "audio_hidden"):
        return batch.audio_hidden, batch.frame_mask
    with torch.no_grad():
        return model.audio_encoder(batch.audio_values, batch.audio_mask)


def _slice(arg, i: int):
    return arg[i : i + 1] if isinstance(arg, torch.Tensor) and arg.dim() > 0 else arg


def _run_ig(forward_fn, x, baseline, target, extra_args, n_steps):
    """One captum IG call. Returns `(attr, delta)` as tensors: `attr` like `x`, `delta` `(B,)`."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError('captum is required for IG: pip install -e ".[xai]"') from exc

    ig = IntegratedGradients(forward_fn)
    with torch.enable_grad():
        attr, delta = ig.attribute(
            x,
            baselines=baseline,
            target=target,
            additional_forward_args=extra_args,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
    return attr.detach(), delta.detach()


def _ig_with_rerun(forward_fn, x, baseline, target, extra_args, n_steps):
    """IG over `x` (B, T, H), summed over the feature dim; re-runs poorly-converged examples at 64 steps.

    Returns `(attr (B,T) float32 ndarray, delta (B,) ndarray, steps (B,) int ndarray)`.
    """
    attr, delta = _run_ig(forward_fn, x, baseline, target, extra_args, n_steps)
    attr = attr.sum(dim=-1)  # (B, T)
    steps = np.full(x.shape[0], n_steps, dtype=np.int64)

    if n_steps < RERUN_STEPS:
        total = attr.sum(dim=-1).abs().clamp_min(1e-12)
        bad = ((delta.abs() / total) > DELTA_REL_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        for i in bad:
            logger.info(
                "IG example %d: |delta|/|sum(attr)| = %.3f > %.2f at n_steps=%d; re-running at n_steps=%d",
                i,
                float(delta[i].abs() / total[i]),
                DELTA_REL_THRESHOLD,
                n_steps,
                RERUN_STEPS,
            )
            extra_i = tuple(_slice(a, i) for a in extra_args) if extra_args is not None else None
            a_i, d_i = _run_ig(
                forward_fn, x[i : i + 1], baseline[i : i + 1], target[i : i + 1], extra_i, RERUN_STEPS
            )
            attr[i] = a_i.sum(dim=-1)[0]
            delta[i] = d_i[0]
            steps[i] = RERUN_STEPS

    return attr.float().cpu().numpy(), delta.float().cpu().numpy(), steps


def _predict(model, text_embeds, audio_hidden, text_mask, frame_mask) -> tuple[torch.Tensor, torch.Tensor]:
    """No-grad forward -> `(probs (B,7), argmax (B,))`, through the same path IG differentiates."""
    with torch.no_grad():
        probs = torch.softmax(_text_logits(model, text_embeds, audio_hidden, text_mask, frame_mask), dim=-1)
    return probs, probs.argmax(dim=-1)


def _resolve_target(target, predicted: torch.Tensor, device) -> torch.Tensor:
    if target is None:
        return predicted
    t = torch.as_tensor(target, dtype=torch.long, device=device)
    if t.dim() == 0:
        t = t.expand(predicted.shape[0]).clone()
    return t


def _result(attr, delta, steps, target, probs) -> dict:
    return {
        "attr": attr,
        "delta": delta,
        "n_steps": steps,
        "target": target.detach().cpu().numpy(),
        "probs": probs.detach().float().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attribute_text(
    model: nn.Module,
    batch,
    target=None,
    n_steps: int = 32,
    device=None,
    pad_token_id: int | None = None,
) -> dict:
    """Per-token IG attributions for `batch`'s text, audio held fixed.

    Attributes through `model.embed_text_ids(batch.text_ids)` with a baseline
    of the pad-token embedding at every position, via a closure over
    `forward_from_embeds` that holds `audio_hidden` and both masks FIXED.
    `target` defaults to the model's PREDICTED class (argmax of a no-grad
    forward); pass class ids (scalar or `(B,)`) to override. `pad_token_id`
    defaults to the text encoder's `config.pad_token_id` (equal to the
    tokenizer's for roberta-base).

    Returns `{"attr": (B, T_t) float32, "delta": (B,), "n_steps": (B,) int,
    "target": (B,), "probs": (B, 7)}` as numpy arrays. `attr` is the raw
    signed sum over the embedding dimension (pad positions get ~0). `delta` is
    captum's convergence delta; examples with `|delta| / |sum(attr)| > 0.05`
    are re-run at 64 steps (`n_steps` records the final steps used per example).
    """
    model.eval()
    device = device or next(model.parameters()).device
    batch = batch.to(device)

    if pad_token_id is None:
        pad_token_id = model.text_encoder.model.config.pad_token_id
    if pad_token_id is None:
        raise ValueError("pad_token_id not given and text encoder config has none")

    audio_hidden, frame_mask = _audio_inputs(model, batch)
    text_mask = batch.text_mask

    with torch.no_grad():
        text_embeds = _embed_ids(model, batch.text_ids).detach()
        baseline = _embed_ids(model, torch.full_like(batch.text_ids, pad_token_id)).detach()

    def logits_fn(text_embeds, audio_hidden, text_mask, frame_mask):
        # The embeds must require grad HERE: the frozen embedding parameter is
        # unrelated, and omitting this yields all-zero attributions silently
        # (docs/TRIBAL_KNOWLEDGE.md 1.6).
        text_embeds = text_embeds.requires_grad_(True)
        return _text_logits(model, text_embeds, audio_hidden, text_mask, frame_mask)

    probs, predicted = _predict(model, text_embeds, audio_hidden, text_mask, frame_mask)
    tgt = _resolve_target(target, predicted, device)

    attr, delta, steps = _ig_with_rerun(
        logits_fn, text_embeds, baseline, tgt, (audio_hidden, text_mask, frame_mask), n_steps
    )
    return _result(attr, delta, steps, tgt, probs)


def attribute_audio(
    model: nn.Module,
    batch,
    target=None,
    n_steps: int = 32,
    device=None,
) -> dict:
    """Per-frame IG attributions for `batch`'s audio, text held fixed.

    Attributes `audio_hidden` (the encoder's frame-level output, PRE-`proj_a`;
    see the module docstring's documented deviation) against a zeros baseline,
    with the text embeddings and both masks held FIXED. `audio_hidden` comes
    from the batch if cached, else from the frozen audio encoder. Same return
    keys, `target` default and re-run rule as `attribute_text`, with `attr`
    `(B, T_f)` summed over the hidden dimension (padded frames get 0).
    """
    model.eval()
    device = device or next(model.parameters()).device
    batch = batch.to(device)

    if not hasattr(model, "forward_from_embeds"):
        raise NotImplementedError(
            f"{type(model).__name__} has no audio-hidden entry point; IG-audio needs MultimodalERC.forward_from_embeds."
        )

    audio_hidden, frame_mask = _audio_inputs(model, batch)
    audio_hidden = audio_hidden.detach()
    text_mask = batch.text_mask

    with torch.no_grad():
        text_embeds = _embed_ids(model, batch.text_ids).detach()
    baseline = torch.zeros_like(audio_hidden)

    def logits_fn(audio_hidden, text_embeds, text_mask, frame_mask):
        audio_hidden = audio_hidden.requires_grad_(True)  # see attribute_text: required for non-zero attributions
        return _text_logits(model, text_embeds, audio_hidden, text_mask, frame_mask)

    probs, predicted = _predict(model, text_embeds, audio_hidden, text_mask, frame_mask)
    tgt = _resolve_target(target, predicted, device)

    attr, delta, steps = _ig_with_rerun(
        logits_fn, audio_hidden, baseline, tgt, (text_embeds, text_mask, frame_mask), n_steps
    )
    return _result(attr, delta, steps, tgt, probs)
