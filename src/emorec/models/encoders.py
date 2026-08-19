"""Frozen text/audio encoder wrappers and the on-disk audio feature cache.

CONTRACT: both encoders are frozen feature extractors (ED1). `TextEncoder`
keeps exactly the two new `<cur>`/`</cur>` embedding rows trainable (INV3);
`AudioEncoder` has zero trainable parameters. Both are held in `eval()`
permanently (J-03) — dropout OFF is the frozen-feature-extractor convention
and it is what makes the audio cache valid; `train()` is overridden on both
so a parent module's `model.train()` can never silently re-enable their
dropout.

The frame mask for audio is ALWAYS derived through the installed
transformers model's own `_get_feature_vector_attention_mask` — never by
hand-rolling the WavLM/Wav2Vec2 conv stride arithmetic
(docs/TRIBAL_KNOWLEDGE.md §1.2, docs/CLAUDE.md dragon #2). An off-by-one
hand-rolled mask passes every shape test while leaking a pad frame into
attention; it is silent corruption, not a crash.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModel

# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Frozen RoBERTa-family text encoder with the two new `<cur>`/`</cur>` embedding rows trainable.

    `tokenizer_len` is the tokenizer's vocab size AFTER `add_special_tokens`
    (CC-3) — `resize_token_embeddings(tokenizer_len)` is called immediately
    after loading, before freezing, so the new rows exist to be made
    trainable. `new_token_ids` are the two new token ids (e.g.
    `[dataset.cur_token_id, dataset.end_cur_token_id]`) — passed in
    explicitly, NEVER re-derived from a tokenizer inside this class (the
    encoder has no business owning tokenizer logic).

    Freezing (ED1 + INV3): every parameter has `requires_grad=False` EXCEPT
    the input embedding weight, which stays `requires_grad=True` (autograd
    needs a live path to it) but has a gradient hook that ZEROES every row
    except `new_token_ids` — so an optimizer step only ever changes those
    two rows, even though the Parameter tensor itself spans the whole
    vocab. `n_trainable()` reports `len(new_token_ids) * hidden_size` (the
    values that actually receive a nonzero gradient), NOT
    `embedding.weight.numel()` (which would misleadingly count the whole,
    mostly-frozen-by-hook, embedding matrix).

    `_model`: test-only override — inject a small stub `AutoModel`-compatible
    module (e.g. a tiny `RobertaModel`) to bypass `from_pretrained` in
    offline/CPU tests. Production code never sets this.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer_len: int,
        new_token_ids: list[int],
        _model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = _model if _model is not None else AutoModel.from_pretrained(cfg.model.text_ckpt)
        self.model.resize_token_embeddings(tokenizer_len)

        self.new_token_ids = list(new_token_ids)

        for p in self.model.parameters():
            p.requires_grad = False

        embedding = self.model.get_input_embeddings()
        embedding.weight.requires_grad = True

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            mask[self.new_token_ids] = 1.0
            return grad * mask

        embedding.weight.register_hook(_hook)

        self.model.eval()

    def train(self, mode: bool = True) -> "TextEncoder":
        """Override: the wrapped encoder is ALWAYS kept in eval() (J-03) — a parent
        module's `model.train()` must never re-enable its dropout, which would
        silently break both training reproducibility and the frozen-feature
        convention the audio cache also depends on."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `text_ids`/`text_mask` (both `(B, T_t)`).

        Returns `(hidden, text_mask)` where `hidden` is `(B, T_t, hidden_size)`
        (768 for the real `roberta-base` checkpoint) and `text_mask` is
        returned UNCHANGED — text needs no resolution conversion the way
        audio does.
        """
        outputs = self.model(input_ids=text_ids, attention_mask=text_mask)
        return outputs.last_hidden_state, text_mask

    def trainable_parameters(self) -> list[nn.Parameter]:
        """The `nn.Parameter` object(s) an optimizer should be given — here, just the embedding weight."""
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        """Count of VALUES that actually receive a nonzero gradient: `len(new_token_ids) * hidden_size`.

        Deliberately NOT `sum(p.numel() for p in trainable_parameters())` --
        the embedding weight Parameter is `requires_grad=True` in its
        entirety (that's how autograd reaches it at all), but the gradient
        hook zeroes every row except the new ones, so only these values are
        ever updated by an optimizer step.
        """
        return len(self.new_token_ids) * self.model.config.hidden_size


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------


class AudioEncoder(nn.Module):
    """Frozen WavLM/Wav2Vec2-family audio encoder with a transformers-derived frame mask.

    ALL parameters are frozen (`requires_grad=False`) — there is no
    trainable path through this module in Stage 1. `sample_rate` is
    asserted to be 16000 at construction (docs/CLAUDE.md dragon #4): the
    WavLM/Wav2Vec2 family is pretrained at 16 kHz and silently accepting
    another rate would corrupt every downstream number without an error.

    `_model`: test-only override, see `TextEncoder`.
    """

    def __init__(self, cfg: DictConfig, _model: nn.Module | None = None) -> None:
        super().__init__()
        if cfg.data.sample_rate != 16000:
            raise ValueError(
                f"AudioEncoder requires cfg.data.sample_rate == 16000 (the WavLM/Wav2Vec2 "
                f"pretraining rate), got {cfg.data.sample_rate}."
            )

        self.model = _model if _model is not None else AutoModel.from_pretrained(cfg.model.audio_ckpt)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True) -> "AudioEncoder":
        """Override: see `TextEncoder.train` -- the wrapped encoder is always eval()."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, audio_values: torch.Tensor, audio_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `audio_values` `(B, T_a)` raw waveform with RAW-SAMPLE-resolution `audio_mask` `(B, T_a)` bool.

        Returns `(hidden, frame_mask)`: `hidden` is `(B, T_f, hidden_size)`
        (768 for the real `microsoft/wavlm-base-plus` checkpoint);
        `frame_mask` is `(B, T_f)` bool, True for real frames. `frame_mask`
        is derived ONLY via the model's own
        `_get_feature_vector_attention_mask(feature_vector_length,
        attention_mask)` — the raw-sample `attention_mask` (not
        pre-summed lengths) is what that method expects on the installed
        transformers version; it computes lengths internally via
        `attention_mask.cumsum(-1)[:, -1]`. Never hand-roll the conv stride
        arithmetic (docs/TRIBAL_KNOWLEDGE.md §1.2).

        Raises `ValueError` if `audio_values` is zero-width (the
        `load_audio=False` text-only path from CC-3's `MeldDataset`/
        `collate`) rather than running the encoder on garbage — a caller
        must route text-only batches around this module entirely.
        """
        if audio_values.shape[1] == 0:
            raise ValueError(
                "AudioEncoder.forward received a zero-width audio batch "
                "(audio_values.shape[1] == 0). This is the load_audio=False "
                "text-only path from MeldDataset/collate (CC-3) -- route "
                "text-only batches around the audio encoder, do not call it."
            )

        attention_mask = audio_mask.long()
        outputs = self.model(audio_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        frame_mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        return hidden, frame_mask

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Always empty: `AudioEncoder` has no trainable parameters in Stage 1."""
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# ---------------------------------------------------------------------------
# Audio feature cache
# ---------------------------------------------------------------------------


def cache_key(audio_ckpt: str) -> str:
    """First 8 hex chars of `sha1(audio_ckpt)` -- the cache-invalidating checkpoint fingerprint.

    Used as `{cfg.data.cache_dir}/{cache_key}/`: changing `cfg.model.audio_ckpt`
    changes this key, so a different encoder writes to (and reads from) a
    different directory entirely rather than silently reusing stale cached
    features (`sha1` here is a cache fingerprint, not a security control).
    """
    return hashlib.sha1(audio_ckpt.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def cache_path(cache_root: str | Path, utt_id: str) -> Path:
    """`{cache_root}/{split}/{stem}.npz` for a split-namespaced `utt_id` (e.g. `"dev/dia0_utt0"`).

    Creates the parent directory (idempotent) so callers can write to the
    returned path immediately. Raises `ValueError` if `utt_id` isn't
    split-namespaced (CC-2/CC-3 contract: every utt_id is `f"{split}/{stem}"`).
    """
    if "/" not in utt_id:
        raise ValueError(f"utt_id must be split-namespaced (e.g. 'dev/dia0_utt0'), got {utt_id!r}")
    split, stem = utt_id.split("/", 1)
    path = Path(cache_root) / split / f"{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_cached_features(path: str | Path, hidden_fp16: torch.Tensor, frame_mask: torch.Tensor) -> None:
    """Save one utterance's (already TRUE-length-sliced, unbatched) features to `path`.

    Writes `np.savez_compressed` with keys `"hidden"` (float16, `(T_f, hidden_size)`)
    and `"frame_mask"` (bool, `(T_f,)`). Unbatched: one utterance per file,
    no padding stored — callers must slice to the true frame length (via
    `frame_mask.sum()`) BEFORE calling this.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden_np = hidden_fp16.detach().cpu().numpy().astype(np.float16)
    mask_np = frame_mask.detach().cpu().numpy().astype(bool)
    np.savez_compressed(path, hidden=hidden_np, frame_mask=mask_np)


def load_cached_features(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one utterance's cached features, restoring `hidden` to float32.

    Returns `(hidden, frame_mask)`: `hidden` is `float32 (T_f, hidden_size)`
    (upcast from the on-disk float16), `frame_mask` is `bool (T_f,)`.
    """
    with np.load(Path(path)) as data:
        hidden = torch.from_numpy(data["hidden"].astype(np.float32))
        frame_mask = torch.from_numpy(data["frame_mask"].astype(bool))
    return hidden, frame_mask
