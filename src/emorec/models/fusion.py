"""The gated cross-attention fusion model, exactly per docs/ARCHITECTURE.md.

CONTRACT: `forward_from_embeds` is built NOW (not deferred to the XAI
ticket) — it is the hook Integrated Gradients attributes through, bypassing
the text token-embedding LOOKUP so gradients can flow into arbitrary input
embeddings. Retrofitting it later would mean re-deriving the exact fusion
pipeline twice; instead `forward` and `forward_from_embeds` both delegate to
one private `_fuse` so they can never drift apart.

Mask invariance (INV1, test-enforced, the critical one): the whole
cross-attention + pooling pipeline must be architecturally invariant to how
much padding surrounds a real example, in BOTH modalities. This holds only
if every masked operation is masked correctly — `nn.MultiheadAttention`'s
`key_padding_mask` polarity is True=IGNORE, the OPPOSITE of this repo's
True=real convention everywhere else, and every mask is inverted explicitly
at the point of use, never assumed.
"""

from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from emorec.models.encoders import AudioEncoder, TextEncoder


@dataclass
class Output:
    """The output contract EVERY model in this repo returns (baselines included)."""

    logits: torch.Tensor  # (B, 7)
    gate: torch.Tensor | None  # (B, 2) = (g_t, g_a); None for unimodal baselines
    xattn: dict | None  # cross-attention maps for XAI; None unless requested


class AttnPool(nn.Module):
    """Masked attention pooling: one learned query attends over a masked sequence.

    `forward(x, mask)`: `x` is `(B, T, D)`, `mask` is `(B, T)` bool with
    True=real (this repo's mask convention). Padded positions get score
    `-inf` before the softmax, so they receive exactly zero pooling weight
    regardless of how many of them there are.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * (d_model**-0.5))
        self.scale = d_model**-0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("btd,d->bt", x, self.query) * self.scale  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        return torch.einsum("bt,btd->bd", weights, x)  # (B, D)


class CrossModalBlock(nn.Module):
    """One bidirectional cross-modal block: text attends to audio AND audio attends to text.

    Pre-LayerNorm, residual, masked in BOTH directions.
    `nn.MultiheadAttention`'s `key_padding_mask` convention is True=IGNORE
    — the OPPOSITE of this repo's True=real mask convention — so every
    mask is inverted (`~mask`) immediately before being passed in, never
    silently assumed to already be in the right polarity.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln_t_attn = nn.LayerNorm(d_model)
        self.ln_a_attn = nn.LayerNorm(d_model)
        self.text_attends_audio = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.audio_attends_text = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ln_t_ffn = nn.LayerNorm(d_model)
        self.ln_a_ffn = nn.LayerNorm(d_model)
        self.ffn_t = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.ffn_a = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_norm = self.ln_t_attn(t)
        a_norm = self.ln_a_attn(a)

        # PyTorch convention: key_padding_mask True means IGNORE that key.
        # Our masks are True=real, so invert explicitly at the call site.
        text_key_padding_mask = ~text_mask  # True where text is PAD
        audio_key_padding_mask = ~audio_mask  # True where audio is PAD

        t_attn_out, _ = self.text_attends_audio(
            query=t_norm, key=a_norm, value=a_norm, key_padding_mask=audio_key_padding_mask
        )
        a_attn_out, _ = self.audio_attends_text(
            query=a_norm, key=t_norm, value=t_norm, key_padding_mask=text_key_padding_mask
        )

        t = t + t_attn_out
        a = a + a_attn_out

        t = t + self.ffn_t(self.ln_t_ffn(t))
        a = a + self.ffn_a(self.ln_a_ffn(a))

        return t, a


class MultimodalERC(nn.Module):
    """The gated cross-attention fusion model (docs/ARCHITECTURE.md "System at a glance").

    `proj_t`/`proj_a` project each frozen encoder's hidden states to
    `cfg.model.d_model`; `cfg.model.n_fusion_blocks` `CrossModalBlock`s mix
    the two streams bidirectionally; a masked `AttnPool` per modality
    produces `z_t`/`z_a`; a gate `g = softmax(Linear([z_t;z_a]))` (bias
    zero-initialized, so it starts balanced ~0.5/0.5) weights
    `fused = concat([g_t*z_t, g_a*z_a])`; an MLP maps `2*d_model -> d_model
    -> num_classes`.

    `text_encoder`/`audio_encoder`: optional pre-built encoder instances
    (e.g. constructed with `_model=` stub overrides, CC-4) — if omitted,
    real frozen encoders are constructed from `cfg.model.text_ckpt`/
    `audio_ckpt`. `tokenizer_len`/`new_token_ids` are only used when
    `text_encoder` is omitted (passed straight through to `TextEncoder`).

    `forward(batch)` accepts either CC-3's `Batch` (raw audio -> runs
    `self.audio_encoder`) or CC-5's `CachedBatch` (precomputed
    `audio_hidden`/`frame_mask` -> `AudioEncoder` is never called),
    detected via `hasattr(batch, "audio_hidden")`.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer_len: int,
        new_token_ids: list[int],
        text_encoder: TextEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder if text_encoder is not None else TextEncoder(cfg, tokenizer_len, new_token_ids)
        self.audio_encoder = audio_encoder if audio_encoder is not None else AudioEncoder(cfg)

        d_model = cfg.model.d_model
        n_heads = cfg.model.n_heads
        n_blocks = cfg.model.n_fusion_blocks
        num_classes = cfg.model.num_classes

        self.proj_t = nn.Linear(self.text_encoder.model.config.hidden_size, d_model)
        self.proj_a = nn.Linear(self.audio_encoder.model.config.hidden_size, d_model)

        self.blocks = nn.ModuleList([CrossModalBlock(d_model, n_heads) for _ in range(n_blocks)])

        self.pool_t = AttnPool(d_model)
        self.pool_a = AttnPool(d_model)

        self.gate = nn.Linear(2 * d_model, 2)
        nn.init.zeros_(self.gate.bias)  # starts balanced ~0.5/0.5 (BACKLOG T-14)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def _fuse(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        audio_hidden: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> Output:
        text_mask = text_mask.bool()
        frame_mask = frame_mask.bool()

        t = self.proj_t(text_hidden)
        a = self.proj_a(audio_hidden)

        for block in self.blocks:
            t, a = block(t, a, text_mask, frame_mask)

        z_t = self.pool_t(t, text_mask)
        z_a = self.pool_a(a, frame_mask)

        gate_logits = self.gate(torch.cat([z_t, z_a], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)  # (B, 2), simplex (INV2)

        fused = torch.cat([gate[:, 0:1] * z_t, gate[:, 1:2] * z_a], dim=-1)  # (B, 2*d_model)
        logits = self.mlp(fused)

        return Output(logits=logits, gate=gate, xattn=None)

    def forward(self, batch) -> Output:
        """`batch`: CC-3's `Batch` (raw audio) or CC-5's `CachedBatch` (precomputed audio_hidden)."""
        text_hidden, text_mask = self.text_encoder(batch.text_ids, batch.text_mask)
        if hasattr(batch, "audio_hidden"):
            audio_hidden, frame_mask = batch.audio_hidden, batch.frame_mask
        else:
            audio_hidden, frame_mask = self.audio_encoder(batch.audio_values, batch.audio_mask)
        return self._fuse(text_hidden, text_mask, audio_hidden, frame_mask)

    def embed_text_ids(self, text_ids: torch.Tensor) -> torch.Tensor:
        """The text encoder's own input-embedding lookup for `text_ids` -- the XAI attribution starting point."""
        return self.text_encoder.model.get_input_embeddings()(text_ids)

    def forward_from_embeds(
        self,
        text_embeds: torch.Tensor,
        audio_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> Output:
        """XAI hook (T-18+): run the SAME fusion pipeline from precomputed embeddings, not token ids.

        `text_embeds` `(B, T_t, hidden_size)` bypasses the text encoder's
        embedding-lookup layer entirely (passed as `inputs_embeds` to the
        underlying transformers model, which every RoBERTa-family model
        accepts as an `input_ids` alternative) but still runs through every
        transformer layer — this is what lets Integrated Gradients attribute
        through arbitrary (e.g. interpolated) input embeddings.
        `audio_hidden` is the audio encoder's OWN output resolution
        (pre-`proj_a`, matching what `forward`'s live/cached audio path
        produces) — this method never touches the audio encoder itself.
        Returns identical logits to `forward` when fed that call's own
        `embed_text_ids(text_ids)` output for the same input (tested).
        """
        outputs = self.text_encoder.model(inputs_embeds=text_embeds, attention_mask=text_mask)
        text_hidden = outputs.last_hidden_state
        return self._fuse(text_hidden, text_mask, audio_hidden, frame_mask)
