"""B1/B2/B3 baselines: same encoder wrappers as MultimodalERC, differing ONLY in fusion stage.

CONTRACT: B1/B2/B3 all return `Output` (gate=None, xattn=None — no gate,
no cross-attention, both are properties only the fusion model has). Sharing
`TextEncoder`/`AudioEncoder` and the post-projection (256-d) pooling design
with `MultimodalERC` is what makes the M-vs-B3 comparison a clean ablation
of exactly the cross-attention+gate mechanism (BACKLOG J-04) — B3 pools the
SAME post-`proj_t`/`proj_a` streams M does, just without the CrossModalBlocks
and gate in between.
"""

import torch
from omegaconf import DictConfig
from torch import nn

from emorec.models.encoders import AudioEncoder, TextEncoder
from emorec.models.fusion import AttnPool, Output


class TextOnly(nn.Module):
    """B1: proj_t -> AttnPool -> MLP -> num_classes. `gate=None`.

    MUST NOT touch `AudioEncoder` at all — it is never constructed, never
    called, so `forward` works even on a batch whose audio fields are the
    CC-4 zero-width `load_audio=False` sentinel (which would raise if
    `AudioEncoder` were ever invoked on it).
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer_len: int,
        new_token_ids: list[int],
        text_encoder: TextEncoder | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder if text_encoder is not None else TextEncoder(cfg, tokenizer_len, new_token_ids)

        d_model = cfg.model.d_model
        self.proj_t = nn.Linear(self.text_encoder.model.config.hidden_size, d_model)
        self.pool_t = AttnPool(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, cfg.model.num_classes))

    def forward(self, batch) -> Output:
        text_hidden, text_mask = self.text_encoder(batch.text_ids, batch.text_mask)
        t = self.proj_t(text_hidden)
        z_t = self.pool_t(t, text_mask.bool())
        logits = self.mlp(z_t)
        return Output(logits=logits, gate=None, xattn=None)


class AudioOnly(nn.Module):
    """B2: proj_a -> AttnPool -> MLP -> num_classes. `gate=None`. Never touches `TextEncoder`."""

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer_len: int | None = None,
        new_token_ids: list[int] | None = None,
        audio_encoder: AudioEncoder | None = None,
    ) -> None:
        super().__init__()
        # tokenizer_len/new_token_ids accepted (unused) only so the
        # registry can dispatch every model class with the same call
        # signature -- AudioOnly has no text path at all.
        del tokenizer_len, new_token_ids
        self.audio_encoder = audio_encoder if audio_encoder is not None else AudioEncoder(cfg)

        d_model = cfg.model.d_model
        self.proj_a = nn.Linear(self.audio_encoder.model.config.hidden_size, d_model)
        self.pool_a = AttnPool(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, cfg.model.num_classes))

    def forward(self, batch) -> Output:
        if hasattr(batch, "audio_hidden"):
            audio_hidden, frame_mask = batch.audio_hidden, batch.frame_mask
        else:
            audio_hidden, frame_mask = self.audio_encoder(batch.audio_values, batch.audio_mask)
        a = self.proj_a(audio_hidden)
        z_a = self.pool_a(a, frame_mask.bool())
        logits = self.mlp(z_a)
        return Output(logits=logits, gate=None, xattn=None)


class Concat(nn.Module):
    """B3: independently pool BOTH post-projection (256-d) streams, concat, MLP. `gate=None`.

    This is the base paper's fusion method, kept as a control — it is
    architecturally identical to `MultimodalERC` up through `proj_t`/
    `proj_a`, then skips the `CrossModalBlock`s and gate entirely and pools
    each stream independently before concatenating (BACKLOG J-04).
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
        self.proj_t = nn.Linear(self.text_encoder.model.config.hidden_size, d_model)
        self.proj_a = nn.Linear(self.audio_encoder.model.config.hidden_size, d_model)
        self.pool_t = AttnPool(d_model)
        self.pool_a = AttnPool(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, cfg.model.num_classes)
        )

    def forward(self, batch) -> Output:
        text_hidden, text_mask = self.text_encoder(batch.text_ids, batch.text_mask)
        if hasattr(batch, "audio_hidden"):
            audio_hidden, frame_mask = batch.audio_hidden, batch.frame_mask
        else:
            audio_hidden, frame_mask = self.audio_encoder(batch.audio_values, batch.audio_mask)

        t = self.proj_t(text_hidden)
        a = self.proj_a(audio_hidden)
        z_t = self.pool_t(t, text_mask.bool())
        z_a = self.pool_a(a, frame_mask.bool())

        fused = torch.cat([z_t, z_a], dim=-1)
        logits = self.mlp(fused)
        return Output(logits=logits, gate=None, xattn=None)
