"""CachedAudioMeldDataset: MeldDataset variant reading CC-4's on-disk fp16 audio cache.

CONTRACT: identical interface to `MeldDataset`, EXCEPT `__getitem__` returns
`"audio_hidden"` (precomputed `(T_f, hidden_size)` float32) instead of
`"audio"` (raw waveform) when `use_cache=True` (the default). This is what
lets `scripts/train.py` skip both the `AudioEncoder` forward pass AND the
WAV disk read on every epoch — CC-4's whole reason for existing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from emorec.data.dataset import MeldDataset
from emorec.models.encoders import cache_key, cache_path, load_cached_features


class CachedAudioMeldDataset(MeldDataset):
    """`MeldDataset`, but audio comes from CC-4's on-disk cache instead of raw WAVs.

    `cache_root` is `{cfg.data.cache_dir}/{cache_key(cfg.model.audio_ckpt)}`
    (CC-4's checkpoint-hashed cache directory). `use_cache=False` disables
    caching entirely and falls back to `MeldDataset`'s live-waveform
    `"audio"` behavior, so a caller can switch between cached and online
    audio with one flag instead of swapping dataset classes.

    Raises `FileNotFoundError` (naming the `utt_id` and the expected path)
    if a required `.npz` is missing — a partially-built cache must fail
    loud here, never silently substitute anything.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cfg: DictConfig,
        tokenizer: Any = None,
        split: str | None = None,
        subset_indices: list[int] | None = None,
        use_cache: bool = True,
    ) -> None:
        # Audio is either loaded from cache (below) or live (delegated to
        # MeldDataset) -- never both, so load_audio is exactly `not use_cache`
        # to avoid MeldDataset doing a pointless WAV read when caching.
        super().__init__(
            manifest_path,
            cfg,
            tokenizer=tokenizer,
            split=split,
            subset_indices=subset_indices,
            load_audio=not use_cache,
        )
        self.use_cache = use_cache
        self.cache_root = Path(cfg.data.cache_dir) / cache_key(cfg.model.audio_ckpt)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        if not self.use_cache:
            return item

        utt_id = item["meta"]["utt_id"]
        path = cache_path(self.cache_root, utt_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached audio features for utt_id={utt_id!r} at {path}. "
                "Run scripts/build_cache.py for this manifest first, or construct "
                "this dataset with use_cache=False to fall back to live audio."
            )
        hidden, _frame_mask = load_cached_features(path)

        item = dict(item)
        del item["audio"]
        item["audio_hidden"] = hidden
        return item


@dataclass
class CachedBatch:
    """A collated batch of cached-audio items: padded text, padded precomputed audio features."""

    text_ids: torch.Tensor  # (B, T_t) long
    text_mask: torch.Tensor  # (B, T_t) long, 1 = real token
    audio_hidden: torch.Tensor  # (B, T_f, hidden_size) float32
    frame_mask: torch.Tensor  # (B, T_f) bool, True = real (cached) frame
    label: torch.Tensor  # (B,) long
    meta: list[dict]

    def to(self, device: torch.device | str) -> "CachedBatch":
        """Move all tensor fields to `device`; `meta` is plain Python and is left as-is."""
        return CachedBatch(
            text_ids=self.text_ids.to(device),
            text_mask=self.text_mask.to(device),
            audio_hidden=self.audio_hidden.to(device),
            frame_mask=self.frame_mask.to(device),
            label=self.label.to(device),
            meta=self.meta,
        )


def collate_cached(batch: list[dict], pad_token_id: int) -> CachedBatch:
    """Pad a list of `CachedAudioMeldDataset.__getitem__` items to a `CachedBatch`.

    Text padding mirrors `emorec.data.dataset.collate` exactly. `audio_hidden`
    is zero-padded to the batch's longest `T_f`; `frame_mask` is True for
    real (cached) frames, False for pad — mirroring CC-4's cache being
    unbatched (one utterance, no padding, per `.npz` file).
    """
    batch_size = len(batch)

    text_lens = [item["input_ids"].shape[0] for item in batch]
    max_text_len = max(text_lens)
    text_ids = torch.full((batch_size, max_text_len), pad_token_id, dtype=torch.long)
    text_mask = torch.zeros((batch_size, max_text_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = text_lens[i]
        text_ids[i, :n] = item["input_ids"]
        text_mask[i, :n] = 1

    hidden_size = batch[0]["audio_hidden"].shape[-1]
    frame_lens = [item["audio_hidden"].shape[0] for item in batch]
    max_frames = max(frame_lens) if frame_lens else 0
    audio_hidden = torch.zeros((batch_size, max_frames, hidden_size), dtype=torch.float32)
    frame_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = frame_lens[i]
        if n > 0:
            audio_hidden[i, :n] = item["audio_hidden"]
            frame_mask[i, :n] = True

    label = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    meta = [item["meta"] for item in batch]

    return CachedBatch(
        text_ids=text_ids,
        text_mask=text_mask,
        audio_hidden=audio_hidden,
        frame_mask=frame_mask,
        label=label,
        meta=meta,
    )


def make_collate_fn_cached(tokenizer: Any) -> Callable[[list[dict]], CachedBatch]:
    """Return a picklable `collate_cached` partial bound to `tokenizer.pad_token_id` (see `emorec.data.dataset.make_collate_fn`)."""
    return partial(collate_cached, pad_token_id=tokenizer.pad_token_id)
