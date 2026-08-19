"""The tensor interface every model consumes: manifest JSONL -> MeldDataset -> collate -> Batch.

CONTRACT: this module is the ONLY place dialogue-context strings are
assembled (`build_context`, a pure function — golden-string tested) and the
ONLY place audio truncation happens (ED5: END-keeping, i.e. truncate from
the START so utterance-final prosody survives). Every model and script
consumes data through `MeldDataset` + `collate`/`make_collate_fn` — never
re-implement context assembly or truncation elsewhere.
"""

import json
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from emorec.data.labels import ID_TO_EMOTION

# Literal markers for the text input contract (docs/ARCHITECTURE.md "Text
# input contract"). `build_context` is a PURE function per CONTRACT 1 — it
# takes no cfg/tokenizer, so these are hardcoded here rather than read from
# config. They are kept equal to configs/base.yaml's data.cur_token /
# data.end_cur_token / data.context_sep by construction; MeldDataset reads
# the config values (not these constants) when registering special tokens
# with the tokenizer, so the two are wired together at the config layer.
CUR_TOKEN = "<cur>"
END_CUR_TOKEN = "</cur>"
CONTEXT_SEP = "</s>"


def build_context(dialogue_rows: list[dict], t: int, k: int) -> str:
    """Assemble the text-input-contract string for turn `t` of one dialogue, with `k` turns of context.

    PURE FUNCTION: no tokenizer, no config, no I/O. `dialogue_rows` must
    already be sorted by turn_idx and belong to a single dialogue.

    Format (exact, docs/ARCHITECTURE.md):
        [spk(t-2)]: u_(t-2) </s> [spk(t-1)]: u_(t-1) </s> <cur> [spk(t)]: u_t </cur>

    The current turn is always LAST, always wrapped in `<cur> ... </cur>`.
    Context turns (up to `k` of them, fewer at dialogue start — NEVER
    padded) are joined with " </s> ", with a trailing " </s> " before the
    `<cur>` marker. `k=0`, or `t` at the dialogue start with no prior turns,
    produces ONLY `"<cur> [spk(t)]: u_t </cur>"`. Speaker names are used
    verbatim (MELD literals) — never parsed back out of this string;
    structured fields belong in the manifest row, not this string.
    """
    n = len(dialogue_rows)
    if not (0 <= t < n):
        raise ValueError(f"t={t} out of range for a dialogue with {n} turn(s)")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    def render(row: dict) -> str:
        return f"[{row['speaker']}]: {row['text']}"

    cur_row = dialogue_rows[t]
    cur_span = f"{CUR_TOKEN} {render(cur_row)} {END_CUR_TOKEN}"

    start = max(0, t - k)
    context_rows = dialogue_rows[start:t]
    if not context_rows:
        return cur_span

    context_span = f" {CONTEXT_SEP} ".join(render(row) for row in context_rows)
    return f"{context_span} {CONTEXT_SEP} {cur_span}"


def load_audio(wav_path: str | Path, sample_rate: int, max_audio_sec: float) -> np.ndarray:
    """Load a mono float32 waveform from `wav_path`, END-keeping-truncated to `max_audio_sec`.

    CONTRACT (ED5, binding): audio longer than `max_audio_sec * sample_rate`
    samples is truncated by keeping the LAST `max_audio_sec` seconds
    (`audio[-n:]`) — i.e. truncation removes from the START, because
    utterance-final prosody is what carries emotion. Audio at or under the
    limit is returned unchanged. Raises `ValueError` naming `wav_path` and
    both sample rates if the file's rate doesn't match `sample_rate`
    exactly — never silently resamples.
    """
    path = Path(wav_path)
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != sample_rate:
        raise ValueError(f"{path}: expected sample rate {sample_rate}, got {sr}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    max_samples = int(max_audio_sec * sample_rate)
    if len(data) > max_samples:
        data = data[-max_samples:]
    return np.ascontiguousarray(data, dtype=np.float32)


def _read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MeldDataset(Dataset):
    """Manifest JSONL -> tokenized text + raw audio + label, one utterance per item.

    Groups manifest rows by `dialogue_id`, sorts each dialogue by
    `turn_idx` ONCE at construction, and builds a flat `(dialogue_id, t)`
    index so `__getitem__` is O(1). Every manifest label is validated
    against `emorec.data.labels.ID_TO_EMOTION` at construction — a
    corrupted/foreign label id fails loud at load time, not mid-training.

    `tokenizer`: if `None`, builds `AutoTokenizer.from_pretrained(
    cfg.model.text_ckpt)` and registers `cfg.data.cur_token` /
    `cfg.data.end_cur_token` as additional special tokens. Either way,
    `self.tokenizer`, `self.cur_token_id`, and `self.end_cur_token_id` are
    always set — XAI (CC-4+) locates the `<cur>` span via these ids, never
    by re-parsing the context string.

    `load_audio=False` skips audio entirely (`__getitem__` returns a
    zero-length audio array) — used by the text-only baseline to avoid
    pointless disk reads.

    `subset_indices`: an optional list of flat indices restricting the
    dataset to those items (e.g. a `stratified_subset_indices` result).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cfg: DictConfig,
        tokenizer: Any = None,
        split: str | None = None,
        subset_indices: list[int] | None = None,
        load_audio: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.cfg = cfg
        self.split = split
        self.load_audio = load_audio

        rows = _read_manifest(self.manifest_path)

        self.dialogues: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            label = int(row["label"])
            if label not in ID_TO_EMOTION:
                raise ValueError(
                    f"manifest row {row.get('utt_id')!r} has label id {label} not in "
                    f"emorec.data.labels.ID_TO_EMOTION ({sorted(ID_TO_EMOTION)}) — corrupted manifest?"
                )
            self.dialogues[row["dialogue_id"]].append(row)
        for dialogue_rows in self.dialogues.values():
            dialogue_rows.sort(key=lambda r: r["turn_idx"])

        self._flat_index: list[tuple[int, int]] = [
            (dialogue_id, t)
            for dialogue_id in sorted(self.dialogues)
            for t in range(len(self.dialogues[dialogue_id]))
        ]
        if subset_indices is not None:
            self._flat_index = [self._flat_index[i] for i in subset_indices]

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_ckpt)
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [cfg.data.cur_token, cfg.data.end_cur_token]}
            )
        self.tokenizer = tokenizer
        self.cur_token_id: int = tokenizer.convert_tokens_to_ids(cfg.data.cur_token)
        self.end_cur_token_id: int = tokenizer.convert_tokens_to_ids(cfg.data.end_cur_token)

    def __len__(self) -> int:
        return len(self._flat_index)

    def labels(self) -> list[int]:
        """Label ids aligned with the flat index order — no tokenization, no audio I/O.

        Used by `stratified_subset_indices` to compute class-proportional
        quotas cheaply, without paying `__getitem__`'s tokenization/audio
        cost for every row just to read a label.
        """
        return [self.dialogues[d][t]["label"] for d, t in self._flat_index]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dialogue_id, t = self._flat_index[idx]
        dialogue_rows = self.dialogues[dialogue_id]
        row = dialogue_rows[t]

        context = build_context(dialogue_rows, t, self.cfg.data.context_k)
        encoded = self.tokenizer(context, truncation=True, max_length=self.cfg.data.max_text_len)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

        if self.load_audio:
            audio = load_audio(row["wav_path"], self.cfg.data.sample_rate, self.cfg.data.max_audio_sec)
        else:
            audio = np.zeros(0, dtype=np.float32)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "audio": torch.from_numpy(audio),
            "label": int(row["label"]),
            "meta": {
                "utt_id": row["utt_id"],
                "context": context,
                "speaker": row["speaker"],
                "text": row["text"],
                "duration": float(row["duration"]),
            },
        }


@dataclass
class Batch:
    """A collated batch: padded text, padded audio, labels, and per-item metadata."""

    text_ids: torch.Tensor  # (B, T_t) long
    text_mask: torch.Tensor  # (B, T_t) long, 1 = real token
    audio_values: torch.Tensor  # (B, T_a) float32
    audio_mask: torch.Tensor  # (B, T_a) bool, True = real sample
    label: torch.Tensor  # (B,) long
    meta: list[dict]

    def to(self, device: torch.device | str) -> "Batch":
        """Move all tensor fields to `device`; `meta` is plain Python and is left as-is."""
        return Batch(
            text_ids=self.text_ids.to(device),
            text_mask=self.text_mask.to(device),
            audio_values=self.audio_values.to(device),
            audio_mask=self.audio_mask.to(device),
            label=self.label.to(device),
            meta=self.meta,
        )


def collate(batch: list[dict], pad_token_id: int) -> Batch:
    """Pad a list of `MeldDataset.__getitem__` items to a `Batch`.

    Text is padded to the longest sequence IN THE BATCH with
    `pad_token_id`; `text_mask` is 1 for real tokens, 0 for pad. Audio is
    zero-padded to the longest waveform IN THE BATCH; `audio_mask` is True
    for real samples, False for pad.

    NOTE: `audio_mask` is at RAW SAMPLE resolution. The encoder wrapper
    (CC-4) must convert it to frame resolution via
    `model._get_feature_vector_attention_mask` — never hand-roll the conv
    stride arithmetic (docs/TRIBAL_KNOWLEDGE.md §1.2).
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

    audio_lens = [item["audio"].shape[0] for item in batch]
    max_audio_len = max(audio_lens)
    audio_values = torch.zeros((batch_size, max_audio_len), dtype=torch.float32)
    audio_mask = torch.zeros((batch_size, max_audio_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = audio_lens[i]
        if n > 0:
            audio_values[i, :n] = item["audio"]
            audio_mask[i, :n] = True

    label = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    meta = [item["meta"] for item in batch]

    return Batch(
        text_ids=text_ids,
        text_mask=text_mask,
        audio_values=audio_values,
        audio_mask=audio_mask,
        label=label,
        meta=meta,
    )


def make_collate_fn(tokenizer: Any) -> Callable[[list[dict]], Batch]:
    """Return a picklable `collate` partial bound to `tokenizer.pad_token_id`.

    `functools.partial` over the module-level `collate` function is
    picklable (unlike a closure or lambda), so this is safe to pass as
    `DataLoader(collate_fn=...)` with `num_workers > 0`.
    """
    return partial(collate, pad_token_id=tokenizer.pad_token_id)


def make_dataloader(
    dataset: MeldDataset,
    cfg: DictConfig,
    shuffle: bool,
    seed: int,
    collate_fn: Callable[[list[dict]], Batch],
) -> DataLoader:
    """Build a `DataLoader` with reproducible shuffling and per-worker seeding.

    A `torch.Generator` seeded with `seed` drives shuffling, so two
    `DataLoader`s built with the same `seed` (and no other state changes)
    yield identical batch orderings. `num_workers` comes from
    `cfg.data.n_workers`, defaulting to `0` (not the `min(8, cpu_count())`
    convention used for preprocessing — 0 keeps loading in the main
    process, which is what determinism and CI stability need here) when
    unset. `worker_init_fn` seeds `numpy`/`random`/`torch` in each worker
    process from `torch.initial_seed()`, the standard recipe for avoiding
    identical RNG streams across workers.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    num_workers = cfg.data.n_workers or 0

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def stratified_subset_indices(dataset: MeldDataset, n: int, seed: int) -> list[int]:
    """A class-proportional, seeded, reproducible sample of `n` flat indices from `dataset`.

    Each class's quota is its proportional share of `n` (largest-remainder
    rounding, so no class's quota differs from its exact proportional share
    by more than 1 sample) UNLESS the class has fewer than `n` * its class
    share members, in which case ALL of that class's members are taken and
    the shortfall is redistributed proportionally across the remaining
    classes (recursively, since redistribution can itself overflow another
    small class). Selection within each class is a seeded random sample
    without replacement. Returns a SORTED list of flat indices. Two calls
    with the same `dataset` and `seed` return an identical list.

    This is how the pilot train subset is drawn: report `n` and `seed`
    alongside ANY metric computed from the resulting subset, or the number
    is not reproducible.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    labels = dataset.labels()
    total = len(labels)
    if n > total:
        raise ValueError(f"requested subset size n={n} exceeds dataset size {total}")

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    remaining_classes = set(by_class)
    quota: dict[int, int] = {}
    remaining_n = n
    while remaining_classes:
        remaining_total = sum(len(by_class[c]) for c in remaining_classes)
        if remaining_total == 0:
            break

        raw = {c: remaining_n * len(by_class[c]) / remaining_total for c in remaining_classes}
        base = {c: int(raw[c]) for c in remaining_classes}
        leftover = remaining_n - sum(base.values())
        # Largest-remainder method: give the leftover units to the classes
        # with the biggest fractional remainder; ties broken by class id
        # for determinism.
        order = sorted(remaining_classes, key=lambda c: (-(raw[c] - base[c]), c))
        for c in order[:leftover]:
            base[c] += 1

        overflow = [c for c in remaining_classes if base[c] > len(by_class[c])]
        if not overflow:
            quota.update(base)
            break

        for c in overflow:
            quota[c] = len(by_class[c])
            remaining_n -= len(by_class[c])
            remaining_classes.discard(c)
        # Loop again: redistribute remaining_n over the shrunk remaining_classes.

    rng = random.Random(seed)
    selected: list[int] = []
    for c in sorted(by_class):
        k = quota.get(c, 0)
        if k > 0:
            selected.extend(rng.sample(sorted(by_class[c]), k))

    return sorted(selected)
