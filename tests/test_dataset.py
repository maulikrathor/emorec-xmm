"""Tests for emorec.data.dataset: build_context (golden), MeldDataset, collate, subsetting.

Runs fully offline / with no MELD data: manifests and WAVs are synthesized
in tmp_path via soundfile (no ffmpeg needed). The golden-string tests never
touch a tokenizer (build_context is pure). Tokenizer-dependent tests use a
real roberta-base tokenizer if it can be fetched, else fall back to a tiny
local stub so the suite still runs fully offline.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from emorec.data.dataset import (
    MeldDataset,
    build_context,
    collate,
    load_audio,
    make_collate_fn,
    make_dataloader,
    stratified_subset_indices,
)
from emorec.data.labels import EMOTION_TO_ID, ID_TO_EMOTION

# ---------------------------------------------------------------------------
# Tokenizer: real roberta-base if reachable, else a tiny offline stub.
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal stand-in satisfying the tokenizer interface MeldDataset actually uses.

    NOT a real BPE tokenizer -- only used when roberta-base can't be
    fetched (offline CI). Golden-string tests never depend on this
    (build_context is pure); tests that need real-BPE-specific behavior
    check `using_real_tokenizer` and skip otherwise.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.pad_token_id = 0

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        added = 0
        for tok in special_tokens_dict.get("additional_special_tokens", []):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["<unk>"])

    def _id_for(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def __call__(self, text: str, truncation: bool = True, max_length: int | None = None) -> dict:
        ids = [self._id_for(tok) for tok in text.split(" ")]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _real_or_stub_tokenizer() -> tuple[object, bool]:
    # Broad except is deliberate here (test-only network fallback, not a
    # silent pass): any failure to fetch roberta-base degrades to a real,
    # documented substitute so the suite still runs fully offline.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("roberta-base")
        tok.add_special_tokens({"additional_special_tokens": ["<cur>", "</cur>"]})
        return tok, True
    except Exception:  # noqa: BLE001 -- deliberate network-fallback boundary, see comment above
        stub = _StubTokenizer()
        stub.add_special_tokens({"additional_special_tokens": ["<cur>", "</cur>"]})
        return stub, False


@pytest.fixture(scope="module")
def _tokenizer_and_flag() -> tuple[object, bool]:
    return _real_or_stub_tokenizer()


@pytest.fixture(scope="module")
def tokenizer(_tokenizer_and_flag: tuple[object, bool]) -> object:
    return _tokenizer_and_flag[0]


@pytest.fixture(scope="module")
def using_real_tokenizer(_tokenizer_and_flag: tuple[object, bool]) -> bool:
    return _tokenizer_and_flag[1]


# ---------------------------------------------------------------------------
# Shared fixtures: cfg, manifest + WAVs
# ---------------------------------------------------------------------------


@pytest.fixture
def base_cfg():
    return OmegaConf.create(
        {
            "data": {
                "context_k": 2,
                "max_text_len": 160,
                "max_audio_sec": 10.0,
                "sample_rate": 16000,
                "cur_token": "<cur>",
                "end_cur_token": "</cur>",
                "context_sep": "</s>",
                "n_workers": None,
            },
            "model": {"text_ckpt": "roberta-base"},
            "train": {"batch_size": 2},
        }
    )


def _write_wav(path: Path, duration_s: float, sample_rate: int = 16000, amplitude: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    data = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, data, sample_rate, subtype="PCM_16")


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def dialogue_rows() -> list[dict]:
    """The exact 3-turn dialogue fixture specified by the ticket, as full manifest-row dicts."""
    speakers = ["Chandler", "Monica", "Ross"]
    texts = ["Hey.", "Hi there!", "I'm fine."]
    emotions = ["neutral", "joy", "sadness"]
    return [
        {
            "utt_id": f"test/dia0_utt{t}",
            "dialogue_id": 0,
            "turn_idx": t,
            "speaker": speakers[t],
            "text": texts[t],
            "label": EMOTION_TO_ID[emotions[t]],
            "wav_path": f"artifacts/wav/test/dia0_utt{t}.wav",
            "duration": 1.0,
        }
        for t in range(3)
    ]


@pytest.fixture
def sample_manifest(tmp_path: Path, dialogue_rows: list[dict]) -> Path:
    wav_dir = tmp_path / "wav"
    rows = []
    durations = [1.0, 2.0, 0.5]
    for row in dialogue_rows:
        t = row["turn_idx"]
        wav_path = wav_dir / f"dia0_utt{t}.wav"
        _write_wav(wav_path, duration_s=durations[t])
        rows.append({**row, "wav_path": str(wav_path), "duration": durations[t]})
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, rows)
    return manifest_path


# ---------------------------------------------------------------------------
# T-CRIT-4: golden context strings. build_context is pure -- no tokenizer.
# ---------------------------------------------------------------------------


def test_golden_context_string_k0_t2(dialogue_rows: list[dict]) -> None:
    assert build_context(dialogue_rows, t=2, k=0) == "<cur> [Ross]: I'm fine. </cur>"


def test_golden_context_string_k1_t2(dialogue_rows: list[dict]) -> None:
    assert (
        build_context(dialogue_rows, t=2, k=1)
        == "[Monica]: Hi there! </s> <cur> [Ross]: I'm fine. </cur>"
    )


def test_golden_context_string_k2_t2(dialogue_rows: list[dict]) -> None:
    assert (
        build_context(dialogue_rows, t=2, k=2)
        == "[Chandler]: Hey. </s> [Monica]: Hi there! </s> <cur> [Ross]: I'm fine. </cur>"
    )


def test_golden_context_string_k2_t0_dialogue_start(dialogue_rows: list[dict]) -> None:
    assert build_context(dialogue_rows, t=0, k=2) == "<cur> [Chandler]: Hey. </cur>"


def test_golden_context_string_k2_t1(dialogue_rows: list[dict]) -> None:
    assert (
        build_context(dialogue_rows, t=1, k=2)
        == "[Chandler]: Hey. </s> <cur> [Monica]: Hi there! </cur>"
    )


def test_build_context_invalid_t_raises(dialogue_rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        build_context(dialogue_rows, t=3, k=2)


def test_build_context_negative_k_raises(dialogue_rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        build_context(dialogue_rows, t=0, k=-1)


# ---------------------------------------------------------------------------
# cur/end_cur token ids appear exactly once
# ---------------------------------------------------------------------------


def test_cur_and_end_cur_ids_appear_exactly_once(base_cfg, sample_manifest, tokenizer) -> None:
    ds = MeldDataset(sample_manifest, base_cfg, tokenizer=tokenizer)
    item = ds[2]  # t=2, context_k=2 -> both context turns included
    ids = item["input_ids"].tolist()
    assert ids.count(ds.cur_token_id) == 1
    assert ids.count(ds.end_cur_token_id) == 1


# ---------------------------------------------------------------------------
# END-keeping truncation (ED5) + short-audio passthrough
# ---------------------------------------------------------------------------


def test_end_keeping_truncation(tmp_path: Path) -> None:
    sr = 16000
    n = 12 * sr  # 12 s
    ramp = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    wav_path = tmp_path / "ramp.wav"
    sf.write(wav_path, ramp, sr, subtype="PCM_16")

    kept = load_audio(wav_path, sample_rate=sr, max_audio_sec=10.0)

    assert len(kept) == 160000
    expected_first = ramp[n - 160000]  # the ramp's value at t=2s
    assert float(kept[0]) == pytest.approx(float(expected_first), abs=1e-3)
    assert float(kept[-1]) == pytest.approx(1.0, abs=1e-3)
    # Not the ramp's start value -- proves truncation removed the front, not the back.
    assert abs(float(kept[0])) > 0.05


def test_audio_shorter_than_max_returned_unchanged(tmp_path: Path) -> None:
    sr = 16000
    n = int(2.5 * sr)
    data = np.linspace(-0.5, 0.5, num=n, dtype=np.float32)
    wav_path = tmp_path / "short.wav"
    sf.write(wav_path, data, sr, subtype="PCM_16")

    kept = load_audio(wav_path, sample_rate=sr, max_audio_sec=10.0)

    assert len(kept) == n
    np.testing.assert_allclose(kept, data, atol=1e-3)  # PCM16 round-trip quantization


def test_sample_rate_mismatch_raises_with_both_rates(tmp_path: Path) -> None:
    wav_path = tmp_path / "wrong_sr.wav"
    sf.write(wav_path, np.zeros(8000, dtype=np.float32), 8000, subtype="PCM_16")

    with pytest.raises(ValueError) as excinfo:
        load_audio(wav_path, sample_rate=16000, max_audio_sec=10.0)

    message = str(excinfo.value)
    assert "16000" in message
    assert "8000" in message
    assert str(wav_path) in message


# ---------------------------------------------------------------------------
# load_audio=False
# ---------------------------------------------------------------------------


def test_load_audio_false_returns_zero_length_and_still_collates(base_cfg, sample_manifest, tokenizer) -> None:
    ds = MeldDataset(sample_manifest, base_cfg, tokenizer=tokenizer, load_audio=False)
    item = ds[0]
    assert item["audio"].shape == (0,)

    batch = collate([ds[0], ds[1]], pad_token_id=tokenizer.pad_token_id)
    assert batch.audio_values.shape == (2, 0)
    assert batch.audio_mask.shape == (2, 0)


# ---------------------------------------------------------------------------
# collate: mixed text/audio lengths
# ---------------------------------------------------------------------------


def test_collate_mixed_lengths(base_cfg, sample_manifest, tokenizer) -> None:
    ds = MeldDataset(sample_manifest, base_cfg, tokenizer=tokenizer)
    items = [ds[0], ds[1], ds[2]]
    batch = collate(items, pad_token_id=tokenizer.pad_token_id)

    assert batch.text_ids.shape == batch.text_mask.shape
    assert batch.audio_values.shape == batch.audio_mask.shape
    assert batch.text_ids.shape[0] == 3
    assert batch.audio_values.shape[0] == 3

    for i, item in enumerate(items):
        n_text = item["input_ids"].shape[0]
        n_audio = item["audio"].shape[0]

        assert batch.text_mask[i].sum().item() == n_text
        assert batch.audio_mask[i].sum().item() == n_audio

        if n_text < batch.text_ids.shape[1]:
            assert torch.all(batch.text_ids[i, n_text:] == tokenizer.pad_token_id)
            assert torch.all(batch.text_mask[i, n_text:] == 0)
        if n_audio < batch.audio_values.shape[1]:
            assert torch.all(batch.audio_values[i, n_audio:] == 0.0)
            assert torch.all(~batch.audio_mask[i, n_audio:])


def test_batch_to_moves_tensors_and_keeps_meta(base_cfg, sample_manifest, tokenizer) -> None:
    ds = MeldDataset(sample_manifest, base_cfg, tokenizer=tokenizer)
    batch = collate([ds[0], ds[1]], pad_token_id=tokenizer.pad_token_id)
    moved = batch.to("cpu")
    assert moved.text_ids.device.type == "cpu"
    assert moved.meta == batch.meta


def test_make_collate_fn_matches_collate(base_cfg, sample_manifest, tokenizer) -> None:
    ds = MeldDataset(sample_manifest, base_cfg, tokenizer=tokenizer)
    collate_fn = make_collate_fn(tokenizer)
    a = collate_fn([ds[0], ds[1]])
    b = collate([ds[0], ds[1]], pad_token_id=tokenizer.pad_token_id)
    assert torch.equal(a.text_ids, b.text_ids)
    assert torch.equal(a.audio_values, b.audio_values)


# ---------------------------------------------------------------------------
# Labels are frozen ints, round-trip via ID_TO_EMOTION
# ---------------------------------------------------------------------------


def test_labels_are_frozen_ints_and_roundtrip(base_cfg, sample_manifest, tokenizer) -> None:
    ds = MeldDataset(sample_manifest, base_cfg, tokenizer=tokenizer)
    expected_emotions = ["neutral", "joy", "sadness"]
    for i, expected in enumerate(expected_emotions):
        label = ds[i]["label"]
        assert isinstance(label, int)
        assert ID_TO_EMOTION[label] == expected


def test_meld_dataset_rejects_invalid_label(tmp_path: Path, base_cfg, tokenizer) -> None:
    rows = [
        {
            "utt_id": "test/dia0_utt0",
            "dialogue_id": 0,
            "turn_idx": 0,
            "speaker": "A",
            "text": "hi",
            "label": 999,  # not in ID_TO_EMOTION
            "wav_path": "nope.wav",
            "duration": 1.0,
        }
    ]
    manifest_path = tmp_path / "bad_manifest.jsonl"
    _write_manifest(manifest_path, rows)

    with pytest.raises(ValueError, match="ID_TO_EMOTION"):
        MeldDataset(manifest_path, base_cfg, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# stratified_subset_indices
# ---------------------------------------------------------------------------


def _make_label_only_manifest(tmp_path: Path, name: str, counts: dict[str, int]) -> Path:
    """A manifest with no real audio -- fine, since .labels() never reads wav files."""
    rows = []
    turn_idx = 0
    for emotion, count in counts.items():
        for _ in range(count):
            rows.append(
                {
                    "utt_id": f"{name}/dia0_utt{turn_idx}",
                    "dialogue_id": 0,
                    "turn_idx": turn_idx,
                    "speaker": "A",
                    "text": "x",
                    "label": EMOTION_TO_ID[emotion],
                    "wav_path": str(tmp_path / "nonexistent.wav"),
                    "duration": 1.0,
                }
            )
            turn_idx += 1
    manifest_path = tmp_path / f"{name}.jsonl"
    _write_manifest(manifest_path, rows)
    return manifest_path


@pytest.fixture
def imbalanced_dataset(tmp_path, base_cfg, tokenizer):
    counts = {
        "neutral": 40,
        "joy": 30,
        "sadness": 15,
        "anger": 10,
        "surprise": 3,
        "fear": 1,
        "disgust": 1,
    }
    manifest_path = _make_label_only_manifest(tmp_path, "imbalanced", counts)
    ds = MeldDataset(manifest_path, base_cfg, tokenizer=tokenizer, load_audio=False)
    return ds, counts


def test_stratified_subset_same_seed_identical(imbalanced_dataset) -> None:
    ds, _counts = imbalanced_dataset
    idx1 = stratified_subset_indices(ds, n=20, seed=42)
    idx2 = stratified_subset_indices(ds, n=20, seed=42)
    assert idx1 == idx2
    assert idx1 == sorted(idx1)
    assert len(idx1) == 20


def test_stratified_subset_proportions_within_one(imbalanced_dataset) -> None:
    ds, counts = imbalanced_dataset
    n = 20
    total = sum(counts.values())
    idx = stratified_subset_indices(ds, n=n, seed=7)

    labels = ds.labels()
    got = Counter(labels[i] for i in idx)
    for emotion, count in counts.items():
        label_id = EMOTION_TO_ID[emotion]
        expected = n * count / total
        actual = got.get(label_id, 0)
        assert abs(actual - expected) <= 1 + 1e-9


def test_stratified_subset_rare_class_fully_included(tmp_path, base_cfg, tokenizer) -> None:
    # A=19, B(rare)=1. n=19 -> raw shares 18.05 / 0.95; largest-remainder
    # rounding gives the rare class its leftover unit deterministically
    # (0.95 remainder beats 0.05), so it ends up fully included (quota==1).
    counts = {"neutral": 19, "joy": 1}
    manifest_path = _make_label_only_manifest(tmp_path, "rare", counts)
    ds = MeldDataset(manifest_path, base_cfg, tokenizer=tokenizer, load_audio=False)

    idx = stratified_subset_indices(ds, n=19, seed=1)
    labels = ds.labels()
    got = Counter(labels[i] for i in idx)

    assert got[EMOTION_TO_ID["joy"]] == 1  # the rare class, fully included
    assert len(idx) == 19


def test_stratified_subset_requesting_all_never_raises(imbalanced_dataset) -> None:
    ds, counts = imbalanced_dataset
    total = sum(counts.values())
    idx = stratified_subset_indices(ds, n=total, seed=3)
    assert len(idx) == total
    assert idx == sorted(set(idx))


def test_stratified_subset_n_exceeding_total_raises(imbalanced_dataset) -> None:
    ds, counts = imbalanced_dataset
    total = sum(counts.values())
    with pytest.raises(ValueError):
        stratified_subset_indices(ds, n=total + 1, seed=0)


# ---------------------------------------------------------------------------
# Deterministic DataLoader
# ---------------------------------------------------------------------------


def test_seeded_dataloader_identical_ordering(base_cfg, sample_manifest, tokenizer) -> None:
    ds = MeldDataset(sample_manifest, base_cfg, tokenizer=tokenizer)
    collate_fn = make_collate_fn(tokenizer)

    def order(seed: int) -> list[str]:
        dl = make_dataloader(ds, base_cfg, shuffle=True, seed=seed, collate_fn=collate_fn)
        ids: list[str] = []
        for batch in dl:
            ids.extend(m["utt_id"] for m in batch.meta)
        return ids

    order_a = order(123)
    order_b = order(123)
    assert order_a == order_b
    assert sorted(order_a) == sorted(ds[i]["meta"]["utt_id"] for i in range(len(ds)))


# ---------------------------------------------------------------------------
# configs/base.yaml sanity: still loads, only the 3 new keys added
# ---------------------------------------------------------------------------


def test_configs_base_yaml_still_loads_with_new_keys() -> None:
    from emorec.utils.config import load_config

    cfg = load_config("configs/base.yaml")
    assert cfg.data.cur_token == "<cur>"
    assert cfg.data.end_cur_token == "</cur>"
    assert cfg.data.context_sep == "</s>"


# ---------------------------------------------------------------------------
# Real-tokenizer-only sanity (skipped when running on the offline stub)
# ---------------------------------------------------------------------------


def test_real_tokenizer_produces_nontrivial_bpe(using_real_tokenizer, tokenizer) -> None:
    if not using_real_tokenizer:
        pytest.skip("roberta-base not reachable; running on the offline stub tokenizer")
    encoded = tokenizer("[Ross]: I'm fine.", truncation=True, max_length=160)
    # A real BPE tokenizer splits "I'm" into multiple subword pieces plus
    # adds RoBERTa's <s>/</s> special tokens -- more ids than whitespace words.
    assert len(encoded["input_ids"]) > len(["[Ross]:", "I'm", "fine."])
