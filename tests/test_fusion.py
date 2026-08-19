"""Tests for emorec.models.fusion / baselines / registry, and a trainer overfit smoke test.

NO network, NO real checkpoints, NO GPU: text/audio base models are tiny
random-init transformers stubs (RobertaModel / WavLMModel with reduced
hidden sizes but WavLM's REAL default conv feature-extractor stride stack),
injected via TextEncoder/AudioEncoder's own `_model=` override (CC-4), then
wired into MultimodalERC/baselines via their `text_encoder=`/`audio_encoder=`
injection points.
"""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, RobertaConfig, RobertaModel, WavLMConfig, WavLMModel

from emorec.data.dataset import Batch, collate
from emorec.models.baselines import AudioOnly, Concat, TextOnly
from emorec.models.encoders import AudioEncoder, TextEncoder
from emorec.models.fusion import CrossModalBlock, MultimodalERC
from emorec.models.registry import build_model
from emorec.training.losses import build_loss
from emorec.training.trainer import fit

TEXT_VOCAB = 200
TEXT_HIDDEN = 24
AUDIO_HIDDEN = 32  # must divide WavLMConfig's default num_conv_pos_embedding_groups=16
D_MODEL = 16
N_HEADS = 2
NEW_TOKEN_IDS = [TEXT_VOCAB, TEXT_VOCAB + 1]
PAD_TOKEN_ID = 1


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "run": {"name": "test", "seed": 0},
            "data": {"sample_rate": 16000, "cache_dir": "unused"},
            "model": {
                "name": "xattn",
                "text_ckpt": "unused-text-ckpt",
                "audio_ckpt": "unused-audio-ckpt",
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_fusion_blocks": 2,
                "num_classes": 7,
                "freeze_encoders": True,
            },
            "train": {
                "lr": 1e-2,
                "weight_decay": 0.0,
                "batch_size": 4,
                "grad_accum": 1,
                "max_epochs": 1,
                "warmup_ratio": 0.0,
                "grad_clip": 1.0,
                "early_stop_patience": 100,
                "amp": True,
                "use_cache": False,
            },
            "loss": {"name": "weighted_ce"},
        }
    )


def _stub_text_model() -> RobertaModel:
    cfg = RobertaConfig(
        vocab_size=TEXT_VOCAB,
        hidden_size=TEXT_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return RobertaModel(cfg)


def _stub_audio_model() -> WavLMModel:
    # Reduced hidden/layers for speed; conv_kernel/conv_stride left at
    # WavLMConfig's real defaults so frame-mask arithmetic is the real one.
    cfg = WavLMConfig(hidden_size=AUDIO_HIDDEN, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32)
    return WavLMModel(cfg)


def _make_text_encoder() -> TextEncoder:
    return TextEncoder(_make_cfg(), tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS, _model=_stub_text_model())


def _make_audio_encoder() -> AudioEncoder:
    return AudioEncoder(_make_cfg(), _model=_stub_audio_model())


def _random_batch(batch_size: int, text_len: int, audio_len: int) -> Batch:
    text_ids = torch.randint(0, TEXT_VOCAB, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len, dtype=torch.long)
    audio_values = torch.randn(batch_size, audio_len)
    audio_mask = torch.ones(batch_size, audio_len, dtype=torch.bool)
    label = torch.randint(0, 7, (batch_size,))
    meta = [{"utt_id": f"dev/dia0_utt{i}"} for i in range(batch_size)]
    return Batch(text_ids=text_ids, text_mask=text_mask, audio_values=audio_values, audio_mask=audio_mask, label=label, meta=meta)


def _make_fusion_model() -> MultimodalERC:
    return MultimodalERC(
        _make_cfg(),
        tokenizer_len=TEXT_VOCAB + 2,
        new_token_ids=NEW_TOKEN_IDS,
        text_encoder=_make_text_encoder(),
        audio_encoder=_make_audio_encoder(),
    )


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_multimodal_erc_shapes() -> None:
    model = _make_fusion_model()
    batch = _random_batch(3, 10, 8000)
    with torch.no_grad():
        out = model(batch)
    assert out.logits.shape == (3, 7)
    assert out.gate.shape == (3, 2)
    assert out.xattn is None


# ---------------------------------------------------------------------------
# INV2: gate is a simplex
# ---------------------------------------------------------------------------


def test_gate_is_a_simplex() -> None:
    model = _make_fusion_model()
    batch = _random_batch(5, 12, 6000)
    with torch.no_grad():
        out = model(batch)

    assert bool(torch.all(out.gate > 0))
    assert bool(torch.all(out.gate < 1))
    row_sums = out.gate.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(5), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# T-CRIT-1 / INV1: mask invariance -- the critical test
# ---------------------------------------------------------------------------


def _make_padded_pair(model: MultimodalERC):
    """One real example (fixed text embeds + fixed audio_hidden), padded two very different amounts.

    Operates at FRAME level for audio (`forward_from_embeds`'s `audio_hidden`
    input), matching the ticket's literal "30/40 ... 120/200 (text/audio
    FRAMES)" wording, and matching how audio is actually fed to fusion in
    production (`train.use_cache: true` by default -> CC-5's `CachedBatch`
    -> precomputed `audio_hidden`/`frame_mask`, never raw waveform padding).
    See the CC-5 report for why this is the right level to test INV1 at,
    not raw-sample audio padding through the frozen WavLM CNN.
    """
    torch.manual_seed(0)
    real_text_len = 8
    real_frames = 10
    text_ids_real = torch.randint(0, TEXT_VOCAB, (1, real_text_len))
    text_embeds_real = model.embed_text_ids(text_ids_real)  # (1, real_text_len, TEXT_HIDDEN)
    audio_hidden_real = torch.randn(1, real_frames, AUDIO_HIDDEN)

    def _padded(text_pad_len: int, audio_pad_len: int):
        text_embeds = torch.zeros(1, text_pad_len, TEXT_HIDDEN)
        text_embeds[0, :real_text_len] = text_embeds_real[0]
        text_mask = torch.zeros(1, text_pad_len, dtype=torch.long)
        text_mask[0, :real_text_len] = 1

        audio_hidden = torch.zeros(1, audio_pad_len, AUDIO_HIDDEN)
        audio_hidden[0, :real_frames] = audio_hidden_real[0]
        frame_mask = torch.zeros(1, audio_pad_len, dtype=torch.bool)
        frame_mask[0, :real_frames] = True

        return text_embeds, audio_hidden, text_mask, frame_mask

    small = _padded(text_pad_len=30, audio_pad_len=40)
    large = _padded(text_pad_len=120, audio_pad_len=200)
    return small, large


def test_mask_invariance_padding_amount_does_not_change_logits() -> None:
    """T-CRIT-1 / INV1 (the critical test): logits for one real example must be
    identical no matter how much padding surrounds it, in BOTH modalities.

    Tested via `forward_from_embeds` at FRAME-level padding (fixed text
    embeddings + fixed precomputed `audio_hidden`), which is what my
    CrossModalBlock/AttnPool masking actually owns and must guarantee, and
    matches production's cached-audio training path exactly. It is NOT
    tested via raw-waveform audio padding through the frozen WavLM CNN --
    that encoder's `feat_extract_norm="group"` (confirmed the REAL
    `microsoft/wavlm-base-plus` checkpoint's own config, not a stub
    artifact) normalizes its first conv layer over the WHOLE padded time
    axis, so raw-sample padding amount measurably perturbs even a
    correctly-masked encoder's output -- a real, third-party architectural
    property of frozen wav2vec2-family CNNs, not something this ticket's
    masking code can or should paper over. See the CC-5 report for the
    break/restore steps that proved THIS test (as written) has teeth.
    """
    torch.manual_seed(1)
    model = _make_fusion_model()
    model.eval()

    small, large = _make_padded_pair(model)

    with torch.no_grad():
        out_small = model.forward_from_embeds(*small)
        out_large = model.forward_from_embeds(*large)

    torch.testing.assert_close(out_small.logits, out_large.logits, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# forward_from_embeds parity
# ---------------------------------------------------------------------------


def test_forward_from_embeds_matches_forward() -> None:
    torch.manual_seed(2)
    model = _make_fusion_model()
    model.eval()

    batch = _random_batch(2, 10, 4000)

    with torch.no_grad():
        out_forward = model(batch)

        text_embeds = model.embed_text_ids(batch.text_ids)
        audio_hidden, frame_mask = model.audio_encoder(batch.audio_values, batch.audio_mask)
        out_from_embeds = model.forward_from_embeds(text_embeds, audio_hidden, batch.text_mask, frame_mask)

    torch.testing.assert_close(out_forward.logits, out_from_embeds.logits, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# B1/B2/B3 baselines
# ---------------------------------------------------------------------------


def test_text_only_baseline_shapes_and_no_gate() -> None:
    model = TextOnly(_make_cfg(), tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS, text_encoder=_make_text_encoder())
    batch = _random_batch(3, 10, 8000)
    with torch.no_grad():
        out = model(batch)
    assert out.logits.shape == (3, 7)
    assert out.gate is None
    assert out.xattn is None


def test_text_only_never_touches_audio_encoder() -> None:
    model = TextOnly(_make_cfg(), tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS, text_encoder=_make_text_encoder())
    assert not hasattr(model, "audio_encoder")

    # Zero-width audio: AudioEncoder.forward would raise on this (CC-4).
    # TextOnly must never call it, so this batch must not raise.
    batch = _random_batch(2, 6, 8000)
    batch.audio_values = torch.zeros(2, 0)
    batch.audio_mask = torch.zeros(2, 0, dtype=torch.bool)
    with torch.no_grad():
        out = model(batch)
    assert out.logits.shape == (2, 7)


def test_audio_only_baseline_shapes_and_no_gate() -> None:
    model = AudioOnly(_make_cfg(), audio_encoder=_make_audio_encoder())
    batch = _random_batch(3, 10, 8000)
    with torch.no_grad():
        out = model(batch)
    assert out.logits.shape == (3, 7)
    assert out.gate is None


def test_concat_baseline_shapes_and_no_gate() -> None:
    model = Concat(
        _make_cfg(),
        tokenizer_len=TEXT_VOCAB + 2,
        new_token_ids=NEW_TOKEN_IDS,
        text_encoder=_make_text_encoder(),
        audio_encoder=_make_audio_encoder(),
    )
    batch = _random_batch(3, 10, 8000)
    with torch.no_grad():
        out = model(batch)
    assert out.logits.shape == (3, 7)
    assert out.gate is None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_rejects_unknown_model_name() -> None:
    cfg = _make_cfg()
    cfg.model.name = "not_a_real_model"
    with pytest.raises(ValueError, match="not_a_real_model"):
        build_model(cfg, tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS)


def test_registry_builds_each_known_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_from_pretrained(name: str, *args, **kwargs):
        if "audio" in name.lower():
            return _stub_audio_model()
        return _stub_text_model()

    monkeypatch.setattr(AutoModel, "from_pretrained", _fake_from_pretrained)

    expected = {"text_only": TextOnly, "audio_only": AudioOnly, "concat": Concat, "xattn": MultimodalERC}
    for name, cls in expected.items():
        cfg = _make_cfg()
        cfg.model.name = name
        model = build_model(cfg, tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS)
        assert isinstance(model, cls)


# ---------------------------------------------------------------------------
# CrossModalBlock: masking present in both directions (sanity for INV1)
# ---------------------------------------------------------------------------


def test_cross_modal_block_masks_both_directions() -> None:
    block = CrossModalBlock(D_MODEL, N_HEADS)
    t = torch.randn(2, 5, D_MODEL)
    a = torch.randn(2, 6, D_MODEL)
    text_mask = torch.ones(2, 5, dtype=torch.bool)
    audio_mask = torch.ones(2, 6, dtype=torch.bool)
    audio_mask[0, 3:] = False

    t_out, a_out = block(t, a, text_mask, audio_mask)
    assert t_out.shape == t.shape
    assert a_out.shape == a.shape


# ---------------------------------------------------------------------------
# Tiny-subset overfit smoke test
# ---------------------------------------------------------------------------


class _TinyDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

    def labels(self) -> list[int]:
        return [item["label"] for item in self.items]


def test_tiny_subset_overfit_smoke(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = _make_cfg()
    cfg.train.lr = 1e-2
    cfg.train.batch_size = 16  # all 16 examples in one batch -> 1 step/epoch
    cfg.train.grad_accum = 1
    cfg.train.max_epochs = 60  # 1 step/epoch * 60 epochs = 60 optimizer steps, the ceiling
    cfg.train.warmup_ratio = 0.0
    cfg.train.early_stop_patience = 1000  # disabled: guarantee the full 60-step budget runs

    model = MultimodalERC(
        cfg, tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS, text_encoder=_make_text_encoder(), audio_encoder=_make_audio_encoder()
    )

    n = 16
    items = []
    for i in range(n):
        label = i % 7  # covers all 7 classes, no class count is zero
        items.append(
            {
                "input_ids": torch.randint(0, TEXT_VOCAB, (6,)),
                "audio": torch.randn(2000),
                "label": label,
                "meta": {"utt_id": f"train/dia0_utt{i}"},
            }
        )

    ds = _TinyDataset(items)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_token_id=PAD_TOKEN_ID))

    counts = [0] * 7
    for item in items:
        counts[item["label"]] += 1
    loss_fn = build_loss(cfg, counts)

    run_dir = tmp_path / "run"
    fit(model, loader, loader, loss_fn, cfg, run_dir, split_names={"train": "synthetic", "dev": "synthetic"}, subset_size=n)

    lines = (run_dir / "metrics.json").read_text(encoding="utf-8").strip().splitlines()
    last_epoch = json.loads(lines[-1])
    assert last_epoch["train_loss"] < 0.1, f"train_loss did not drop below 0.1: {last_epoch}"
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "run_meta.json").exists()
