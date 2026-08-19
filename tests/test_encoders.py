"""Tests for emorec.models.encoders: frozen wrappers, frame masking, and the audio cache.

NO network, NO real checkpoints, NO GPU: text/audio base models are tiny
random-init stubs (transformers RobertaModel / WavLMModel with reduced
hidden sizes but the REAL default WavLM conv feature-extractor stride
stack), injected via the `_model=` constructor override so the wrappers'
own freezing/masking/cache logic is what's under test, not a fake.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from transformers import RobertaConfig, RobertaModel, WavLMConfig, WavLMModel

from emorec.models.encoders import (
    AudioEncoder,
    TextEncoder,
    cache_key,
    cache_path,
    load_cached_features,
    save_cached_features,
)

TEXT_VOCAB = 200
TEXT_HIDDEN = 32
AUDIO_HIDDEN = 32
NEW_TOKEN_IDS = [TEXT_VOCAB, TEXT_VOCAB + 1]  # the two <cur>/</cur> ids, appended after resize


def _make_cfg(sample_rate: int = 16000) -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {"sample_rate": sample_rate, "cache_dir": "artifacts/cache"},
            "model": {"text_ckpt": "unused-stub-text-ckpt", "audio_ckpt": "unused-stub-audio-ckpt"},
        }
    )


def _stub_text_model() -> RobertaModel:
    cfg = RobertaConfig(
        vocab_size=TEXT_VOCAB,
        hidden_size=TEXT_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    return RobertaModel(cfg)


def _stub_audio_model() -> WavLMModel:
    # Reduced hidden/layers for speed; conv_kernel/conv_stride left at
    # WavLMConfig's real defaults ([10,3,3,3,3,2,2] / [5,2,2,2,2,2,2]) so
    # the frame-mask arithmetic under test is the real one, not a fake.
    cfg = WavLMConfig(
        hidden_size=AUDIO_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    return WavLMModel(cfg)


def _make_text_encoder() -> TextEncoder:
    return TextEncoder(
        _make_cfg(),
        tokenizer_len=TEXT_VOCAB + 2,
        new_token_ids=NEW_TOKEN_IDS,
        _model=_stub_text_model(),
    )


def _make_audio_encoder(cfg: OmegaConf | None = None) -> AudioEncoder:
    return AudioEncoder(cfg or _make_cfg(), _model=_stub_audio_model())


# ---------------------------------------------------------------------------
# TextEncoder: T-CRIT-3 / INV3 freeze correctness
# ---------------------------------------------------------------------------


def test_text_encoder_only_embedding_weight_is_trainable() -> None:
    enc = _make_text_encoder()
    trainable_params = [p for _, p in enc.named_parameters() if p.requires_grad]
    assert len(trainable_params) == 1
    assert trainable_params[0] is enc.model.get_input_embeddings().weight


def test_text_encoder_freeze_hook_zeroes_grad_except_new_rows() -> None:
    torch.manual_seed(0)  # deterministic: see the loss-choice comment below for why this matters
    enc = _make_text_encoder()

    batch, seq_len = 2, 6
    ids = torch.randint(0, TEXT_VOCAB, (batch, seq_len))
    ids[0, 0] = NEW_TOKEN_IDS[0]  # force lookup of one new row so its grad can be nonzero
    mask = torch.ones(batch, seq_len, dtype=torch.long)

    hidden, _ = enc(ids, mask)
    # Sum of SQUARES, not a plain sum: a plain sum through LayerNorm's
    # (near-)zero-mean output can leave the embedding-row gradient so close
    # to zero it underflows to exactly 0.0 in float32 for some random
    # inits -- a false negative, not a real absence of gradient flow.
    # Squaring avoids that cancellation and reliably produces a detectable
    # nonzero gradient.
    (hidden**2).sum().backward()

    grad = enc.model.get_input_embeddings().weight.grad
    assert grad is not None

    zero_rows = sorted(set(range(grad.shape[0])) - set(NEW_TOKEN_IDS))
    assert torch.all(grad[zero_rows] == 0)
    assert torch.any(grad[NEW_TOKEN_IDS[0]] != 0) or torch.any(grad[NEW_TOKEN_IDS[1]] != 0)


def test_text_encoder_n_trainable_equals_two_times_hidden_size() -> None:
    enc = _make_text_encoder()
    n = enc.n_trainable()
    print(f"TextEncoder.n_trainable() = {n} (2 * hidden_size={TEXT_HIDDEN} -> expected {2 * TEXT_HIDDEN})")
    assert n == 2 * TEXT_HIDDEN


def test_text_encoder_forward_shapes_and_mask_passthrough() -> None:
    enc = _make_text_encoder()
    ids = torch.randint(0, TEXT_VOCAB, (3, 10))
    mask = torch.ones(3, 10, dtype=torch.long)
    with torch.no_grad():
        hidden, out_mask = enc(ids, mask)
    assert hidden.shape == (3, 10, TEXT_HIDDEN)
    assert torch.equal(out_mask, mask)


def test_text_encoder_train_override_keeps_wrapped_model_in_eval() -> None:
    enc = _make_text_encoder()
    enc.train()
    assert enc.model.training is False
    assert enc.training is True  # the wrapper's OWN train flag still flips


def test_text_encoder_eval_mode_invariance() -> None:
    enc = _make_text_encoder()
    ids = torch.randint(0, TEXT_VOCAB, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        h1, _ = enc(ids, mask)
        h2, _ = enc(ids, mask)
    assert torch.equal(h1, h2)


# ---------------------------------------------------------------------------
# AudioEncoder: all frozen, frame mask correctness
# ---------------------------------------------------------------------------


def test_audio_encoder_all_params_frozen() -> None:
    enc = _make_audio_encoder()
    assert all(not p.requires_grad for p in enc.parameters())
    assert enc.n_trainable() == 0


def test_audio_encoder_sample_rate_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="16000"):
        AudioEncoder(_make_cfg(sample_rate=8000), _model=_stub_audio_model())


def test_audio_encoder_zero_width_audio_raises_named_error() -> None:
    enc = _make_audio_encoder()
    audio_values = torch.zeros(2, 0)
    audio_mask = torch.zeros(2, 0, dtype=torch.bool)
    with pytest.raises(ValueError, match="load_audio=False"):
        enc(audio_values, audio_mask)


def test_audio_encoder_train_override_keeps_wrapped_model_in_eval() -> None:
    enc = _make_audio_encoder()
    enc.train()
    assert enc.model.training is False


def test_audio_encoder_eval_mode_invariance() -> None:
    enc = _make_audio_encoder()
    audio_values = torch.randn(2, 8000)
    audio_mask = torch.ones(2, 8000, dtype=torch.bool)
    with torch.no_grad():
        h1, m1 = enc(audio_values, audio_mask)
        h2, m2 = enc(audio_values, audio_mask)
    assert torch.equal(h1, h2)
    assert torch.equal(m1, m2)


def _padded_batch(raw_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    t_a = max(raw_lengths)
    batch = len(raw_lengths)
    audio_values = torch.randn(batch, t_a)
    audio_mask = torch.zeros(batch, t_a, dtype=torch.bool)
    for i, n in enumerate(raw_lengths):
        audio_mask[i, :n] = True
    return audio_values, audio_mask


def test_frame_mask_shape_and_strictly_increasing_with_length() -> None:
    enc = _make_audio_encoder()
    raw_lengths = [4000, 8000, 16000]
    audio_values, audio_mask = _padded_batch(raw_lengths)

    with torch.no_grad():
        hidden, frame_mask = enc(audio_values, audio_mask)

    assert frame_mask.shape[1] == hidden.shape[1]
    assert frame_mask.dtype == torch.bool

    sums = frame_mask.sum(dim=1)
    assert sums[0] < sums[1] < sums[2]
    # The shortest example's trailing frames must be False (not attended).
    assert not bool(frame_mask[0, int(sums[0]) :].any())


def test_frame_mask_matches_transformers_output_lengths() -> None:
    enc = _make_audio_encoder()
    raw_len = 16000
    audio_values = torch.randn(1, raw_len)
    audio_mask = torch.ones(1, raw_len, dtype=torch.bool)

    with torch.no_grad():
        hidden, frame_mask = enc(audio_values, audio_mask)

    # The wrapper must agree with transformers' OWN output-length
    # computation, not with a hand-rolled formula.
    expected = int(enc.model._get_feat_extract_output_lengths(torch.tensor([raw_len])).item())
    assert int(frame_mask.sum().item()) == expected
    assert frame_mask.shape[1] == hidden.shape[1]


# ---------------------------------------------------------------------------
# T-CRIT-5: audio feature cache equivalence
# ---------------------------------------------------------------------------


def test_cache_equivalence(tmp_path: Path) -> None:
    enc = _make_audio_encoder()
    torch.manual_seed(0)
    raw_lengths = [4000, 9000]
    audio_values, audio_mask = _padded_batch(raw_lengths)

    with torch.no_grad():
        hidden, frame_mask = enc(audio_values, audio_mask)

    cache_root = tmp_path / "cache" / cache_key("dummy-ckpt")
    utt_ids = ["dev/dia0_utt0", "dev/dia0_utt1"]

    for i, uid in enumerate(utt_ids):
        true_len = int(frame_mask[i].sum().item())
        save_cached_features(
            cache_path(cache_root, uid),
            hidden[i, :true_len].to(torch.float16),
            frame_mask[i, :true_len],
        )

    for i, uid in enumerate(utt_ids):
        true_len = int(frame_mask[i].sum().item())
        loaded_hidden, loaded_mask = load_cached_features(cache_path(cache_root, uid))
        online_hidden = hidden[i, :true_len]
        online_mask = frame_mask[i, :true_len]

        assert loaded_hidden.dtype == torch.float32
        assert loaded_hidden.shape == online_hidden.shape
        torch.testing.assert_close(loaded_hidden, online_hidden, atol=1e-2, rtol=1e-2)
        assert torch.equal(loaded_mask, online_mask)


def test_cache_key_changes_with_checkpoint_string_and_is_deterministic() -> None:
    k1 = cache_key("microsoft/wavlm-base-plus")
    k2 = cache_key("facebook/wav2vec2-base-960h")
    assert k1 != k2
    assert len(k1) == 8
    assert cache_key("microsoft/wavlm-base-plus") == k1


def test_cache_path_namespaces_by_split(tmp_path: Path) -> None:
    root = tmp_path / "cache_root"
    p_dev = cache_path(root, "dev/dia0_utt0")
    p_train = cache_path(root, "train/dia0_utt0")
    assert p_dev == root / "dev" / "dia0_utt0.npz"
    assert p_train == root / "train" / "dia0_utt0.npz"
    assert p_dev.parent.is_dir()
    assert p_train.parent.is_dir()


def test_cache_path_requires_namespaced_utt_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="namespaced"):
        cache_path(tmp_path, "dia0_utt0")
