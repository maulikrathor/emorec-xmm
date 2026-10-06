"""Tests for emorec.xai.faithfulness (CC-9): deletion curves, random control, randomisation sanity check.

Curve-shape tests use a tiny ADDITIVE toy model whose per-token / per-frame
contributions are known exactly, so a "perfectly faithful" attribution can be
written down. The sanity-check tests use real tiny RoBERTa/WavLM fusion stubs.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers import RobertaConfig, RobertaModel, WavLMConfig, WavLMModel

from emorec.data.cached_dataset import CachedBatch
from emorec.data.dataset import Batch
from emorec.models.encoders import AudioEncoder, TextEncoder
from emorec.models.fusion import MultimodalERC, Output
from emorec.xai.faithfulness import (
    DEFAULT_FRACTIONS,
    combine_curves,
    comprehensiveness,
    faithfulness_auc,
    mean_ci,
    per_example_auc,
    random_baseline_curves,
    randomized_head_copy,
    sanity_check_randomization,
    sufficiency,
)
from emorec.xai.ig import attribute_audio, attribute_text

PAD_ID = 1


# ---------------------------------------------------------------------------
# Additive toy model with exactly known contributions
# ---------------------------------------------------------------------------


class _Additive(nn.Module):
    """logit[0] = sum of (masked) text-token values + sum of (masked) audio-frame values; other logits 0.

    The pad embedding is 0, so replacing a token by pad removes exactly its
    contribution; zeroing an audio frame removes exactly its value. Values are
    small (logit ~ 4) so the softmax is not saturated and drops stay measurable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(50, 1)
        with torch.no_grad():
            self.emb.weight.copy_(0.1 + 0.2 * ((0.37 * torch.arange(50, dtype=torch.float32)) % 1.0)[:, None])
            self.emb.weight[PAD_ID] = 0.0
        self.text_encoder = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(pad_token_id=PAD_ID)))

    def embed_text_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)

    def forward_from_embeds(self, text_embeds, audio_hidden, text_mask, frame_mask) -> Output:
        s = (text_embeds[..., 0] * text_mask).sum(1) + (audio_hidden[..., 0] * frame_mask).sum(1)
        logits = torch.zeros(text_embeds.shape[0], 7)
        logits[:, 0] = s
        return Output(logits=logits, gate=None, xattn=None)


def _toy_batch(b: int = 3, t: int = 12, f: int = 16, seed: int = 0) -> CachedBatch:
    g = torch.Generator().manual_seed(seed)
    text_ids = torch.randint(2, 50, (b, t), generator=g)
    text_mask = torch.ones(b, t, dtype=torch.long)
    text_ids[:, -2:] = PAD_ID  # right padding, masked out
    text_mask[:, -2:] = 0
    frame_mask = torch.ones(b, f, dtype=torch.bool)
    frame_mask[:, -3:] = False
    audio_hidden = 0.05 + 0.2 * torch.rand(b, f, 1, generator=g)
    audio_hidden[~frame_mask] = 0.0
    return CachedBatch(
        text_ids=text_ids, text_mask=text_mask, audio_hidden=audio_hidden, frame_mask=frame_mask,
        label=torch.zeros(b, dtype=torch.long), meta=[{"utt_id": f"dev/dia{i}_utt0"} for i in range(b)],
    )  # fmt: skip


def _perfect_attr(model: _Additive, batch: CachedBatch, modality: str) -> np.ndarray:
    """The exact per-position contribution to the target logit: a perfectly faithful attribution."""
    if modality == "text":
        return model.embed_text_ids(batch.text_ids)[..., 0].detach().numpy()
    return batch.audio_hidden[..., 0].numpy()


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_comprehensiveness_on_perfect_attribution_is_monotone_non_decreasing(modality: str) -> None:
    model, batch = _Additive(), _toy_batch()
    curve = comprehensiveness(model, batch, _perfect_attr(model, batch, modality), modality)

    assert curve["per_example"].shape == (3, len(DEFAULT_FRACTIONS))
    assert (curve["per_example"] > 0).all()
    assert (np.diff(curve["per_example"], axis=1) >= -1e-7).all()
    assert (np.diff(curve["mean"]) >= -1e-7).all()


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_sufficiency_on_perfect_attribution_is_monotone_non_increasing(modality: str) -> None:
    model, batch = _Additive(), _toy_batch()
    curve = sufficiency(model, batch, _perfect_attr(model, batch, modality), modality)

    assert (np.diff(curve["per_example"], axis=1) <= 1e-7).all()
    assert (np.diff(curve["mean"]) <= 1e-7).all()


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_perfect_attribution_beats_random_control(modality: str) -> None:
    model, batch = _Additive(), _toy_batch()
    perfect = _perfect_attr(model, batch, modality)
    rand = random_baseline_curves(model, batch, modality, seed=3, n_draws=8)

    assert faithfulness_auc(comprehensiveness(model, batch, perfect, modality)) > faithfulness_auc(
        rand["comprehensiveness"]
    )
    assert faithfulness_auc(sufficiency(model, batch, perfect, modality)) < faithfulness_auc(rand["sufficiency"])


def test_padding_positions_are_never_selected() -> None:
    model, batch = _Additive(), _toy_batch()
    attr = _perfect_attr(model, batch, "text").copy()
    plain = comprehensiveness(model, batch, attr, "text")

    attr[:, -2:] = 1e9  # huge attribution on masked pad positions must not matter
    poisoned = comprehensiveness(model, batch, attr, "text")

    np.testing.assert_allclose(plain["per_example"], poisoned["per_example"], atol=1e-7)


def test_attr_shape_mismatch_raises() -> None:
    model, batch = _Additive(), _toy_batch()
    with pytest.raises(ValueError, match="does not match"):
        comprehensiveness(model, batch, np.zeros((3, 5)), "text")


def test_faithfulness_auc_matches_hand_computed_trapezoid() -> None:
    curve = {"fractions": np.array([0.05, 0.1, 0.2, 0.5]), "mean": np.array([0.1, 0.2, 0.4, 0.5])}
    hand = 0.05 * (0.1 + 0.2) / 2 + 0.1 * (0.2 + 0.4) / 2 + 0.3 * (0.4 + 0.5) / 2
    assert hand == pytest.approx(0.1725)
    assert faithfulness_auc(curve) == pytest.approx(hand)

    per = {"fractions": curve["fractions"], "per_example": np.array([[0.1, 0.2, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0]])}
    np.testing.assert_allclose(per_example_auc(per), [hand, 0.0])


def test_random_baseline_curves_reproducible_under_fixed_seed() -> None:
    model, batch = _Additive(), _toy_batch()
    a = random_baseline_curves(model, batch, "text", seed=5, n_draws=3)
    b = random_baseline_curves(model, batch, "text", seed=5, n_draws=3)
    c = random_baseline_curves(model, batch, "text", seed=6, n_draws=3)

    for kind in ("comprehensiveness", "sufficiency"):
        np.testing.assert_array_equal(a[kind]["per_example"], b[kind]["per_example"])
        assert a[kind]["kind"] == kind and a[kind]["modality"] == "text"
    assert not np.array_equal(a["comprehensiveness"]["per_example"], c["comprehensiveness"]["per_example"])


def test_combine_curves_and_mean_ci() -> None:
    model = _Additive()
    c1 = comprehensiveness(model, _toy_batch(2, seed=1), _perfect_attr(model, _toy_batch(2, seed=1), "text"), "text")
    c2 = comprehensiveness(model, _toy_batch(4, seed=2), _perfect_attr(model, _toy_batch(4, seed=2), "text"), "text")
    both = combine_curves([c1, c2])
    assert both["per_example"].shape == (6, 4)
    np.testing.assert_allclose(both["mean"], both["per_example"].mean(axis=0))

    ci = mean_ci(per_example_auc(both), n=200, seed=1)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert mean_ci(per_example_auc(both), n=200, seed=1) == ci


# ---------------------------------------------------------------------------
# Randomisation sanity check (real tiny fusion stub)
# ---------------------------------------------------------------------------


def _cfg():
    return OmegaConf.create(
        {
            "data": {"sample_rate": 16000},
            "model": {"name": "xattn", "text_ckpt": "u", "audio_ckpt": "u", "d_model": 16, "n_heads": 2,
                      "n_fusion_blocks": 2, "num_classes": 7},
        }
    )  # fmt: skip


def _fusion_model() -> MultimodalERC:
    torch.manual_seed(0)
    text = TextEncoder(
        _cfg(), 202, [200, 201],
        _model=RobertaModel(RobertaConfig(vocab_size=200, hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32)),
    )  # fmt: skip
    audio = AudioEncoder(
        _cfg(), _model=WavLMModel(WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32))
    )
    return MultimodalERC(_cfg(), 202, [200, 201], text_encoder=text, audio_encoder=audio).eval()


def _fusion_batch(b: int = 2) -> Batch:
    g = torch.Generator().manual_seed(1)
    return Batch(
        text_ids=torch.randint(2, 200, (b, 8), generator=g),
        text_mask=torch.ones(b, 8, dtype=torch.long),
        audio_values=torch.randn(b, 4000, generator=g),
        audio_mask=torch.ones(b, 4000, dtype=torch.bool),
        label=torch.zeros(b, dtype=torch.long),
        meta=[{"utt_id": f"dev/dia{i}_utt0"} for i in range(b)],
    )


def test_sanity_check_randomization_never_mutates_the_input_model() -> None:
    model, batch = _fusion_model(), _fusion_batch()
    attr = {
        "text": attribute_text(model, batch, n_steps=8)["attr"],
        "audio": attribute_audio(model, batch, n_steps=8)["attr"],
    }
    before = {k: v.clone() for k, v in model.state_dict().items()}
    head = model.mlp
    head_before = [p.detach().clone() for p in head.parameters()]

    result = sanity_check_randomization(model, batch, attr, seed=3, n_steps=8)

    assert model.mlp is head
    for p, p0 in zip(head.parameters(), head_before, strict=True):
        assert torch.equal(p, p0)
    after = model.state_dict()
    assert after.keys() == before.keys()
    for k, v in after.items():
        assert torch.equal(v, before[k]), f"{k} was mutated"

    assert set(result["modalities"]) == {"text", "audio"} and set(result["broken"]) == {"text", "audio"}
    assert result["head"] == "mlp" and result["seed"] == 3
    for mod in ("text", "audio"):
        rho = result["modalities"][mod]["spearman"]
        assert len(rho) == 2 and all(-1.0 <= r <= 1.0 or np.isnan(r) for r in rho)
        assert isinstance(result["broken"][mod], bool)


def test_randomized_head_copy_differs_is_seeded_and_shares_the_rest() -> None:
    model = _fusion_model()
    c1, c2, c3 = (randomized_head_copy(model, s) for s in (1, 1, 2))

    w, w1, w2, w3 = (m.mlp[0].weight for m in (model, c1, c2, c3))
    assert not torch.equal(w, w1)  # the head really was re-initialised
    assert torch.equal(w1, w2)  # seeded
    assert not torch.equal(w1, w3)
    assert c1.text_encoder is model.text_encoder and c1.audio_encoder is model.audio_encoder  # shared, not copied
    assert c1.mlp is not model.mlp
    assert model.mlp[0].weight is w  # untouched
