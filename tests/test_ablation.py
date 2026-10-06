"""Tests for emorec.xai.ablation (CC-9): inference-time modality ablation on a trained model.

Offline: tiny random-init RoBERTa/WavLM stubs injected via the encoders' `_model=` override.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaModel, WavLMConfig, WavLMModel

from emorec.data.dataset import collate
from emorec.models.baselines import TextOnly
from emorec.models.encoders import AudioEncoder, TextEncoder
from emorec.models.fusion import MultimodalERC
from emorec.xai.ablation import MODES, ablate_modality, derangement, run_ablation_suite

TEXT_VOCAB = 200
NEW_TOKEN_IDS = [TEXT_VOCAB, TEXT_VOCAB + 1]
PAD_TOKEN_ID = 1


def _cfg():
    return OmegaConf.create(
        {
            "data": {"sample_rate": 16000},
            "model": {
                "name": "xattn",
                "text_ckpt": "unused",
                "audio_ckpt": "unused",
                "d_model": 16,
                "n_heads": 2,
                "n_fusion_blocks": 2,
                "num_classes": 7,
            },
        }
    )


def _text_encoder() -> TextEncoder:
    stub = RobertaModel(
        RobertaConfig(vocab_size=TEXT_VOCAB, hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32)
    )
    return TextEncoder(_cfg(), tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS, _model=stub)


def _model() -> MultimodalERC:
    torch.manual_seed(0)
    audio = AudioEncoder(
        _cfg(), _model=WavLMModel(WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32))
    )
    return MultimodalERC(_cfg(), TEXT_VOCAB + 2, NEW_TOKEN_IDS, text_encoder=_text_encoder(), audio_encoder=audio).eval()


class _Items(Dataset):
    """Variable text/audio lengths, so batches are genuinely padded."""

    def __init__(self, n: int) -> None:
        g = torch.Generator().manual_seed(0)
        self.items = [
            {
                "input_ids": torch.randint(2, TEXT_VOCAB, (4 + i % 5,), generator=g),
                "audio": torch.randn(2500 + 300 * (i % 4), generator=g),
                "label": i % 7,
                "meta": {"utt_id": f"dev/dia{i}_utt0"},
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def _loader(n: int = 9, batch_size: int = 3) -> DataLoader:
    return DataLoader(_Items(n), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate(b, PAD_TOKEN_ID))


def test_none_mode_reproduces_model_forward_logits() -> None:
    model = _model()
    loader = _loader()
    with torch.no_grad():
        expected = torch.cat([model(b).logits for b in loader]).numpy()

    out = ablate_modality(model, loader, "none")

    np.testing.assert_allclose(out["logits"], expected, atol=1e-6)
    assert out["utt_ids"] == [f"dev/dia{i}_utt0" for i in range(9)]
    assert out["y_true"] == [i % 7 for i in range(9)]
    assert out["y_pred"] == expected.argmax(axis=1).tolist()
    assert len(out["y_prob"]) == 9 and {"accuracy", "macro_f1", "weighted_f1", "per_class_f1"} <= set(out)
    assert 0.0 < out["mean_g_audio"] < 1.0 and out["mean_g_text"] == pytest.approx(1 - out["mean_g_audio"], abs=1e-6)


def test_zero_audio_and_zero_text_change_the_logits() -> None:
    model = _model()
    loader = _loader()
    base = ablate_modality(model, loader, "none")["logits"]

    for mode in ("zero_audio", "zero_text"):
        out = ablate_modality(model, loader, mode)["logits"]
        assert np.abs(out - base).max() > 1e-4, f"{mode} did not change the logits"


def test_ablation_does_not_modify_the_model() -> None:
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for mode in MODES:
        ablate_modality(model, _loader(), mode)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


@pytest.mark.parametrize("n", [2, 3, 5, 8, 17])
def test_derangement_has_no_fixed_points_and_is_a_permutation(n: int) -> None:
    for seed in range(20):
        perm = derangement(n, np.random.default_rng(seed))
        assert sorted(perm) == list(range(n))
        assert all(i != p for i, p in enumerate(perm))


def test_derangement_is_reproducible_and_degenerate_for_one() -> None:
    assert derangement(8, np.random.default_rng(3)) == derangement(8, np.random.default_rng(3))
    assert derangement(1, np.random.default_rng(0)) == [0]


def test_shuffle_audio_is_reproducible_and_pairs_every_example() -> None:
    model = _model()
    base = ablate_modality(model, _loader(), "none")["logits"]

    a = ablate_modality(model, _loader(), "shuffle_audio", seed=7)
    b = ablate_modality(model, _loader(), "shuffle_audio", seed=7)
    c = ablate_modality(model, _loader(), "shuffle_audio", seed=8)

    np.testing.assert_array_equal(a["logits"], b["logits"])
    assert a["n_unshuffled"] == 0  # batch size 3 > 1: derangement exists for every batch
    assert np.abs(a["logits"] - base).max() > 1e-4
    assert np.abs(a["logits"] - c["logits"]).max() > 1e-6  # a different seed pairs differently


def test_shuffle_audio_counts_unpairable_singleton_batch() -> None:
    out = ablate_modality(_model(), _loader(n=4, batch_size=3), "shuffle_audio")  # batches of 3 and 1
    assert out["n_unshuffled"] == 1


def test_suite_has_one_row_per_mode_with_zero_control_deltas() -> None:
    df = run_ablation_suite(_model(), _loader(), device=None, seed=1)

    assert df["mode"].tolist() == list(MODES)
    for col in ("accuracy", "macro_f1", "weighted_f1", "delta_wF1", "delta_macro", "f1_neutral", "f1_disgust"):
        assert col in df.columns
    control = df.loc[df["mode"] == "none"].iloc[0]
    assert control["delta_wF1"] == 0.0 and control["delta_macro"] == 0.0
    shuffled = df.loc[df["mode"] == "shuffle_audio"].iloc[0]
    assert shuffled["delta_wF1"] == pytest.approx(shuffled["weighted_f1"] - control["weighted_f1"])


def test_unknown_mode_and_unsupported_model_raise() -> None:
    with pytest.raises(ValueError, match="unknown ablation mode"):
        ablate_modality(_model(), _loader(), "zero_everything")
    text_only = TextOnly(_cfg(), TEXT_VOCAB + 2, NEW_TOKEN_IDS, text_encoder=_text_encoder()).eval()
    with pytest.raises(NotImplementedError):
        ablate_modality(text_only, _loader(), "none")
