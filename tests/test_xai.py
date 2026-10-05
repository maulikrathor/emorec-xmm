"""Tests for emorec.xai (CC-8): IG attribution, gate analysis, figures.

NO network, NO real checkpoints, NO GPU: tiny random-init RoBERTa/WavLM stubs
injected via TextEncoder/AudioEncoder's `_model=` override (the same pattern as
tests/test_fusion.py, duplicated here so test files stay independent).
"""

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaModel, WavLMConfig, WavLMModel

from emorec.data.cached_dataset import CachedBatch
from emorec.data.dataset import Batch, collate
from emorec.models.baselines import TextOnly
from emorec.models.encoders import AudioEncoder, TextEncoder
from emorec.models.fusion import MultimodalERC
from emorec.xai import ig
from emorec.xai.gate import collect_gates, gate_summary
from emorec.xai.ig import (
    attribute_audio,
    attribute_text,
    attribution_path,
    cur_span,
    frames_to_times,
    save_attribution,
)
from emorec.xai.viz import (
    plot_audio_timeline,
    plot_gate_distribution,
    plot_per_class_comparison,
    plot_token_heatmap,
)

TEXT_VOCAB = 200
TEXT_HIDDEN = 24
AUDIO_HIDDEN = 32
D_MODEL = 16
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
                "d_model": D_MODEL,
                "n_heads": 2,
                "n_fusion_blocks": 2,
                "num_classes": 7,
            },
        }
    )


def _text_encoder() -> TextEncoder:
    stub = RobertaModel(
        RobertaConfig(
            vocab_size=TEXT_VOCAB,
            hidden_size=TEXT_HIDDEN,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    return TextEncoder(_cfg(), tokenizer_len=TEXT_VOCAB + 2, new_token_ids=NEW_TOKEN_IDS, _model=stub)


def _audio_encoder() -> AudioEncoder:
    stub = WavLMModel(WavLMConfig(hidden_size=AUDIO_HIDDEN, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32))
    return AudioEncoder(_cfg(), _model=stub)


def _fusion_model() -> MultimodalERC:
    torch.manual_seed(0)
    model = MultimodalERC(
        _cfg(), TEXT_VOCAB + 2, NEW_TOKEN_IDS, text_encoder=_text_encoder(), audio_encoder=_audio_encoder()
    )
    return model.eval()


def _text_only_model() -> TextOnly:
    torch.manual_seed(0)
    return TextOnly(_cfg(), TEXT_VOCAB + 2, NEW_TOKEN_IDS, text_encoder=_text_encoder()).eval()


def _batch(b: int = 2, text_len: int = 8, audio_len: int = 4000, seed: int = 1) -> Batch:
    g = torch.Generator().manual_seed(seed)
    return Batch(
        text_ids=torch.randint(2, TEXT_VOCAB, (b, text_len), generator=g),
        text_mask=torch.ones(b, text_len, dtype=torch.long),
        audio_values=torch.randn(b, audio_len, generator=g),
        audio_mask=torch.ones(b, audio_len, dtype=torch.bool),
        label=torch.randint(0, 7, (b,), generator=g),
        meta=[{"utt_id": f"dev/dia0_utt{i}"} for i in range(b)],
    )


# ---------------------------------------------------------------------------
# IG: shapes, non-zero (requires_grad_ trap), completeness
# ---------------------------------------------------------------------------


def test_attribute_text_shape_finite_and_keys() -> None:
    model = _fusion_model()
    batch = _batch(b=3, text_len=9)

    out = attribute_text(model, batch)

    assert set(out) >= {"attr", "delta", "target", "probs"}
    assert out["attr"].shape == (3, 9)
    assert out["attr"].dtype == np.float32
    assert np.isfinite(out["attr"]).all()
    assert out["delta"].shape == (3,) and np.isfinite(out["delta"]).all()
    assert out["probs"].shape == (3, 7)
    assert out["target"].shape == (3,)
    assert (out["n_steps"] >= 32).all()  # 32, or 64 if re-run for poor convergence


def test_attribute_text_is_not_all_zero() -> None:
    """Guards the requires_grad_ trap: frozen embedding params must not zero the attributions."""
    out = attribute_text(_fusion_model(), _batch())
    per_example_mass = np.abs(out["attr"]).sum(axis=1)
    assert (per_example_mass > 1e-8).all(), per_example_mass


def test_attribute_audio_shape_finite_and_not_all_zero() -> None:
    model = _fusion_model()
    batch = _batch(b=2, audio_len=4000)
    n_frames = model.audio_encoder(batch.audio_values, batch.audio_mask)[0].shape[1]

    out = attribute_audio(model, batch)

    assert out["attr"].shape == (2, n_frames)
    assert np.isfinite(out["attr"]).all()
    assert (np.abs(out["attr"]).sum(axis=1) > 1e-8).all()


def test_default_target_is_predicted_class_and_override_is_honoured() -> None:
    model = _fusion_model()
    batch = _batch()
    with torch.no_grad():
        predicted = model(batch).logits.argmax(dim=-1).numpy()

    default = attribute_text(model, batch)
    assert default["target"].tolist() == predicted.tolist()
    np.testing.assert_allclose(default["probs"].argmax(axis=1), predicted)

    other = (predicted + 1) % 7
    overridden = attribute_text(model, batch, target=other)
    assert overridden["target"].tolist() == other.tolist()


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_approximate_completeness(modality: str) -> None:
    """sum(attr) ~= F(x) - F(baseline) for the target logit, within 20% (32-step Riemann approximation)."""
    model = _fusion_model()
    batch = _batch(b=2, seed=3)
    audio_hidden, frame_mask = model.audio_encoder(batch.audio_values, batch.audio_mask)
    with torch.no_grad():
        text_embeds = model.embed_text_ids(batch.text_ids)
        pad_embeds = model.embed_text_ids(torch.full_like(batch.text_ids, PAD_TOKEN_ID))

    def f(te, ah):
        return model.forward_from_embeds(te, ah, batch.text_mask, frame_mask).logits

    if modality == "text":
        out = attribute_text(model, batch, n_steps=32)
        with torch.no_grad():
            f_x, f_base = f(text_embeds, audio_hidden), f(pad_embeds, audio_hidden)
    else:
        out = attribute_audio(model, batch, n_steps=32)
        with torch.no_grad():
            f_x, f_base = f(text_embeds, audio_hidden), f(text_embeds, torch.zeros_like(audio_hidden))

    idx = torch.as_tensor(out["target"])
    diff = (f_x.gather(1, idx[:, None]) - f_base.gather(1, idx[:, None])).squeeze(1).numpy()
    total = out["attr"].sum(axis=1)

    assert (np.abs(diff) > 1e-4).all(), f"degenerate stub: F(x)-F(baseline)={diff}"
    np.testing.assert_array_less(np.abs(total - diff), 0.2 * np.abs(diff))


def test_poorly_converged_examples_are_rerun_at_64_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ig, "DELTA_REL_THRESHOLD", -1.0)  # every example "fails" the delta rule
    out = attribute_text(_fusion_model(), _batch(), n_steps=8)
    assert out["n_steps"].tolist() == [64, 64]


def test_attribute_audio_cached_batch_matches_raw_batch() -> None:
    model = _fusion_model()
    batch = _batch(b=2, seed=5)
    with torch.no_grad():
        hidden, frame_mask = model.audio_encoder(batch.audio_values, batch.audio_mask)
    cached = CachedBatch(
        text_ids=batch.text_ids,
        text_mask=batch.text_mask,
        audio_hidden=hidden,
        frame_mask=frame_mask,
        label=batch.label,
        meta=batch.meta,
    )
    raw = attribute_audio(model, batch)
    via_cache = attribute_audio(model, cached)
    np.testing.assert_allclose(raw["attr"], via_cache["attr"], atol=1e-5)


def test_text_only_model_text_attribution_matches_its_own_forward() -> None:
    model = _text_only_model()
    batch = _batch()

    out = attribute_text(model, batch)

    with torch.no_grad():
        expected = torch.softmax(model(batch).logits, dim=-1).numpy()
    np.testing.assert_allclose(out["probs"], expected, atol=1e-5)
    assert out["attr"].shape == (2, 8)
    assert (np.abs(out["attr"]).sum(axis=1) > 1e-8).all()
    with pytest.raises(NotImplementedError):
        attribute_audio(model, batch)


# ---------------------------------------------------------------------------
# cur_span / frames_to_times / save_attribution
# ---------------------------------------------------------------------------


def test_cur_span_finds_current_utterance_tokens() -> None:
    cur, end = 50, 51
    ids = [0, 7, 8, 2, 9, cur, 11, 12, 13, end, 2, 1, 1]
    assert cur_span(ids, cur, end) == (6, 9)
    assert ids[6:9] == [11, 12, 13]
    assert cur_span(torch.tensor(ids), cur, end) == (6, 9)


def test_cur_span_raises_on_missing_or_duplicate_markers() -> None:
    cur, end = 50, 51
    with pytest.raises(ValueError, match="<cur>"):
        cur_span([0, 5, 6, end, 2], cur, end)
    with pytest.raises(ValueError, match="</cur>"):
        cur_span([0, cur, 5, 6, 2], cur, end)  # e.g. truncated away by max_text_len
    with pytest.raises(ValueError, match="<cur>"):
        cur_span([0, cur, 5, cur, 6, end, 2], cur, end)
    with pytest.raises(ValueError, match="</cur>"):
        cur_span([0, cur, 5, end, 6, end, 2], cur, end)


def test_frames_to_times_monotonic_from_zero() -> None:
    t = frames_to_times(100)
    assert t.shape == (100,)
    assert t[0] == pytest.approx(0.0)
    assert (np.diff(t) > 0).all()
    assert t[-1] == pytest.approx(99 / 49.9)


def _record(gate=(0.3, 0.7)) -> dict:
    return {
        "text_attr": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "text_tokens": ["<s>", "Ġhello", "</s>"],
        "cur_span": (1, 2),
        "audio_attr": np.array([0.5, -0.5, 0.25, 0.0], dtype=np.float32),
        "frame_times": frames_to_times(4),
        "pred": 3,
        "label": 1,
        "probs": np.full(7, 1 / 7, dtype=np.float32),
        "gate": gate,
        "convergence_delta": (0.01, 0.02),
    }


def test_save_attribution_round_trips_every_documented_key(tmp_path) -> None:
    rec = _record()
    path = attribution_path(tmp_path / "ig", "dev/dia0_utt3")
    assert path.name == "dev_dia0_utt3.npz"
    save_attribution(path, rec)

    with np.load(path) as z:  # no allow_pickle: all arrays must be plain dtypes
        assert set(z.files) == {
            "text_attr", "text_tokens", "cur_span", "audio_attr", "frame_times",
            "pred", "label", "probs", "gate", "convergence_delta",
        }  # fmt: skip
        np.testing.assert_allclose(z["text_attr"], rec["text_attr"])
        assert z["text_tokens"].tolist() == ["<s>", "Ġhello", "</s>"]
        assert z["cur_span"].tolist() == [1, 2]
        np.testing.assert_allclose(z["audio_attr"], rec["audio_attr"])
        np.testing.assert_allclose(z["frame_times"], rec["frame_times"])
        assert int(z["pred"]) == 3 and int(z["label"]) == 1
        np.testing.assert_allclose(z["probs"], rec["probs"])
        np.testing.assert_allclose(z["gate"], [0.3, 0.7], atol=1e-6)
        np.testing.assert_allclose(z["convergence_delta"], [0.01, 0.02])


def test_save_attribution_gate_none_is_empty_and_missing_key_raises(tmp_path) -> None:
    save_attribution(tmp_path / "a.npz", _record(gate=None))
    with np.load(tmp_path / "a.npz") as z:
        assert z["gate"].shape == (0,)

    bad = _record()
    del bad["probs"]
    with pytest.raises(KeyError, match="probs"):
        save_attribution(tmp_path / "b.npz", bad)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


class _Items(Dataset):
    def __init__(self, n: int) -> None:
        g = torch.Generator().manual_seed(0)
        self.items = [
            {
                "input_ids": torch.randint(2, TEXT_VOCAB, (6,), generator=g),
                "audio": torch.randn(3000, generator=g),
                "label": i % 7,
                "meta": {"utt_id": f"dev/dia{i}_utt0"},
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def _loader(n: int = 5) -> DataLoader:
    return DataLoader(_Items(n), batch_size=2, shuffle=False, collate_fn=lambda b: collate(b, PAD_TOKEN_ID))


def test_collect_gates_one_row_per_example_and_gate_sums_to_one() -> None:
    df = collect_gates(_fusion_model(), _loader(5))

    assert list(df.columns) == ["utt_id", "label", "pred", "g_text", "g_audio", "correct"]
    assert len(df) == 5
    assert df["utt_id"].tolist() == [f"dev/dia{i}_utt0" for i in range(5)]
    assert df["label"].tolist() == [i % 7 for i in range(5)]
    np.testing.assert_allclose(df["g_text"] + df["g_audio"], 1.0, atol=1e-6)
    assert (df["correct"] == (df["label"] == df["pred"])).all()


def test_collect_gates_unimodal_model_gives_nan_gate_columns_without_crashing() -> None:
    df = collect_gates(_text_only_model(), _loader(3))

    assert len(df) == 3
    assert df["g_text"].isna().all() and df["g_audio"].isna().all()
    assert df["pred"].between(0, 6).all()
    assert gate_summary(df) == {"n": 3, "has_gate": False}


def _hand_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utt_id": [f"dev/dia{i}_utt0" for i in range(6)],
            "label": [0, 0, 1, 1, 2, 2],
            "pred": [0, 1, 1, 1, 0, 2],
            "g_audio": [0.9, 0.7, 0.2, 0.4, 0.5, 0.1],
        }
    ).assign(g_text=lambda d: 1 - d["g_audio"], correct=lambda d: d["label"] == d["pred"])


def test_gate_summary_matches_hand_computed_values() -> None:
    s = gate_summary(_hand_frame())

    assert s["n"] == 6 and s["has_gate"] is True
    assert s["overall"]["mean"] == pytest.approx((0.9 + 0.7 + 0.2 + 0.4 + 0.5 + 0.1) / 6)
    assert s["overall"]["median"] == pytest.approx(0.45)
    assert s["per_class"]["neutral"] == {"mean": pytest.approx(0.8), "std": pytest.approx(0.1), "n": 2}
    assert s["per_class"]["joy"]["mean"] == pytest.approx(0.3)
    assert s["per_class"]["sadness"]["mean"] == pytest.approx(0.3)
    assert s["per_class"]["anger"]["n"] == 0 and np.isnan(s["per_class"]["anger"]["mean"])
    assert s["frac_audio_dominant"] == pytest.approx(2 / 6)  # 0.9, 0.7 (0.5 is not > 0.5)
    # correct rows: idx 0 (0.9), 2 (0.2), 3 (0.4), 5 (0.1); incorrect: idx 1 (0.7), 4 (0.5)
    assert s["by_correctness"]["correct"] == {"mean": pytest.approx(0.4), "n": 4}
    assert s["by_correctness"]["incorrect"] == {"mean": pytest.approx(0.6), "n": 2}
    assert s["top_audio_dominant"][:2] == ["dev/dia0_utt0", "dev/dia1_utt0"]
    assert s["top_text_dominant"][:2] == ["dev/dia5_utt0", "dev/dia2_utt0"]
    assert len(s["top_audio_dominant"]) == len(s["top_text_dominant"]) == 6  # fewer than 10 available


def test_gate_summary_reports_saturation() -> None:
    df = pd.DataFrame(
        {
            "utt_id": [f"u{i}" for i in range(4)],
            "label": [0, 0, 1, 1],
            "pred": [0, 0, 1, 1],
            "g_audio": [0.999, 0.99, 0.98, 0.3],
        }
    ).assign(g_text=lambda d: 1 - d["g_audio"], correct=True)
    s = gate_summary(df)
    assert s["frac_saturated_audio"] == pytest.approx(0.75)
    assert s["frac_saturated_text"] == 0.0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _assert_png(path) -> None:
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_token_heatmap_writes_png(tmp_path) -> None:
    tokens = ["<s>", "[Ross]:", "ĠHi", "Ġthere", "</s>", "<cur>", "Ġfine", "Ġ.", "</cur>"] * 3  # wraps rows
    attr = np.linspace(-1, 1, len(tokens))
    path = tmp_path / "figs" / "tokens.png"
    plot_token_heatmap(tokens, attr, (6, 8), "joy", "neutral", path)
    _assert_png(path)


def test_plot_audio_timeline_writes_png(tmp_path) -> None:
    times = frames_to_times(120)
    path = tmp_path / "audio.png"
    plot_audio_timeline(np.sin(times), times, "anger", path)
    _assert_png(path)


def test_plot_gate_distribution_writes_png(tmp_path) -> None:
    path = tmp_path / "gate.png"
    plot_gate_distribution(_hand_frame(), path)
    _assert_png(path)
    with pytest.raises(ValueError):
        plot_gate_distribution(_hand_frame().assign(g_audio=np.nan), tmp_path / "none.png")


def test_plot_per_class_comparison_writes_png(tmp_path) -> None:
    names = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
    metrics = {
        "B1": {"per_class_f1": dict(zip(names, [0.7, 0.5, 0.3, 0.4, 0.4, 0.1, 0.1], strict=True))},
        "M": dict(zip(names, [0.7, 0.55, 0.3, 0.45, 0.4, 0.15, 0.1], strict=True)),
    }
    path = tmp_path / "f1.png"
    plot_per_class_comparison(metrics, path)
    _assert_png(path)
