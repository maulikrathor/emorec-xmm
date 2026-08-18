"""Tests for emorec.data.preprocess: ffmpeg extraction, loudness norm, and manifest writing.

Every test that touches ffmpeg is marked skipif-no-ffmpeg. No test here
asserts truncation — the <=10 s END-keeping truncation is CC-3's contract
(dataset.py), not preprocess.py's.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from emorec.data.labels import EMOTION_TO_ID, ID_TO_EMOTION
from emorec.data.meld_paths import resolve_meld_layout
from emorec.data.metadata import load_split_csv
from emorec.data.preprocess import (
    MANIFEST_KEYS,
    NormResult,
    extract_wav,
    loudness_normalize,
    process_split,
    write_manifest,
)

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


# ---------------------------------------------------------------------------
# extract_wav
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_extract_wav_produces_16k_mono_wav_of_expected_duration(tiny_mp4: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.wav"
    extract_wav(tiny_mp4, out, sample_rate=16000)

    info = sf.info(out)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert abs(info.duration - 1.5) <= 0.05


# ---------------------------------------------------------------------------
# loudness_normalize
# ---------------------------------------------------------------------------


def _write_sine_wav(path: Path, duration_s: float, sr: int = 16000, amplitude: float = 0.05) -> None:
    t = np.arange(int(duration_s * sr)) / sr
    data = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, data, sr, subtype="PCM_16")


def test_loudness_normalize_brings_measured_lufs_near_target(tmp_path: Path) -> None:
    wav = tmp_path / "quiet.wav"
    _write_sine_wav(wav, duration_s=2.0, amplitude=0.02)

    result = loudness_normalize(wav, target_lufs=-23.0, clamp_db=20.0, min_duration_s=0.4)

    assert result.reason == "OK"
    data, sr = sf.read(wav)
    measured_after = pyln.Meter(sr).integrated_loudness(data)
    assert abs(measured_after - (-23.0)) <= 1.0


def test_short_clip_is_skipped_and_not_modified(tmp_path: Path) -> None:
    wav = tmp_path / "short.wav"
    _write_sine_wav(wav, duration_s=0.2, amplitude=0.05)
    before = wav.read_bytes()

    result = loudness_normalize(wav, target_lufs=-23.0, clamp_db=20.0, min_duration_s=0.4)

    assert result.reason == "SKIP_TOO_SHORT"
    assert wav.read_bytes() == before


def test_silent_clip_is_skipped_and_does_not_raise(tmp_path: Path) -> None:
    wav = tmp_path / "silent.wav"
    data = np.zeros(16000 * 2, dtype=np.float32)
    sf.write(wav, data, 16000, subtype="PCM_16")

    result = loudness_normalize(wav, target_lufs=-23.0, clamp_db=20.0, min_duration_s=0.4)

    assert result.reason == "SKIP_SILENT"


def test_gain_exceeding_clamp_is_clamped_and_counted(tmp_path: Path) -> None:
    wav = tmp_path / "very_quiet.wav"
    # Amplitude quiet enough that target_lufs - measured_lufs >> clamp_db,
    # but not so quiet pyloudnorm's silence gating reports -inf (that path
    # is covered separately by test_silent_clip_is_skipped_and_does_not_raise).
    _write_sine_wav(wav, duration_s=2.0, amplitude=0.001)

    result = loudness_normalize(wav, target_lufs=-23.0, clamp_db=5.0, min_duration_s=0.4)

    assert result.reason == "OK"
    assert result.clamped is True
    assert abs(result.gain_db) == pytest.approx(5.0)


def test_norm_result_is_a_plain_dataclass() -> None:
    r = NormResult("OK", -23.0, 0.0, False, False)
    assert r.reason == "OK"


# ---------------------------------------------------------------------------
# write_manifest: determinism, schema, forward slashes
# ---------------------------------------------------------------------------


def _sample_row(**overrides) -> dict:
    row = {
        "utt_id": "train/dia0_utt0",
        "dialogue_id": 0,
        "turn_idx": 0,
        "speaker": "Chandler",
        "text": "also I was the point person",
        "label": EMOTION_TO_ID["neutral"],
        "wav_path": "artifacts/wav/train/dia0_utt0.wav",
        "duration": 3.21,
    }
    row.update(overrides)
    return row


def test_write_manifest_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    rows = [_sample_row(dialogue_id=1, turn_idx=0, utt_id="train/dia1_utt0"), _sample_row()]

    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    write_manifest(rows, path_a)
    write_manifest(list(reversed(rows)), path_b)

    assert path_a.read_bytes() == path_b.read_bytes()


def test_manifest_row_has_exactly_eight_keys_in_order(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    write_manifest([_sample_row()], path)

    line = path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(line)
    assert list(record.keys()) == list(MANIFEST_KEYS)
    assert list(MANIFEST_KEYS) == [
        "utt_id", "dialogue_id", "turn_idx", "speaker", "text", "label", "wav_path", "duration",
    ]


def test_manifest_wav_path_uses_forward_slashes(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    write_manifest([_sample_row()], path)

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "\\" not in record["wav_path"]
    assert record["wav_path"] == "artifacts/wav/train/dia0_utt0.wav"


def test_manifest_label_roundtrips_to_original_emotion_string() -> None:
    row = _sample_row(label=EMOTION_TO_ID["fear"])
    assert ID_TO_EMOTION[row["label"]] == "fear"


# ---------------------------------------------------------------------------
# process_split end-to-end (n_workers=1, sequential path -- fast & CI-safe)
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_process_split_end_to_end_resume_and_exclusions(toy_meld_root, tmp_path: Path) -> None:
    layout = resolve_meld_layout(toy_meld_root)
    df = load_split_csv(layout.splits["train"].csv)

    # Real media content is required for extract_wav to succeed; the fixture
    # only writes placeholder bytes, so replace the matched files with a
    # real tiny mp4 for exactly the utterances we're about to process.
    real_mp4_src = _make_real_mp4(tmp_path)
    for dia, utt in [(0, 0), (0, 1)]:
        target = layout.splits["train"].video_dir / f"dia{dia}_utt{utt}.mp4"
        shutil.copyfile(real_mp4_src, target)

    small_df = df[df["Dialogue_ID"] == 0].head(2)

    cfg = OmegaConf.create(
        {
            "data": {
                "wav_dir": str(tmp_path / "wav"),
                "sample_rate": 16000,
                "target_lufs": -23.0,
                "lufs_clamp_db": 20.0,
                "min_lufs_duration_s": 0.4,
                "n_workers": 1,
            }
        }
    )

    result = process_split("train", small_df, layout.splits["train"].video_dir, cfg, limit=None, force=False)
    assert len(result.rows) == 2
    assert result.exclusions == []

    wav_path = Path(tmp_path / "wav" / "train" / "dia0_utt0.wav")
    assert wav_path.exists()
    mtime_before = wav_path.stat().st_mtime_ns

    # Resume: re-running without force must not re-extract (mtime unchanged).
    result2 = process_split("train", small_df, layout.splits["train"].video_dir, cfg, limit=None, force=False)
    assert len(result2.rows) == 2
    assert wav_path.stat().st_mtime_ns == mtime_before


def _make_real_mp4(tmp_path: Path) -> Path:
    out = tmp_path / "_real_source.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
        "-f", "lavfi", "-i", "testsrc=duration=1.5:size=64x64:rate=10",
        "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out
