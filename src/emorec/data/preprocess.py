"""MP4 -> 16 kHz mono WAV extraction, loudness normalization, and manifest writing.

CONTRACT: this module does NOT truncate audio. The <=10 s END-keeping
truncation (ED5) lives in dataset.py (CC-3), where its golden-fixture test is
specified (BACKLOG T-06). Storing full-length WAVs here keeps manifest
`duration` meaningful and keeps `data.max_audio_sec` an ablatable config knob
rather than something baked irreversibly into the cached audio. This
resolves the ambiguity between BLUEPRINT §8-9 and IMPLEMENTATION_GUIDE §3.

Per-utterance failures (missing media, ffmpeg errors, unreadable output)
NEVER abort a run — they are excluded with a typed reason code and logged.
Any other exception is a bug and is allowed to propagate.
"""

import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from subprocess import run as run_subprocess

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from omegaconf import DictConfig
from tqdm import tqdm

from emorec.data.labels import EMOTION_TO_ID
from emorec.data.media import map_utterances_to_media
from emorec.data.metadata import make_utt_id

logger = logging.getLogger(__name__)

MANIFEST_KEYS: tuple[str, ...] = (
    "utt_id",
    "dialogue_id",
    "turn_idx",
    "speaker",
    "text",
    "label",
    "wav_path",
    "duration",
)


class PreprocessError(Exception):
    """Raised when ffmpeg extraction fails; carries the exact command and stderr for debugging."""

    def __init__(self, command: str, stderr: str) -> None:
        self.command = command
        self.stderr = stderr
        super().__init__(f"ffmpeg failed. command={command!r} stderr={stderr!r}")


def extract_wav(mp4: Path, wav: Path, sample_rate: int) -> None:
    """Extract a mono `sample_rate` Hz PCM16 WAV from `mp4` via an ffmpeg subprocess.

    Runs `ffmpeg -y -v error -i <mp4> -vn -ac 1 -ar <sample_rate> -acodec
    pcm_s16le <wav>`, capturing stderr. Raises `PreprocessError` (carrying
    the exact command string and stderr text) if ffmpeg exits non-zero,
    can't be launched at all (e.g. not on PATH), or produces a missing/
    zero-byte output file. Overwrites `wav` if it already exists.
    """
    mp4 = Path(mp4)
    wav = Path(wav)
    wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(mp4),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(wav),
    ]
    command_str = " ".join(cmd)

    try:
        result = run_subprocess(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise PreprocessError(command_str, f"could not launch ffmpeg: {exc}") from exc

    if result.returncode != 0:
        raise PreprocessError(command_str, result.stderr)
    if not wav.exists() or wav.stat().st_size == 0:
        raise PreprocessError(command_str, result.stderr or "ffmpeg produced a missing/zero-byte output file")


@dataclass(frozen=True)
class NormResult:
    """Outcome of one `loudness_normalize` call."""

    reason: str  # "OK" | "SKIP_TOO_SHORT" | "SKIP_SILENT"
    measured_lufs: float | None
    gain_db: float | None
    clamped: bool
    peak_limited: bool


def loudness_normalize(
    wav: Path,
    target_lufs: float,
    clamp_db: float,
    min_duration_s: float = 0.4,
) -> NormResult:
    """Loudness-normalize `wav` in place to `target_lufs`, gain-clamped to +/- `clamp_db`.

    Guards (each returns without writing, never raises):
      - duration < `min_duration_s` -> `SKIP_TOO_SHORT` (pyloudnorm raises on
        very short audio — TRIBAL_KNOWLEDGE.md §1.5 — this guard runs first).
      - measured integrated loudness is -inf/non-finite (silence) ->
        `SKIP_SILENT`.
    Otherwise `gain_db = target_lufs - measured`, clamped to
    `[-clamp_db, +clamp_db]` (BACKLOG J-02; `clamped` is reported so callers
    can count it). After applying gain, if the resulting peak sample exceeds
    0.99 it is rescaled to peak 0.99 and `peak_limited` is reported.
    """
    wav = Path(wav)
    data, sr = sf.read(wav, dtype="float32", always_2d=False)
    duration = (len(data) / sr) if sr else 0.0

    if duration < min_duration_s:
        return NormResult("SKIP_TOO_SHORT", None, None, False, False)

    meter = pyln.Meter(sr)
    measured = float(meter.integrated_loudness(data))
    if not math.isfinite(measured):
        return NormResult("SKIP_SILENT", measured, None, False, False)

    gain_db = target_lufs - measured
    clamped = False
    if gain_db > clamp_db:
        gain_db = clamp_db
        clamped = True
    elif gain_db < -clamp_db:
        gain_db = -clamp_db
        clamped = True

    gain_linear = 10.0 ** (gain_db / 20.0)
    normalized = data * gain_linear

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    peak_limited = False
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
        peak_limited = True

    sf.write(wav, normalized, sr, subtype="PCM_16")
    return NormResult("OK", measured, gain_db, clamped, peak_limited)


def _process_one(task: dict) -> dict:
    """Worker body for `process_split`: extract+normalize one utterance, or report why not.

    Runs in a `ProcessPoolExecutor` worker (or the main process for
    `n_workers<=1`), so `task` is a plain picklable dict, never a DictConfig
    or Path-bearing dataclass. Returns `{"status": "ok", "row": ..., "norm":
    ...}` or `{"status": "excluded", "utt_id": ..., "reason": ..., "detail":
    ...}` — never raises for a per-utterance failure.
    """
    utt_id = task["utt_id"]
    mp4_path = Path(task["mp4_path"])
    wav_path = Path(task["wav_path"])

    try:
        already_done = wav_path.exists() and wav_path.stat().st_size > 0
        if task["force"] or not already_done:
            extract_wav(mp4_path, wav_path, task["sample_rate"])
            norm = loudness_normalize(
                wav_path, task["target_lufs"], task["clamp_db"], task["min_duration_s"]
            )
            norm_reason, clamped, peak_limited = norm.reason, norm.clamped, norm.peak_limited
        else:
            norm_reason, clamped, peak_limited = "RESUMED", False, False
    except PreprocessError as exc:
        return {"status": "excluded", "utt_id": utt_id, "reason": "FFMPEG_FAILED", "detail": str(exc)}

    try:
        data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as exc:
        return {"status": "excluded", "utt_id": utt_id, "reason": "UNREADABLE_AUDIO", "detail": str(exc)}

    if data.size == 0:
        return {
            "status": "excluded",
            "utt_id": utt_id,
            "reason": "ZERO_LENGTH_AUDIO",
            "detail": "0 samples after extraction",
        }

    duration = round(len(data) / sr, 3)
    row = {
        "utt_id": utt_id,
        "dialogue_id": task["dialogue_id"],
        "turn_idx": task["turn_idx"],
        "speaker": task["speaker"],
        "text": task["text"],
        "label": task["label"],
        "wav_path": task["wav_path_posix"],
        "duration": duration,
    }
    return {"status": "ok", "row": row, "norm": {"reason": norm_reason, "clamped": clamped, "peak_limited": peak_limited}}


@dataclass
class SplitResult:
    """Return value of `process_split`: rows ready for `write_manifest`, exclusions, and norm-guard counts."""

    rows: list[dict]
    exclusions: list[dict]
    norm_counts: dict[str, int]


def process_split(
    split: str,
    df: pd.DataFrame,
    video_dir: Path,
    cfg: DictConfig,
    limit: int | None = None,
    force: bool = False,
) -> SplitResult:
    """Extract+normalize WAVs and build manifest rows for every matched utterance in `df`.

    Media lookup is scoped to `video_dir` (that split's own directory) via
    `map_utterances_to_media`. If `limit` is set, only the first `limit`
    rows (sorted by `(Dialogue_ID, Utterance_ID)`) are considered at all —
    used for `--stage smoke`. Resume-safe: an utterance whose WAV already
    exists and is non-empty is not re-extracted/re-normalized unless
    `force=True` (its duration is still measured from the existing file so a
    manifest row is still produced). Runs across
    `cfg.data.n_workers` processes via `ProcessPoolExecutor`
    (`cfg.data.n_workers=None` -> `min(8, os.cpu_count())`); `n_workers<=1`
    runs sequentially in the current process instead of paying pool-startup
    cost. A per-utterance failure excludes that utterance with a typed
    reason code (`MEDIA_MISSING`, `FFMPEG_FAILED`, `ZERO_LENGTH_AUDIO`,
    `UNREADABLE_AUDIO`) and never aborts the rest of the split.
    """
    df_sorted = df.sort_values(["Dialogue_ID", "Utterance_ID"], kind="stable").reset_index(drop=True)
    if limit is not None:
        df_sorted = df_sorted.head(limit)

    matched, missing = map_utterances_to_media(df_sorted, split, video_dir)
    row_lookup = {(r.Dialogue_ID, r.Utterance_ID): r for r in df_sorted.itertuples()}

    wav_dir = Path(cfg.data.wav_dir) / split
    wav_dir.mkdir(parents=True, exist_ok=True)

    exclusions: list[dict] = [
        {"utt_id": make_utt_id(split, m.dialogue_id, m.utterance_id), "reason": "MEDIA_MISSING", "detail": ""}
        for m in missing
    ]

    tasks: list[dict] = []
    for m in matched:
        row = row_lookup[(m.dialogue_id, m.utterance_id)]
        utt_id = make_utt_id(split, m.dialogue_id, m.utterance_id)
        wav_path = wav_dir / f"dia{m.dialogue_id}_utt{m.utterance_id}.wav"
        # Never os.path.join: force forward slashes even if cfg.data.wav_dir
        # was overridden with a backslash-containing (e.g. Windows absolute)
        # path -- wav_path in the manifest must be POSIX-style on every OS.
        wav_dir_posix = str(cfg.data.wav_dir).replace("\\", "/")
        wav_path_posix = f"{wav_dir_posix}/{split}/dia{m.dialogue_id}_utt{m.utterance_id}.wav"
        tasks.append(
            {
                "utt_id": utt_id,
                "dialogue_id": int(m.dialogue_id),
                "turn_idx": int(m.utterance_id),
                "speaker": row.Speaker,
                "text": row.Utterance,
                "label": EMOTION_TO_ID[row.Emotion],
                "mp4_path": str(m.mp4),
                "wav_path": str(wav_path),
                "wav_path_posix": wav_path_posix,
                "sample_rate": int(cfg.data.sample_rate),
                "target_lufs": float(cfg.data.target_lufs),
                "clamp_db": float(cfg.data.lufs_clamp_db),
                "min_duration_s": float(cfg.data.min_lufs_duration_s),
                "force": force,
            }
        )

    n_workers = cfg.data.n_workers or min(8, os.cpu_count() or 1)
    if n_workers <= 1 or not tasks:
        results = [_process_one(t) for t in tqdm(tasks, desc=f"{split} (1 worker)")]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                tqdm(executor.map(_process_one, tasks), total=len(tasks), desc=f"{split} ({n_workers} workers)")
            )

    rows: list[dict] = []
    norm_counts = {"skipped_short": 0, "skipped_silent": 0, "clamped": 0, "peak_limited": 0}
    for res in results:
        if res["status"] == "ok":
            rows.append(res["row"])
            norm = res["norm"]
            if norm["reason"] == "SKIP_TOO_SHORT":
                norm_counts["skipped_short"] += 1
            elif norm["reason"] == "SKIP_SILENT":
                norm_counts["skipped_silent"] += 1
            if norm["clamped"]:
                norm_counts["clamped"] += 1
            if norm["peak_limited"]:
                norm_counts["peak_limited"] += 1
        else:
            exclusions.append({"utt_id": res["utt_id"], "reason": res["reason"], "detail": res.get("detail", "")})

    rows.sort(key=lambda r: (r["dialogue_id"], r["turn_idx"]))
    logger.info("%s: %d written, %d excluded", split, len(rows), len(exclusions))
    return SplitResult(rows=rows, exclusions=exclusions, norm_counts=norm_counts)


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write `rows` as deterministic JSONL: one object per line, fixed key order, sorted by (dialogue_id, turn_idx).

    Each row must contain exactly `MANIFEST_KEYS` (extra keys are dropped,
    missing keys raise `KeyError` — schema minimalism is a binding
    instruction, docs/PROMPTS.md M0-M1 block: no convenience fields "for
    later"). Uses `newline="\\n"` and `ensure_ascii=False` unconditionally so
    two runs on the same input produce byte-for-byte identical files on any
    platform, including Windows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows, key=lambda r: (r["dialogue_id"], r["turn_idx"]))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in ordered_rows:
            record = {key: row[key] for key in MANIFEST_KEYS}
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
