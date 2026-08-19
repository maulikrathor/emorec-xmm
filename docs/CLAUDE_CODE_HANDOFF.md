# CLAUDE_CODE_HANDOFF.md — session continuity document

**Why this document exists:** context compaction loses conversational history. This file
is written so a FRESH Claude Code session can pick up CC-6+ work without re-deriving
anything already decided. Read this before `docs/CLAUDE.md`'s session-start protocol —
this file is the "what happened so far," `CLAUDE.md` is the "how to work."

Written at the close of CC-5, immediately before context compaction.

---

## 1. Project summary and immediate objective

`emorec-xmm`: explainable multimodal (text + audio) emotion recognition on MELD.
A gated cross-attention fusion model over frozen RoBERTa-base (text) and
WavLM-base-plus (audio) encoders, evaluated against unimodal/concat baselines,
with a planned IG/attention/occlusion explainability layer and a Gemini
build-vs-buy benchmark later. Architecture is FROZEN
(`docs/ARCHITECTURE.md`, `docs/DECISIONS.md` ED1–ED12) — no deviation without
an ADR.

**Immediate objective (Review-1, a faculty review):** get ONE real trained
result on real MELD data — B1 (text-only), B2 (audio-only), and M (gated
fusion) — with honest dev metrics, a confusion matrix, and the trivial
all-neutral baseline reported alongside. CC-1 through CC-5 built the entire
pipeline and model code for this and are fully tested OFFLINE with stub data.
**No real MELD training run has happened yet.** That is the very next step
(§9), and it happens on Kaggle, not this Windows dev machine (§10).

---

## 2. Completed tickets CC-1 through CC-5

Each ticket = one focused implementation session with its own test suite,
green before merging. Commits are on `main`, no ticket branches.

### CC-1 — repo scaffold, utils, frozen label map
**Commit:** `cc1dac3` — `chore: scaffold src-layout package, CI, utils, frozen label map, MIT license`
(BACKLOG T-01 + T-02)
- src-layout package `emorec`, pinned deps, ruff + pytest CI (GitHub Actions).
- `emorec.utils.seed.seed_everything(seed)` — the ONLY sanctioned seeding entrypoint.
- `emorec.utils.config.load_config(path, overrides)` — OmegaConf load + dotlist merge + resolve.
  `run_dir(cfg)` creates `artifacts/runs/{name}-{UTC ts}/`, writes `config.yaml` + `git_sha.txt`.
- `emorec.utils.log.JsonlLogger` — one JSON object per line, `"ts"` injected, flushes on write.
- `emorec.data.labels.EMOTION_TO_ID` — **the frozen label map** (§3).
- `configs/base.yaml` — single source of default config truth (still being extended by every later ticket, ADD-only).
- Tests: `test_smoke.py`, `test_labels.py`, `test_utils.py`.

### CC-2 (+ CC-2b correction, same commit) — MELD data pipeline
**Commit:** `5b39aa7` — `feat(data): MELD metadata validation, media inventory, wav extraction, manifests`
(BACKLOG T-04 + media-inventory work; CC-2b was a same-day correction folded into this commit)
- `emorec.data.meld_paths` — portable MELD root resolution (`--meld-root` CLI >
  `MELD_ROOT` env > `cfg.data.meld_root`; never a hardcoded host path). Resolves
  the real mirror's asymmetric layout (train CSV nested, dev/test CSVs at root;
  ALL video dirs nested one level: `train/train_splits`, `dev/dev_splits_complete`,
  `test/output_repeated_splits_test`). `is_appledouble()` filters macOS
  AppleDouble (`._*.mp4`) resource-fork files out of every media count —
  discovered on the real Kaggle mirror (~2,197 of them in the test split).
- `emorec.data.metadata` — hard validations V1–V7 (row counts, label contract,
  id uniqueness, global utt_id namespacing) + report-only findings R1–R6
  (class balance, split-integrity duplicate detection via
  `(Season, Episode, StartTime, EndTime)`, episode overlap, speaker priors).
- `emorec.data.media` — split-scoped media↔utterance mapping; orphan media
  counted, NEVER labeled, NEVER deleted; per-split arithmetic reconciliation
  (`real_media == matched + orphan_media`, `labeled_rows == matched + missing_media`).
- `emorec.data.preprocess` — ffmpeg 16 kHz mono extraction, −23 LUFS
  normalization (short/silent guards, ±20 dB clamp), deterministic 8-key JSONL
  manifests (§3), resume-safe, typed exclusion reason codes
  (`MEDIA_MISSING`/`FFMPEG_FAILED`/`ZERO_LENGTH_AUDIO`/`UNREADABLE_AUDIO`).
  **Audio truncation is deliberately NOT here** — deferred to CC-3 per ED5.
- `emorec.data.text_clean` — Windows-1252/smart-quote encoding fixes ONLY;
  never strips punctuation/case (punctuation carries emotion signal).
- `scripts/inspect_meld.py` (metadata+media validation only, no audio written,
  `--print-paths` diagnostic mode) and `scripts/prepare_data.py` (full
  WAV+manifest build, `--stage smoke|full`).
- Tests: `test_meld_paths.py`, `test_metadata.py`, `test_media.py`,
  `test_preprocess.py` (ffmpeg-gated, skip cleanly if absent), `conftest.py`
  (`toy_meld_root` fixture mirroring the real mirror's layout + AppleDouble decoys).

### CC-3 — MeldDataset, context builder, collator
**Commit:** `58e546c` — `feat(data): MeldDataset, dialogue-context builder, dual-mask collator`
(BACKLOG T-06)
- `emorec.data.dataset.build_context(dialogue_rows, t, k)` — PURE function,
  golden-string tested (§3).
- `MeldDataset` — groups/sorts manifest rows by dialogue once at init, O(1)
  `__getitem__`; registers `<cur>`/`</cur>` as tokenizer special tokens, stores
  their ids for later XAI span-location use; `load_audio=False` path for
  text-only baselines (never reads WAVs).
- `load_audio()` — END-keeping truncation per ED5 (§3).
- `collate()` → `Batch` dataclass: text padded with `pad_token_id`,
  `text_mask` 1=real; audio zero-padded, `audio_mask` True=real, **at RAW
  SAMPLE resolution** (frame conversion is the encoder's job, CC-4).
- `make_dataloader()` — seeded `torch.Generator` + `worker_init_fn` for
  reproducible shuffling.
- `stratified_subset_indices(dataset, n, seed)` — class-proportional,
  largest-remainder rounding, seeded, reproducible pilot subsets.
- Tests: `test_dataset.py` (25 tests) — 5 golden context strings (T-CRIT-4),
  END-keeping truncation, sr-mismatch guard, collate mixed-length batches,
  stratified-subset proportions/determinism, seeded-DataLoader-ordering
  determinism. Real `roberta-base` tokenizer used if reachable, else an
  offline stub (tokenizer-dependent assertions self-skip).

### CC-4 — frozen encoders + audio feature cache
**Commit:** `be0d4c8` — `feat(model): frozen text and audio encoders with feature caching`
(BACKLOG T-07 + T-08)
- `emorec.models.encoders.TextEncoder` — frozen RoBERTa; ONLY the two new
  `<cur>`/`</cur>` embedding rows trainable, via a gradient hook (§4).
- `emorec.models.encoders.AudioEncoder` — frozen WavLM; frame mask derived
  ONLY via the installed transformers model's own
  `_get_feature_vector_attention_mask` (§4 — API name discovery).
  `sample_rate != 16000` raises at construction; zero-width audio raises a
  named error (never silently runs on the text-only sentinel).
- `cache_key(audio_ckpt)` / `cache_path(cache_root, utt_id)` /
  `save_cached_features` / `load_cached_features` — per-utterance, unbatched,
  fp16 on-disk cache (§4).
- `scripts/build_cache.py` — resume-safe, `stratified_subset_indices`-aware,
  AMP-on-cuda, slices to true frame length before saving, never caches text.
- Tests: `test_encoders.py` (17 tests) — INV3 freeze correctness (gradient
  hook proof), frame-mask correctness vs `_get_feat_extract_output_lengths`,
  T-CRIT-5 cache equivalence (fp16 tolerance), eval-mode invariance. Tiny
  random-config `RobertaModel`/`WavLMModel` stubs injected via `_model=`,
  never real checkpoints.

### CC-5 — metrics, losses, trainer, fusion model, baselines
**Commit:** `4283fa6` — `feat(training): metrics, losses, trainer, baselines and fusion`
(scope: BACKLOG T-09/T-11/T-14-equivalent, time-boxed before Review-1;
bootstrap CI/McNemar explicitly deferred, §8)
- `emorec.training.metrics` — `compute_metrics` (sklearn-backed, always
  `labels=list(range(7))` so absent classes report 0.0, never crash),
  `trivial_baseline_metrics` (all-neutral honesty anchor),
  `save_confusion_matrix` (matplotlib Agg backend set at import),
  `classification_report_text`. `# TODO(post-review): bootstrap_ci + mcnemar`
  left as a module comment, additive-only when added.
- `emorec.training.losses.build_loss(cfg, class_counts)` — weighted CE,
  `w_c = 1/sqrt(count_c)` mean-normalized to 1; TRAIN-split-only counts is the
  CALLER's responsibility (enforced by convention, not by the function).
- `emorec.training.trainer.fit()` / `evaluate()` — AdamW over
  `requires_grad=True` params only; cosine+warmup schedule computed on
  OPTIMIZER steps (`len(train_loader)//grad_accum * max_epochs`), never raw
  micro-batches; AMP autocast+GradScaler on cuda (order: `unscale_ → clip →
  step`); early stop on dev weighted-F1; writes `last.pt`/`best.pt`/
  `metrics.json` (JSONL, one line/epoch)/`run_meta.json` (split names, subset
  size, seed, model name, full resolved config, git SHA, epoch count, wall time).
- `emorec.models.fusion` — `Output` dataclass (§3), `AttnPool` (masked
  learned-query pooling), `CrossModalBlock` (bidirectional cross-attention,
  pre-LN, residual, masked BOTH directions — `key_padding_mask` polarity
  inverted explicitly, see §4), `MultimodalERC` (the full M model, plus
  `forward_from_embeds` built now for future XAI use).
- `emorec.models.baselines` — `TextOnly` (B1, never touches `AudioEncoder`),
  `AudioOnly` (B2, never touches `TextEncoder`), `Concat` (B3, pools the SAME
  post-projection 256-d streams M does, no cross-attention — clean ablation).
- `emorec.models.registry.build_model(cfg, tokenizer_len, new_token_ids)` —
  name→class dispatch (`text_only`/`audio_only`/`concat`/`xattn`).
- `emorec.data.cached_dataset.CachedAudioMeldDataset` /
  `CachedBatch` / `collate_cached` — reads CC-4's on-disk cache instead of
  raw WAVs; `train.use_cache: true` is the default training path.
- `scripts/train.py`, `scripts/evaluate.py` — full CLI entrypoints (§9 has
  exact invocations). `evaluate.py` has a HARD GUARD refusing to run on any
  manifest whose filename contains `"test"` without `--allow-test` (A6 discipline).
- `configs/text_only.yaml`, `configs/audio_only.yaml`, `configs/xattn.yaml` —
  minimal overrides of `base.yaml` (ED11).
- Tests: `test_metrics.py` (9), `test_fusion.py` (12, including T-CRIT-1/INV1
  mask invariance — proved to actually fail before being finalized, see §4 —
  and a 60-step tiny-subset overfit smoke test).

---

## 3. Frozen contracts — do not change casually

Changing any of these invalidates checkpoints, reported numbers, or silently
breaks a downstream ticket. A change requires an ADR (`docs/DECISIONS.md`), not
a code edit.

**Label map** (`src/emorec/data/labels.py`), written as a literal dict, never derived:
```python
EMOTION_TO_ID = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3,
                 "surprise": 4, "fear": 5, "disgust": 6}
```

**Manifest row schema** — exactly these 8 keys, this order, no others
(`MANIFEST_KEYS` in `src/emorec/data/preprocess.py`):
```
utt_id, dialogue_id, turn_idx, speaker, text, label, wav_path, duration
```
`utt_id` is split-namespaced (`"dev/dia0_utt0"`) because `dia0_utt0` exists in
every split independently.

**Dialogue-context string format** (`build_context` in
`src/emorec/data/dataset.py`) — the five golden strings (T-CRIT-4), byte-exact:
```
k=0, t=2: "<cur> [Ross]: I'm fine. </cur>"
k=1, t=2: "[Monica]: Hi there! </s> <cur> [Ross]: I'm fine. </cur>"
k=2, t=2: "[Chandler]: Hey. </s> [Monica]: Hi there! </s> <cur> [Ross]: I'm fine. </cur>"
k=2, t=0: "<cur> [Chandler]: Hey. </cur>"                          (dialogue start, t<k, no padding)
k=2, t=1: "[Chandler]: Hey. </s> <cur> [Monica]: Hi there! </cur>"
```

**ED5 — END-keeping audio truncation** (`load_audio` in `src/emorec/data/dataset.py`):
audio longer than `max_audio_sec * sample_rate` is truncated by keeping the
LAST `max_audio_sec` seconds (`audio[-n:]`) — removes from the START, because
utterance-final prosody carries emotion.

**`Output` dataclass** (`src/emorec/models/fusion.py`), returned by EVERY
model (baselines included):
```python
@dataclass
class Output:
    logits: Tensor            # (B, 7)
    gate: Tensor | None       # (B, 2) = (g_t, g_a); None for unimodal
    xattn: dict | None        # cross-attention maps; None unless requested
```

**Invariants (test-enforced):**
- **INV1** (T-CRIT-1, the critical one): logits invariant to padding amount,
  same example, different batch padding, atol 1e-5. Tested at FRAME-level
  padding (`forward_from_embeds` with precomputed `audio_hidden`) — NOT
  raw-waveform padding, see §4's GroupNorm discovery.
- **INV2**: gate components in `(0,1)`, each row sums to 1.0.
- **INV3**: frozen encoder params `requires_grad=False`; exactly the ~2.3M
  fusion params + 2 new embedding rows are trainable in Stage 1.

---

## 4. Important implementation decisions and why

**`TextEncoder`'s embedding gradient-hook freezing.** The embedding weight
Parameter stays `requires_grad=True` in its ENTIRETY (autograd needs a live
path to it), but a gradient hook zeroes every row except the two new
`<cur>`/`</cur>` ids before the optimizer ever sees the gradient:
```python
def _hook(grad):
    mask = torch.zeros_like(grad)
    mask[self.new_token_ids] = 1.0
    return grad * mask
embedding.weight.register_hook(_hook)
```
`n_trainable()` reports `len(new_token_ids) * hidden_size`, NOT
`embedding.weight.numel()` — the latter would misleadingly count the whole,
mostly-frozen-by-hook, embedding matrix.

**transformers 5.x API discovery — `_get_feat_extract_output_lengths`.** The
architecture docs/BACKLOG referenced `_get_feature_vector_output_lengths`;
that name does NOT exist on transformers 5.x (verified via
`inspect`/`dir(WavLMModel)` against the installed 5.15.x). The correct name on
this major version is `_get_feat_extract_output_lengths`. `AudioEncoder`
calls `_get_feature_vector_attention_mask(feature_vector_length,
attention_mask)` (the RAW-sample `attention_mask`, not pre-summed lengths —
it sums internally via `.cumsum(-1)`), which is confirmed still correctly
named/shaped on 5.x. **If transformers is ever upgraded again, re-verify both
names via `inspect`/`dir()` before touching `encoders.py`** — this API has
already moved once between major versions.

**Deterministic freeze test uses `(hidden**2).sum()`, not `hidden.sum()`.** A
plain sum through RoBERTa's LayerNorm output is (near-)zero-mean, and for some
random inits the resulting embedding-row gradient legitimately underflows to
exactly `0.0` in float32 — a false negative on a numerically correct
implementation, discovered empirically while writing CC-4's INV3 test.
Squaring avoids the cancellation. If a similar "the gradient is suspiciously
exactly zero" symptom appears elsewhere, check the loss choice before
assuming a masking bug.

**Per-utterance unbatched fp16 audio cache.** `save_cached_features`/
`load_cached_features` store ONE utterance per `.npz` file, already sliced to
its true (unpadded) frame length. This is what makes `CachedAudioMeldDataset`
+ `CachedBatch` genuinely padding-invariant at train time (§ next bullet) —
padding is re-applied fresh at collate time, at the frame level, where masking
is provably exact.

**Cache key = `sha1(audio_ckpt)[:8]`.** Changing `cfg.model.audio_ckpt`
changes the cache directory entirely (`{cache_dir}/{cache_key}/...`) rather
than risking silent reuse of stale features from a different encoder.

**Encoder `train()` overrides forcing HF encoders to stay in `eval()`.** Both
`TextEncoder.train()` and `AudioEncoder.train()` override `nn.Module.train()`
to call `self.model.eval()` unconditionally afterward. This means a parent
module's `model.train()` (called every epoch by `trainer.fit()`) can NEVER
accidentally re-enable encoder dropout — required both for the
frozen-feature-extractor convention (J-03) and because the audio cache is
only valid if dropout was off when it was built.

**GroupNorm / raw-audio padding sensitivity, discovered during CC-5's INV1
testing — READ THIS BEFORE TOUCHING AudioEncoder OR THE CACHE PIPELINE.**
The REAL `microsoft/wavlm-base-plus` checkpoint config (confirmed via its
actual `config.json`, not assumed) sets `"feat_extract_norm": "group"`.
`GroupNorm` in WavLM's first conv layer normalizes over the ENTIRE padded
time axis of the raw waveform — meaning frozen wav2vec2/WavLM-family CNN
feature extractors are NOT invariant to how much raw-sample padding
surrounds a clip in its batch, regardless of how correctly any downstream
attention mask is applied. This is a genuine third-party architectural
property, not a bug in this repo's code, and it means:
- Two cache-building runs with different batch composition/order could in
  principle produce very slightly different cached features for the same
  utterance (a real but second-order, one-time effect at cache-build time).
- Raw-audio (`use_cache=false`) training reintroduces this as a small extra
  source of batch-composition-dependent noise.
- **Once a feature is cached (sliced to true length, saved unbatched), it is
  fixed** — this is exactly why the cache-based path is the one true
  padding-invariant path (next bullet).

**Frame-level masking is the production invariant, not raw-sample masking.**
INV1 is tested via `forward_from_embeds` with a fixed, precomputed
`audio_hidden` padded to two very different frame counts — this is what
`CrossModalBlock`/`AttnPool` actually own and must guarantee, and it matches
`train.use_cache: true` (the default), where fusion ALWAYS consumes
precomputed `audio_hidden`/`frame_mask` from `CachedBatch`, never raw
waveform. Do not "fix" INV1 by trying to make raw-waveform batching bit-exact
— that fight is with WavLM's own GroupNorm, not winnable here, and not
necessary given the cache path.

---

## 5. Current repository file tree

```
src/emorec/
  __init__.py
  data/
    __init__.py            # exports MeldDataset, build_context, collate, Batch
    labels.py               # FROZEN label map (§3)
    meld_paths.py            # MELD root/layout resolution, AppleDouble filtering
    metadata.py               # CSV loading, V1-V7 hard validations, R1-R6 reports
    media.py                   # utterance<->media mapping, orphan/inventory reporting
    text_clean.py                # encoding-artefact cleanup only
    preprocess.py                 # ffmpeg extraction, LUFS norm, manifest writing
    dataset.py                     # MeldDataset, build_context, collate, DataLoader helpers
    cached_dataset.py                # CachedAudioMeldDataset, CachedBatch, collate_cached
  models/
    __init__.py             # exports encoders, baselines, fusion, registry
    encoders.py               # TextEncoder, AudioEncoder, audio feature cache fns
    fusion.py                   # Output, AttnPool, CrossModalBlock, MultimodalERC
    baselines.py                  # TextOnly (B1), AudioOnly (B2), Concat (B3)
    registry.py                     # build_model(cfg, tokenizer_len, new_token_ids)
  training/
    __init__.py              # exports losses, metrics, trainer
    losses.py                  # build_loss (weighted CE)
    metrics.py                   # compute_metrics, trivial_baseline_metrics, plotting
    trainer.py                     # fit(), evaluate()
  utils/
    __init__.py
    seed.py                   # seed_everything
    config.py                   # load_config, run_dir
    log.py                        # JsonlLogger
  xai/
    __init__.py               # EMPTY -- CC-6+ territory

scripts/
  inspect_meld.py             # metadata+media validation only, --print-paths
  prepare_data.py               # WAV extraction + manifest build (--stage smoke|full)
  build_cache.py                  # audio feature cache builder
  train.py                          # train B1/B2/B3/M
  evaluate.py                         # evaluate a checkpoint (hard test-split guard)

tests/
  conftest.py                 # toy_meld_root fixture + shared MELD-mirror test helpers
  test_smoke.py                 # import emorec
  test_labels.py                  # frozen label map contract
  test_utils.py                     # seed/config/log
  test_meld_paths.py                  # root/layout resolution incl. AppleDouble
  test_media.py                         # media mapping, orphan/inventory
  test_metadata.py                        # CSV validation V1-V7, R1-R6
  test_preprocess.py                        # ffmpeg extraction, LUFS norm (ffmpeg-gated)
  test_dataset.py                             # build_context golden strings, collate, ED5
  test_encoders.py                              # INV3, frame masking, T-CRIT-5 cache equiv.
  test_metrics.py                                 # compute_metrics, weighted-CE weight math
  test_fusion.py                                    # INV1, INV2, forward_from_embeds, overfit smoke

configs/
  base.yaml                   # single source of default truth, ADD-only across tickets
  text_only.yaml                # model.name: text_only (B1)
  audio_only.yaml                 # model.name: audio_only (B2)
  xattn.yaml                        # model.name: xattn (M) -- also base.yaml's own default
```

Not yet created (deliberately, later tickets): `configs/concat.yaml` (B3 has
code but no dedicated config file yet — use `model.name=concat` as a dotlist
override), anything under `src/emorec/xai/`, `src/emorec/bench/`,
`src/emorec/serve/`.

---

## 6. Current test status

- **117 tests total**, all green with network+ffmpeg available.
- **115 pass / 2 skip** with ffmpeg hidden from `PATH` (the 2 skips are both
  in `test_preprocess.py`, cleanly marked `skipif`, never fail).
- Per-file breakdown: `test_smoke`(1), `test_labels`(4), `test_utils`(3),
  `test_meld_paths`(10), `test_media`(10), `test_metadata`(15),
  `test_preprocess`(11, 2 ffmpeg-gated), `test_dataset`(25),
  `test_encoders`(17), `test_metrics`(9), `test_fusion`(12).
- **No MELD data, no GPU, no real HF checkpoints required** for any of the
  above — `test_dataset.py` uses a real `roberta-base` tokenizer if network
  is reachable, else falls back to an offline stub (tokenizer-dependent
  assertions self-skip via `pytest.skip`); `test_encoders.py`/`test_fusion.py`
  use tiny random-config `RobertaModel`/`WavLMModel` stubs injected via each
  encoder's `_model=` constructor override, never `from_pretrained` on a real
  checkpoint.
- **Known benign warnings:** `UserWarning: Support for mismatched
  key_padding_mask and attn_mask is deprecated` from PyTorch's own attention
  internals, fired whenever WavLM's self-attention receives a real padding
  mask (both in `test_encoders.py` and `test_fusion.py`). Not from this
  repo's code; not an error; safe to ignore.
- Run commands: `pytest -q` (full suite), `ruff check src tests scripts`
  (must be clean). CI runs both on every push (`.github/workflows/ci.yml`).

---

## 7. Git state

- **Branch:** `main` (no ticket branches used; every CC-N ticket is a direct
  commit to `main`).
- **Latest CC-5 commit:** `4283fa6` — `feat(training): metrics, losses,
  trainer, baselines and fusion`.
- **Working tree:** clean immediately before this handoff commit.
- Remote: `origin` = `https://github.com/maulikrathor/emorec-xmm.git`.

---

## 8. Known limitations and deliberate deferrals

**Explicitly out of scope for CC-5, by the ticket's own instruction (not an
oversight):**
- **No real MELD training has been performed yet.** Every CC-5 test uses
  synthetic/stub data. This is the #1 next action (§9).
- **Bootstrap CI and McNemar** — `metrics.py` has a module-level
  `# TODO(post-review): bootstrap_ci + mcnemar per EVALUATION.md section 4
  (ED12) — additive, no refactor required.` The metrics module was
  deliberately designed so these can be added as new functions later.
- **3-seed aggregation** — headline runs eventually need seeds {13, 42, 2026}
  (`docs/ARCHITECTURE.md`); not implemented, no seed-sweep script exists yet.
- **Full training set** — no run against the complete MELD train split has
  happened; only unit-test-scale synthetic data.
- **Test-split evaluation** — never touched, by design (A6 discipline,
  enforced by `scripts/evaluate.py`'s hard guard). Dev only, until a
  deliberate final run.
- **XAI (T-18+)** — `src/emorec/xai/__init__.py` is empty.
  `MultimodalERC.forward_from_embeds` and `embed_text_ids` exist and are
  tested (parity with `forward`) specifically so this doesn't need
  retrofitting, but no attribution method is implemented.
- **Gemini benchmark** — `src/emorec/bench/` doesn't exist yet.
- **ONNX export** — not started.
- **Gradio demo** — `src/emorec/serve/` doesn't exist yet.
- **`configs/concat.yaml`** — B3 (`Concat`) has full code + tests but no
  dedicated top-level config file (not in CC-5's file list); use
  `model.name=concat` as a dotlist override in the meantime.
- **LoRA / Stage 2** — conditional on an M4 gate decision per BACKLOG; not
  started, not blocking anything now.

---

## 9. Next work

**Immediate (CC-6, or whatever the next ticket is named): Kaggle validation
of the whole CC-1–CC-5 pipeline against REAL MELD data.** In order:
1. Resolve the real MELD root on Kaggle (`--meld-root` / `MELD_ROOT` env —
   see §10, do not hardcode a path in source).
2. `python scripts/inspect_meld.py --meld-root <path>` — metadata+media
   validation only, should complete in well under a minute.
3. `python scripts/prepare_data.py --stage full --split all --meld-root <path>`
   — WAV extraction + manifest build for train/dev/test. (Or confirm/reuse
   the `meld-wav16k` Kaggle dataset if it already contains prepared
   WAVs/manifests — see §10.)
4. `python scripts/build_cache.py --manifest artifacts/manifests/train.jsonl
   --config configs/base.yaml` (repeat for dev, and test only if/when a final
   test-set run is deliberately scheduled) — builds the audio feature cache.
5. Train B1: `python scripts/train.py --config configs/text_only.yaml
   --train-manifest artifacts/manifests/train.jsonl --dev-manifest
   artifacts/manifests/dev.jsonl --seed 42 --device auto`.
6. Train B2: same with `--config configs/audio_only.yaml`.
7. Train M: same with `--config configs/xattn.yaml --use-cache`.
8. Train B3 if time allows: same with `--config configs/base.yaml
   model.name=concat` (dotlist override, no dedicated config file yet).
9. Collect dev metrics/confusion matrices/classification reports for all of
   the above; compare each against the trivial all-neutral baseline printed
   alongside every `train.py` run.
10. Only after all of the above: decide, with the user, what CC-6+ actually
    covers (bootstrap CI/McNemar? 3-seed runs? XAI? — see §8 for what's
    deferred and available to pick up next).

**Dependencies for anything past this list:** XAI (`forward_from_embeds`) and
the fusion model's gate values are already exposed and tested — later
attribution work can build directly on them without touching `fusion.py`'s
core forward path.

---

## 10. Environment facts

**Local Windows development environment (where CC-1–CC-5 were implemented):**
- OS: Windows, shell: Git Bash / PowerShell.
- No MELD data present locally. No GPU present locally (CPU-only).
- Python 3.11.4, torch 2.13.0+cpu, transformers 5.15.x, omegaconf 2.3.1,
  scikit-learn 1.9.0 (installed via `pip install -e ".[dev]"` into a fresh
  `.venv`, recreated and discarded every session — never committed).
- Network IS available in this environment (real `roberta-base` tokenizer
  downloads succeed; `AutoModel.from_pretrained` calls to REAL checkpoints
  were deliberately never exercised in tests — only stub configs).
- All 117 tests pass here without MELD data, without a GPU, and (for the
  ffmpeg-gated subset) without ffmpeg installed (clean skip).

**Kaggle environment (where real training must happen — not yet used this
session):**
- GPU: T4-class (per `docs/ARCHITECTURE.md`'s compute assumptions).
- MELD dataset location: **must be resolved via `--meld-root` CLI arg or
  `MELD_ROOT` environment variable at run time — never hardcode a Kaggle
  path in source** (`src/emorec/data/meld_paths.py`'s own module docstring:
  "no host- or platform-specific absolute path may appear here").
- A Kaggle dataset named **`meld-wav16k`** exists (or is intended) as a
  possible source of already-prepared 16 kHz WAVs — check whether it already
  satisfies `scripts/prepare_data.py`'s output contract
  (`artifacts/wav/{split}/dia{D}_utt{U}.wav` + `artifacts/manifests/{split}.jsonl`,
  the 8-key schema in §3) before re-running extraction from scratch.
- **`/kaggle/working` is EPHEMERAL** — it does not persist between sessions.
  Any Kaggle-side dataset (including `meld-wav16k`, and any output dataset
  used to persist `artifacts/cache/`, `artifacts/runs/`, trained
  checkpoints) is the actual persistence mechanism — copy anything
  irreplaceable (checkpoints the moment a run finishes, the audio cache) out
  of `/kaggle/working` into a Kaggle Dataset before the session ends, or it
  is gone.
- Exact Kaggle-side Python/torch/transformers versions were not re-verified
  this session (no Kaggle session was run) — re-check them at the start of
  CC-6 with `pip show torch transformers` before assuming version parity
  with the local dev environment above. The transformers API-name discovery
  in §4 (`_get_feat_extract_output_lengths`) was verified against
  transformers 5.15.x locally; re-verify if Kaggle resolves a different 5.x
  point release, though the major-version behavior should hold.
