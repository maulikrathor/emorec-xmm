# BACKLOG.md — Complete implementation backlog

**Why this document exists:** the ticket-level work breakdown. Every ticket is the smallest independently testable unit of progress, in dependency-true order, with everything an engineer (human or Opus) needs to implement it without asking questions. Ticket IDs (T-##) are canonical and referenced by PROMPTS.md. Status is tracked by appending `[DONE <date> <git-sha>]` to a ticket's title line — never delete ticket content.

Conventions: **Difficulty** 1–5 (5 = hardest). **Time** = focused hours for a competent engineer with AI assistance; owner should expect ×1.3. Total ≈ 105–125 h ≈ the 12-week plan. Judgment calls are marked `⚖ J-##` and resolved in the Judgment Registry at the bottom.

---

## M0 — Skeleton

### T-01 Repo scaffold, packaging, CI
- **Objective:** installable src-layout package with lint+test CI.
- **Files:** `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `src/emorec/__init__.py`, empty subpackage `__init__.py`s, `tests/test_smoke.py`, `LICENSE`, `README.md` (v0 from docs bundle), `docs/*` (commit the entire docs bundle verbatim), `configs/base.yaml` (skeleton with commented sections).
- **Depends on:** nothing.
- **Details:** Python ≥3.11. Pinned deps; extras `[xai] [bench] [serve] [dev]` per IMPLEMENTATION_GUIDE §1. ruff config: line-length 100, target py311. CI: `pip install -e ".[dev]"`, `ruff check`, `pytest -q`; CPU; cache pip. Do NOT add torch to CI-critical smoke yet if install time hurts — torch-cpu wheel is fine and needed from T-02 onward, accept the CI minute cost.
- **Outputs:** green CI badge on first push.
- **Unit tests:** `test_smoke.py`: `import emorec` succeeds.
- **Integration tests:** CI run itself.
- **Common mistakes:** flat layout instead of src layout (breaks the no-scripts-import rule silently); forgetting `[tool.setuptools.package-dir]`/hatch equivalent so `pip install -e` picks up `src/`.
- **Debugging:** if `import emorec` fails post-install, inspect `pip show -f emorec` file list.
- **Done when:** clean clone → `pip install -e ".[dev]" && pytest -q && ruff check src tests` all green locally and in CI.
- **Difficulty:** 1 · **Time:** 2 h
- **Commit:** `chore: scaffold src-layout package, CI, docs bundle`
- **PR:** Repo skeleton: packaging, lint/test CI, frozen docs committed. No functional code.

### T-02 utils: seeding, config, logging
- **Objective:** the three primitives everything else uses.
- **Files:** `src/emorec/utils/{seed.py,config.py,log.py,__init__.py}`, `tests/test_utils.py`.
- **Depends on:** T-01.
- **Details:** `seed_everything(seed)` sets random/numpy/torch(+cuda) seeds, `torch.backends.cudnn.deterministic=True`, `benchmark=False`. `load_config(path, overrides: list[str])` = OmegaConf load + merge_with_dotlist + `OmegaConf.resolve`. `JsonlLogger(path)` appends dict rows with timestamp. `run_dir(cfg)` helper creates `artifacts/runs/{name}-{timestamp}` and dumps resolved config + `git rev-parse HEAD` into it.
- **Outputs:** importable utils.
- **Unit tests:** two seeded 3-layer-MLP forward+backward runs on random data produce bitwise-equal losses; dotlist override changes a nested key; JsonlLogger round-trips.
- **Integration tests:** none.
- **Common mistakes:** seeding after DataLoader workers spawn (also set `worker_init_fn` + generator later in T-06); forgetting cuda seeds.
- **Debugging:** nondeterminism → check cudnn flags and any `torch.use_deterministic_algorithms` warnings.
- **Done when:** tests green.
- **Difficulty:** 1 · **Time:** 2 h
- **Commit:** `feat(utils): seeding, omegaconf loader, jsonl logger, run dirs`
- **PR:** Deterministic seeding, config system (single-source base.yaml + dotlist overrides), structured run logging.

### T-03 MELD download + verification
- **Objective:** one idempotent command fetching MELD with integrity checks.
- **Files:** `src/emorec/data/download.py`, `scripts/prepare_data.py` (stage 1 of it), `tests/test_download.py`.
- **Depends on:** T-02.
- **Details:** download MELD.Raw (audio-visual tarballs) + CSVs to `artifacts/raw/`; sha256 recorded in a `checksums.json` committed to repo — **compute the checksums on first successful download and commit them** (upstream doesn't publish them; ⚖ J-01 covers mirror choice if the official link is dead). Resume-safe (skip if file exists AND checksum matches). Clear failure message pointing at `network settings / manual placement` fallback: user may drop files into `artifacts/raw/` manually and the script only verifies.
- **Outputs:** raw data on disk, verified.
- **Unit tests:** checksum mismatch on a tmp file raises with actionable message; existing-and-valid file is skipped (mock download fn).
- **Integration tests:** real download run once, checksums committed.
- **Common mistakes:** re-downloading gigabytes on every run (no skip logic); hard-failing when the user manually placed files.
- **Debugging:** if extraction fails, verify tar integrity before blaming code.
- **Done when:** second invocation is a fast no-op; bad file detected.
- **Difficulty:** 2 · **Time:** 3 h (network time excluded)
- **Commit:** `feat(data): idempotent MELD download with checksum verification`
- **PR:** MELD fetcher: resume-safe, checksum-verified, manual-placement fallback. Checksums committed from first verified download.

## M1 — Data pipeline

### T-04 Preprocess: audio extraction, normalization, manifests
- **Objective:** MP4s → clean 16 kHz WAVs + JSONL manifests.
- **Files:** `src/emorec/data/preprocess.py`, extend `scripts/prepare_data.py`, `tests/test_preprocess.py` (+ a 1 s fixture MP4 in `tests/fixtures/`).
- **Depends on:** T-03.
- **Details:** ffmpeg subprocess `-ac 1 -ar 16000 -vn`; capture stderr; failures → exclusion list. pyloudnorm to −23 LUFS with guards: skip norm if duration <0.4 s or measured loudness is -inf (silent); clamp gain to ±20 dB (⚖ J-02). Write `{split}.jsonl` rows `{utt_id, dialogue_id, turn_idx, speaker, text, label, wav_path, duration}`; label mapping FIXED as `{neutral:0, joy:1, sadness:2, anger:3, surprise:4, fear:5, disgust:6}` defined ONCE in `emorec/data/labels.py` and imported everywhere (never re-derive from sorted() — silent reorder = catastrophic). MELD CSVs have encoding quirks — read with `encoding='utf-8'` and fall back `latin-1`; strip stray `\x92`-style bytes. Exclusion list with reasons/counts → `artifacts/reports/excluded.json`.
- **Outputs:** `artifacts/wav/{split}/*.wav`, `artifacts/manifests/{split}.jsonl`, exclusion report.
- **Unit tests:** fixture MP4 → wav sr==16000, mono, duration±0.05 s; manifest row count == CSV rows − exclusions; label mapping frozen (literal dict equality test).
- **Integration tests:** full run over real MELD; spot-listen 3 random wavs (human check, once).
- **Common mistakes:** deriving label ids from `sorted(set(labels))` (breaks if any split lacks a class); losing dialogue/turn ordering (sort by (dialogue_id, turn_idx) explicitly); loudness-norm exceptions killing the whole run instead of excluding the clip.
- **Debugging:** ffmpeg issues → run the printed command manually; count mismatches → diff utt_id sets between CSV and manifest.
- **Done when:** all three manifests exist; exclusion counts logged per split; tests green.
- **Difficulty:** 3 · **Time:** 6 h (R4 budget: this is the ticket that historically overruns)
- **Commit:** `feat(data): mp4→16k wav extraction, loudness norm, jsonl manifests, frozen label map`
- **PR:** Full MELD preprocessing. Corrupt clips excluded with per-split counts (see artifacts/reports/excluded.json). Label mapping frozen in labels.py.

### T-05 Data audit
- **Objective:** falsify blueprint assumptions early, in writing.
- **Files:** `src/emorec/data/audit.py`, `scripts/audit.py`, `notebooks/01_data_audit.ipynb` (calls package only), `tests/test_audit.py`.
- **Depends on:** T-04 (+T-06 tokenizer for token-length histogram — run audit twice if needed, or make token histogram a flag; choose flag).
- **Details:** class distribution per split; duration histogram + % >10 s; k=2 token-length histogram with the REAL tokenizer incl. added special tokens (validates max_len 160 → else ADR per A3); missing/corrupt summary; per-speaker emotion distribution table (evidence for limitation #6). Output markdown to `artifacts/reports/data_audit.md`.
- **Outputs:** committed audit report (copy into `docs/reports/` — reports that gate decisions get committed, heavy artifacts don't).
- **Unit tests:** runs end-to-end on a 20-row toy manifest.
- **Integration tests:** real audit; human reads it.
- **Common mistakes:** measuring token lengths without the k=2 context assembled (understates); without special tokens counted.
- **Debugging:** n/a.
- **Done when:** report committed; max_len decision confirmed or ADR-0001 filed.
- **Difficulty:** 2 · **Time:** 3 h
- **Commit:** `feat(data): audit report — class balance, durations, token lengths, speaker priors`
- **PR:** Data audit gating M1 exit. Validates (or ADRs) max_len=160 and 10 s audio cap; documents speaker-prior evidence.

### T-06 Dataset, context builder, collator
- **Objective:** the exact tensor interface every model consumes.
- **Files:** `src/emorec/data/dataset.py`, `tests/test_dataset.py`.
- **Depends on:** T-04.
- **Details:** `build_context(dialogue_rows, t, k) -> str` pure, per ARCHITECTURE contract; handles t<k (fewer turns, no padding turns) and k=0 (`<cur> [spk]: u </cur>` only). Tokenizer: `roberta-base` + `add_special_tokens({'additional_special_tokens': ['<cur>','</cur>']})`; store ids on the dataset. RoBERTa quirk: pass pre-assembled full string to the tokenizer once (do NOT tokenize turns separately and concatenate ids — spacing/BPE boundaries shift). Audio: soundfile load, assert sr==16000, END-keeping truncation to 10 s, float32. Collator: tokenizer.pad for text; zero-pad + bool mask for audio; returns `Batch` dataclass incl. `meta`. DataLoader determinism: `generator` seeded + `worker_init_fn`.
- **Outputs:** `MeldDataset`, `collate`.
- **Unit tests:** T-CRIT-4 golden strings (k∈{0,1,2}, t<k case) — write expected strings BY HAND first; `<cur>`/`</cur>` ids present exactly once each in encoded ids; truncation keeps the END (construct 12 s ramp signal, assert kept samples are the tail); mask shapes; batch of mixed lengths collates.
- **Integration tests:** iterate 2 real batches from each split without error; wall-time a full epoch of pure data loading (baseline for later perf work).
- **Common mistakes:** off-by-one in `<cur>` span when later locating it (locate by token-id search at XAI time, don't store char offsets); tokenizing per-turn; forgetting `add_prefix_space` interactions — avoided by full-string tokenization; speaker names with colons breaking your own parsing (never parse the string back — keep structured fields).
- **Debugging:** print `tokenizer.convert_ids_to_tokens` for one sample and eyeball it once; that catches 90% of context bugs.
- **Done when:** all tests green incl. goldens.
- **Difficulty:** 3 · **Time:** 5 h
- **Commit:** `feat(data): MeldDataset with dialogue-context builder, special tokens, dual-mask collator`
- **PR:** Dataset layer. Context builder golden-tested (incl. dialogue-start, k=0); END-keeping audio truncation verified; deterministic loaders.

## M2 — Training infrastructure & baselines

### T-07 Encoder wrappers, freezing, special-token rows
- **Objective:** `TextEncoder`/`AudioEncoder` with correct freeze semantics.
- **Files:** `src/emorec/models/encoders.py`, `tests/test_encoders.py`.
- **Depends on:** T-06.
- **Details:** Text: `AutoModel.from_pretrained('roberta-base')`, `resize_token_embeddings(len(tokenizer))`; freeze all; make ONLY the 2 new embedding rows trainable via `register_hook` on embedding weight grad that zeroes all rows except the new ids (keep `requires_grad=True` on the embedding matrix, everything else False). Audio: `AutoModel` from cfg string; **GOTCHA (undocumented until now):** `facebook/wav2vec2-base-960h`'s feature extractor defaults `return_attention_mask=False` while `microsoft/wavlm-base-plus` uses True — always construct the mask yourself from raw lengths and pass it; get frame-level mask via `model._get_feature_vector_attention_mask(hidden.shape[1], audio_mask)`. Wrappers return `(hidden, frame_mask)`. Encoders run in `eval()` under `torch.no_grad()` when frozen — EXCEPT the text embedding layer path must retain grad for the two rows: implement frozen-text forward as no_grad for encoder body is WRONG here; instead rely on requires_grad=False params (autograd skips them) and keep the forward grad-enabled but cheap. Audio encoder may use true no_grad (nothing trainable inside it in Stage 1). Put encoders in eval mode always in Stage 1 (dropout off — deliberate, ⚖ J-03).
- **Outputs:** wrappers.
- **Unit tests:** T-CRIT-3 (trainable param names == expected set; count ≈2.3 M + hidden sizes); grad hook zeroes non-new rows (backward on toy loss, inspect embedding grad); frame_mask length matches hidden length for two different audio lengths.
- **Integration tests:** forward one real batch on CPU.
- **Common mistakes:** wrapping the whole text forward in no_grad (kills the special-token training silently — loss still decreases via fusion, bug invisible without the grad test); recomputing feature-extractor normalization inconsistently between cache and online paths.
- **Debugging:** `[(n,p.requires_grad) for n,p in model.named_parameters() if p.requires_grad]` is your first command for any freeze question.
- **Done when:** tests green; a printed trainable-params summary matches the blueprint budget.
- **Difficulty:** 4 · **Time:** 5 h
- **Commit:** `feat(models): frozen encoder wrappers, trainable special-token rows, frame masks`
- **PR:** Encoder layer with exact freeze semantics (grad-hook row masking), wav2vec2/wavlm mask-quirk handled, frame-mask derivation via HF helper.

### T-08 Audio feature cache
- **Objective:** 3–4× epoch speedup that makes 3-seed runs feasible.
- **Files:** cache logic inside `encoders.py` + `scripts/build_audio_cache.py`, `tests/test_cache.py`.
- **Depends on:** T-07.
- **Details:** iterate manifests, save fp16 `(T,768)` + frame mask → `artifacts/cache/{sha1(ckpt)[:8]}/{utt_id}.npz`. Dataset/collator gain `use_audio_cache` path returning cached hidden states instead of waveforms (Batch grows optional `audio_hidden` field; model consumes either). Build on GPU if available, else CPU overnight.
- **Outputs:** cache dir; `--use-cache` flag on training.
- **Unit tests:** T-CRIT-5 cached≡online (atol fp16-appropriate 1e-2 on hidden, exact on mask); checkpoint-string change → different cache dir.
- **Integration tests:** timed epoch with vs without cache; record speedup in ROADMAP log.
- **Common mistakes:** caching pre-normalization waveform features instead of encoder output; forgetting the mask in the npz; cache silently reused after switching encoder (the hash-keyed path prevents it — test it).
- **Debugging:** parity failures → check feature-extractor normalization applied identically in both paths.
- **Done when:** parity test green; measured speedup logged.
- **Difficulty:** 2 · **Time:** 4 h (+ one unattended cache-build run)
- **Commit:** `perf(models): checkpoint-hashed fp16 audio feature cache`
- **PR:** Disk cache of frozen audio encoder outputs; parity-tested vs online; ~Nx measured epoch speedup (number in PR).

### T-09 Losses, metrics, statistics
- **Objective:** all measurement code, pure and sklearn-verified.
- **Files:** `src/emorec/training/{losses.py,metrics.py}`, `tests/test_losses.py`, `tests/test_metrics.py`.
- **Depends on:** T-02.
- **Details:** weighted CE: weights = (1/√count_c) normalized to mean 1, computed from TRAIN split counts only (passed in, never recomputed from dev/test). Focal γ=2 with same weighting. Metrics: sklearn-backed wF1/mF1/per-class/confusion behind our own function signatures; `bootstrap_ci(y, ŷ, metric_fn, n=1000, seed)` resampling utterance indices; `mcnemar(pred_a, pred_b, y)` exact binomial on discordant pairs (b=A-right-B-wrong, c=inverse; p = 2·min tail of Binom(b+c, 0.5) at min(b,c), clamp ≤1).
- **Outputs:** measurement toolkit.
- **Unit tests:** weight math vs hand-computed 3-class toy; F1s equal sklearn on random data; CI on synthetic Bernoulli data brackets true accuracy ~95% over repeats (loose statistical test, seed-fixed); McNemar vs statsmodels on a known table.
- **Integration:** none.
- **Common mistakes:** computing class weights from the full dataset (leak); bootstrap resampling at the batch level instead of utterance level; McNemar on unaligned prediction orders (align by utt_id ALWAYS — sort both by meta id before comparing).
- **Debugging:** any stats disagreement → shrink to a 10-example hand-checkable case.
- **Done when:** tests green.
- **Difficulty:** 2 · **Time:** 4 h
- **Commit:** `feat(training): weighted CE/focal, metrics, bootstrap CI, exact McNemar`
- **PR:** All measurement primitives, sklearn/statsmodels-verified. Class weights train-split-only; comparisons utt_id-aligned by construction.

### T-10 Trainer
- **Objective:** the single training loop for every model in the repo.
- **Files:** `src/emorec/training/trainer.py`, `scripts/train.py`, `tests/test_trainer.py`.
- **Depends on:** T-07, T-09 (+T-08 optional path).
- **Details:** per ARCHITECTURE §loss/opt: AdamW(2e-4, wd .01) on trainable params only; cosine schedule with 10% warmup **computed over total optimizer steps (accounting for accumulation)**; AMP autocast+GradScaler; accum 4; clip 1.0; ≤15 epochs; early stop patience 3 on dev wF1; save best+last; JsonlLogger per-step loss/lr, per-epoch metric dict incl. mean gate per class (when model returns gates); resumable from last. Dev eval in eval() + no_grad, full dev each epoch.
- **Outputs:** `Trainer(cfg).fit()`; run dirs with config+SHA+seeds.
- **Unit tests:** tiny-subset (32 samples) overfit: train loss <0.1 within 50 steps using a small random-weights stub model config (no HF download in CI — real-encoder overfit run locally).
- **Integration tests:** local: 100-step run with real B1 config on GPU/CPU → this IS the A1 compute probe; record it/s and projected epoch time in ROADMAP log (M0 gate item, lands here by necessity).
- **Common mistakes:** scheduler stepped per micro-batch instead of per optimizer step (LR decays 4× too fast — the classic accumulation bug; assert `scheduler.last_epoch == optimizer_steps` in a test); early-stop comparing loss instead of wF1; scaler.step ordering with clipping (unscale_ → clip → step).
- **Debugging:** loss NaN → check AMP + loudness outliers; plateau at ~48 wF1 ≈ majority-class collapse → weights not applied.
- **Done when:** smoke test green; probe numbers logged.
- **Difficulty:** 3 · **Time:** 6 h
- **Commit:** `feat(training): full trainer — amp, accumulation-correct scheduling, early stop, resume`
- **PR:** Universal trainer. Includes the A1 compute probe results (it/s, projected epoch time) in the ROADMAP log.

### T-11 Baseline model classes B1/B2/B3
- **Objective:** unimodal + concat models sharing the encoder wrappers and Output contract.
- **Files:** `src/emorec/models/{baselines.py,registry.py}`, `configs/{text_only,audio_only,concat}.yaml`, `tests/test_baselines.py`.
- **Depends on:** T-07.
- **Details:** B1: text hidden → masked attention-pool → MLP(768→256→7). B2: audio analog. B3: both pooled (independent pools on RAW projected streams — proj to 256 first so B3 and M differ only by the cross-attention+gate stage; this makes E3 a clean ablation of exactly the fusion mechanism ⚖ J-04 documents the alternative). All return `Output(logits, gate=None, xattn=None)`. Registry dispatch on `cfg.model.name`.
- **Outputs:** three trainable configs.
- **Unit tests:** shapes; masked pooling ignores pad (pad-invariance test at baseline level too — cheap and catches pooling-mask bugs before fusion).
- **Common mistakes:** B3 pooling unprojected 768-d streams (breaks the clean-ablation property vs M).
- **Debugging:** n/a beyond tests.
- **Done when:** all three forward a real batch; tests green.
- **Difficulty:** 2 · **Time:** 3 h
- **Commit:** `feat(models): B1/B2/B3 baselines sharing encoders, projections, Output contract`
- **PR:** Baseline zoo constructed so M-vs-B3 isolates exactly the cross-attention+gate mechanism.

### T-12 Train B1 + GATE
- **Objective:** first real result; the pipeline-correctness gate.
- **Files:** none new (configs may get lr/epoch tweaks via dotlist logged in run dirs).
- **Depends on:** T-10, T-11.
- **Details:** train B1 1 seed. **GATE: dev wF1 ≥ 60.** If failed, debug in this order: (1) context builder output eyeballed via convert_ids_to_tokens, (2) label mapping (confusion matrix diagonal pattern reveals permutation bugs instantly — an off-diagonal band = permuted labels), (3) class weights actually reaching the loss, (4) lr (try 1e-4/5e-4). Do not touch fusion until green.
- **Outputs:** run dir + dev metrics; gate verdict in ROADMAP log.
- **Unit/Integration tests:** the gate itself.
- **Common mistakes:** "it's 58, close enough, moving on" — no. 58 vs 60 is exactly the size of a subtle context/weighting bug.
- **Debugging:** confusion matrix first, always.
- **Done when:** gate met and logged.
- **Difficulty:** 2 (mostly waiting) · **Time:** 3 h active + runs
- **Commit:** `exp(E1): B1 text baseline — gate passed at <wF1> dev`
- **PR:** B1 trained; M2 gate evidence attached (metrics json, confusion matrix png).

### T-13 Train B2, B3; 3-seed B1/B3; E1 tables
- **Objective:** complete baseline grid with statistics.
- **Files:** `scripts/evaluate.py` (test-set eval + CI + predictions-json dump), results table into `EXPERIMENTS.md` E1 row + README.
- **Depends on:** T-12.
- **Details:** B2 gate ≥38 dev wF1 (if B2 lands 33–38: run once with lr 5e-4 and 20 epochs before declaring failure — audio-only converges slowly; below 33 → suspect cache/mask bug). Then 3 seeds B1/B3 (B2 stays 1 seed, documented). `evaluate.py` writes `predictions.json` (utt_id→pred) per run — these files are the substrate for every later McNemar/faithfulness/dominance comparison; treat them as first-class artifacts.
- **Outputs:** E1 complete with mean±std and CIs.
- **Common mistakes:** evaluating test now for exploration — test predictions are generated ONCE per final model and cached; dev drives all decisions.
- **Done when:** E1 row done; README table baselines filled.
- **Difficulty:** 2 · **Time:** 4 h active + runs
- **Commit:** `exp(E1): full baseline grid, 3-seed B1/B3, bootstrap CIs`
- **PR:** Baseline results with statistics; predictions.json artifacts established as the comparison substrate.

## M3 — Fusion

### T-14 Fusion model + T-CRIT-1 + XAI hooks
- **Objective:** the centerpiece module, correct by construction.
- **Files:** `src/emorec/models/fusion.py`, `configs/xattn.yaml`, `tests/test_fusion.py`.
- **Depends on:** T-11.
- **Details:** per ARCHITECTURE diagram exactly. Pre-LN; `nn.MultiheadAttention(batch_first=True)` with `key_padding_mask` = ~bool_mask in BOTH directions; per-stream FFN + residuals. Attention-pool: learned query, masked softmax. Gate: `softmax(Linear(concat(z_t,z_a)) )` over 2 logits — initialize the linear bias to 0 (starts balanced ~0.5/0.5). Expose `forward_from_embeds(text_embeds, audio_hidden, text_mask, frame_mask)` NOW (T-18 depends on it) and return xattn maps in Output during no-grad passes (flag-gated to avoid memory cost in training). **Write T-CRIT-1 FIRST, run it red against a deliberately broken version (comment out one key_padding_mask) to prove the test has teeth, then green.**
- **Outputs:** `MultimodalERC`.
- **Unit tests:** T-CRIT-1 (both directions probed: vary text pad only, audio pad only, both); INV2 gate simplex; shapes; forward_from_embeds ≡ forward given same inputs.
- **Common mistakes:** key_padding_mask polarity (PyTorch: True = IGNORE — the #1 bug); masking queries but not keys; softmax over padded pool weights without −inf masking; xattn map retention during training exploding memory.
- **Debugging:** if INV1 fails only for audio padding → frame-mask derivation (T-07) is the suspect, not fusion.
- **Done when:** all tests green, including the proved-it-can-fail step (note it in the PR).
- **Difficulty:** 4 · **Time:** 7 h
- **Commit:** `feat(models): gated cross-attention fusion + mask-invariance proof + XAI hooks`
- **PR:** Fusion module. T-CRIT-1 demonstrated red-then-green against an intentionally broken mask. forward_from_embeds hook in place for Captum.

### T-15 Train M, E3 headline comparison
- **Objective:** the project's headline number, with honest statistics.
- **Files:** results into EXPERIMENTS E3 + README; `notebooks/02_results_analysis.ipynb` started.
- **Depends on:** T-13, T-14.
- **Details:** 3 seeds; test eval once per seed via evaluate.py; M-vs-B3: 3×3 seed-pairwise McNemar summary + CI overlap. Whatever the outcome, write the paragraph (significant win / null / loss) into the notebook and README honestly. Log mean gate per class — first look at dominance signal #3.
- **Done when:** E3 row + README updated; gate-per-class table saved.
- **Difficulty:** 2 · **Time:** 3 h active + runs
- **Commit:** `exp(E3): gated xattn vs concat — <result summary>, McNemar p=<p>`
- **PR:** Headline comparison with full statistics; per-class gate statistics logged (feeds M5 dominance).

### T-16 E2 context ablation (+E4 encoder swap if time)
- **Objective:** quantify dialogue-context value; optionally WavLM-vs-Wav2Vec2.
- **Depends on:** T-15. **Details:** k∈{0,1,2,4}, 1 seed each, dev-reported (these are ablations, not headlines; say so in the table). E4 = checkpoint swap + cache rebuild (hash handles it) + 1 seed.
- **Common mistakes:** comparing k-ablations on test (burns test-purity for a P1/P2 ablation).
- **Done when:** E2 (and maybe E4) rows filled. · **Difficulty:** 1 · **Time:** 2 h active + runs
- **Commit:** `exp(E2,E4): context-window and audio-encoder ablations`
- **PR:** Ablation results, dev-split, clearly labeled non-headline.

## M4 — Gate week

### T-17 LoRA Stage-2 (CONDITIONAL — M4 decision gate)
- **Objective:** recover fine-tuning gains if and only if schedule+compute allow.
- **Files:** `configs/xattn_lora.yaml`, peft integration in encoders.py, ADR-000X recording the go/no-go.
- **Depends on:** T-15 done ON SCHEDULE + compute probe ≥ T4/6 h sessions. Else: skip permanently via ADR; that path costs 30 minutes and is a fully valid outcome.
- **Details:** peft LoRA r=8 α=16 on q,v of top-4 layers each encoder; cache disabled (encoder now changes); lr 1e-4; expect 3–4× slower epochs; 1 seed first, extend to 3 only if the gain > 1 wF1.
- **Common mistakes:** leaving the audio cache on (silently trains against stale features — the hash key only guards checkpoint STRING, not adapter state; hard-disable cache in the lora config and assert in code).
- **Done when:** either E6 row filled or skip-ADR merged. · **Difficulty:** 3 · **Time:** 4 h active + runs
- **Commit:** `exp(E6): LoRA stage-2` or `docs(adr): skip LoRA stage-2 (schedule/compute)`
- **PR:** Conditional outcome per M4 gate, decision reasoning in ADR.

## M5 — XAI

### T-18 Integrated Gradients (both modalities)
- **Objective:** first attribution family, artifact schema established.
- **Files:** `src/emorec/xai/ig.py`, `scripts/run_xai.py`, `tests/test_ig.py`.
- **Depends on:** T-14 (+trained ckpt from T-15).
- **Details:** per IMPLEMENTATION_GUIDE §10 wrapper pattern via `forward_from_embeds`; one modality per pass. Text baseline: pad-token embedding row broadcast; audio baseline: zeros at post-proj (equivalently pre-proj zeros through linear ≠ zero — attribute AT post-proj features to keep baseline exact zeros; expose the needed hook). Target = predicted class (not gold — we explain the model's decision; gold-class attribution is a flag for case studies). n_steps 32; log convergence delta per sample; auto-retry 64 when |delta|>5% of output. Save npz per MODULES schema; locate `<cur>` span by token-id search.
- **Unit tests:** shapes; completeness (sum attr ≈ F(x)−F(baseline) within delta) on a tiny random model; cur-span located correctly on a crafted sample.
- **Integration:** run over dev subset (200) first; inspect 5 heatmap printouts by eye before the full test-set run (GPU-hours matter).
- **Common mistakes:** attributing gold class silently; internal batching in Captum interacting with your masks (pass masks via additional_forward_args, verify batch expansion); running the full test set before eyeballing anything.
- **Debugging:** all-near-zero text attributions → wrong layer (attributed frozen embedding without grad path; confirm requires_grad on the embeds tensor passed in, use `.requires_grad_()` on the input embeds inside the attribution wrapper).
- **Done when:** full test-set IG artifacts on disk for the best checkpoint; delta stats logged.
- **Difficulty:** 4 · **Time:** 7 h
- **Commit:** `feat(xai): layer-IG for text (cur-span aware) and audio (post-proj), npz artifacts`
- **PR:** IG pipeline with convergence-delta logging and the shared attribution schema; completeness-tested.

### T-19 Attention attribution
- **Objective:** second method family, intrinsically extracted.
- **Files:** `src/emorec/xai/attention.py`, `tests/test_attention.py`.
- **Depends on:** T-14, T-18 (schema).
- **Details:** no-grad pass with xattn flag; aggregate: mean over heads, then for text-token scores use audio→text maps' column mass + pool weights; ⚖ J-05 (aggregation choice) — document formula in the module docstring; same npz schema.
- **Done when:** test-set artifacts on disk. · **Difficulty:** 2 · **Time:** 3 h
- **Commit:** `feat(xai): cross-attention attribution extraction`
- **PR:** Attention-based scores under the shared schema; aggregation formula documented (J-05).

### T-20 Occlusion + stratified subset
- **Objective:** perturbation-family arbiter on a tractable subset.
- **Files:** `src/emorec/xai/occlusion.py`, `scripts/stratify_subset.py`, `tests/test_occlusion.py`.
- **Depends on:** T-14; ckpt.
- **Details:** subset: 500 test utts stratified by class (proportional, min 20/class, seeded — commit the id list to `docs/reports/occlusion_subset.json` so it's frozen forever). Text: sliding 3-token window within `<cur>` span, replace with pad embeds; audio: 1 s windows zeroed at post-proj. Score = Δlogit(pred). Batch the occluded variants (this is embarrassingly parallel — build a batch of all windows per utterance).
- **Common mistakes:** occluding context tokens too (dominance analysis needs cur-span-only text occlusion; context occlusion is a separate curiosity, skip); non-frozen subset (breaks agreement comparability across reruns).
- **Done when:** subset artifacts on disk. · **Difficulty:** 3 · **Time:** 5 h
- **Commit:** `feat(xai): windowed occlusion on frozen stratified subset`
- **PR:** Occlusion arbiter; subset id list committed for permanent comparability.

### T-21 Faithfulness evaluation + sanity checks
- **Objective:** the tables that make the XAI layer defensible.
- **Files:** `src/emorec/xai/faithfulness.py`, `scripts/run_faithfulness.py`, `tests/test_faithfulness.py`.
- **Depends on:** T-18..T-20.
- **Details:** per EVALUATION §5. Deletion/retention operate through forward_from_embeds with the same operators as IG baselines; compute per method × modality; AUC; Spearman agreement distributions on the occlusion subset; parameter-randomization sanity (randomize MLP head, recompute IG on 100 samples, assert rank-correlation with original attributions collapses — if it doesn't, that's a reported red flag, not a silent pass).
- **Common mistakes:** deleting by zeroing token IDS instead of embeddings (changes sequence semantics); comparing Spearman on different token universes (restrict all methods to cur-span text + valid frames).
- **Done when:** `artifacts/reports/faithfulness.md` tables generated + copied to docs/reports; E7 partially filled.
- **Difficulty:** 4 · **Time:** 6 h
- **Commit:** `feat(xai): faithfulness curves/AUC, method agreement, randomization sanity`
- **PR:** Quantitative XAI evaluation — the course-critical deliverable. Includes the sanity-check verdict either way.

### T-22 Modality dominance + prosody correlation
- **Objective:** the project's signature analysis.
- **Files:** `src/emorec/xai/dominance.py`, `tests/test_dominance.py`.
- **Depends on:** T-18, T-21 (+gate logs from T-15).
- **Details:** three signals per EVALUATION §6 (max-normalize per modality BEFORE mass ratios); ablation dominance via forward_from_embeds with baseline substitution; per-class aggregation with CIs; 3-way agreement; librosa pitch-variance (pyin or yin on voiced frames) + RMS energy per utterance, Spearman vs audio-dominance — the SLP tie-in.
- **Common mistakes:** comparing raw IG magnitudes across modalities (scale artifact — the normalization is the whole trick); pyin failures on silent/laugh clips (guard + exclude with count).
- **Done when:** dominance report + agreement table generated; E7 complete.
- **Difficulty:** 3 · **Time:** 5 h
- **Commit:** `feat(xai): 3-way modality dominance, agreement, prosody correlation`
- **PR:** Dominance quantified three independent ways with agreement stats; prosodic correlates reported (SLP deliverable).

### T-23 Viz + case studies
- **Objective:** the figures for report, README, and demo.
- **Files:** `src/emorec/xai/viz.py`, case-study figure script.
- **Depends on:** T-18..T-22.
- **Details:** token heatmap (matplotlib, Agg backend, cur-span outlined), audio timeline (0.5 s bins under waveform), gate dial; 6 case studies (2 text-dominant, 2 audio-helpful, 2 method-disagreement) selected by querying the dominance/agreement artifacts, not hand-picked vibes — write the selection query into the script.
- **Done when:** figures in docs/reports; M5 exit gate fully met. · **Difficulty:** 2 · **Time:** 4 h
- **Commit:** `feat(xai): visualization suite + criterion-selected case studies`
- **PR:** All M5 figures; case-study selection is scripted and reproducible.

## M6 — LLM benchmark

### T-24 Gemini runner (mock-first)
- **Objective:** money-safe benchmark infrastructure before any real call.
- **Files:** `src/emorec/bench/gemini_runner.py`, `src/emorec/bench/prompts/v1.txt`, `tests/test_gemini_runner.py`.
- **Depends on:** T-04 manifests.
- **Details:** per ED8 + ARCHITECTURE: injectable client; cache `artifacts/gemini/{prompt_hash}/{utt_id}.json` storing raw response + parsed label + token counts + latency; resume = skip cached; budget cap: projected-cost check before starting (test-set size × few-shot prompt token estimate × price constants in config) AND running-total abort; temp 0; JSON schema-constrained; ≤2 retries then invalid. Prompt v1: task description + 7 few-shot (one per class from train, ids recorded in the file header) + response schema. Two conditions flag: `--audio/--no-audio`.
- **Unit tests (mock only):** T-CRIT-6 cache round trip, resume skips, parse-failure→invalid, cap abort mid-run.
- **Done when:** all mock tests green; zero real API calls made yet. · **Difficulty:** 3 · **Time:** 5 h
- **Commit:** `feat(bench): cache-first budget-capped gemini runner (mock-tested)`
- **PR:** Benchmark infra fully tested against a mock client. Prompt v1 frozen with recorded few-shot ids.

### T-25 Gemini runs + comparison
- **Objective:** E10 executed and the build-vs-buy table produced.
- **Files:** `src/emorec/bench/compare.py`; results into EXPERIMENTS/README.
- **Depends on:** T-24, T-13/T-15 predictions.
- **Details:** dry-run cost projection first; if > cap → stratified 1K subsample (reuse stratify script, new seed, commit ids). Run audio+text, then text-only. compare.py merges predictions.json artifacts + gemini cache → tables (F1 family, invalid rate, $/1K, p50/p95) + break-even volume (⚖ J-06 for the amortization assumptions — document them in the table footnote). Snapshot the price sheet values used into the report.
- **Common mistakes:** rate-limit hammering (add simple sleep/backoff); comparing on mismatched subsets (Gemini subset must be evaluated against OUR model on the same ids — compare.py enforces id-set equality).
- **Done when:** E10 filled; README comparison rows filled. · **Difficulty:** 2 · **Time:** 3 h active + run time
- **Commit:** `exp(E10): gemini 2.5 flash benchmark, both conditions, break-even analysis`
- **PR:** Build-vs-buy results with cost/latency and dated pricing snapshot; id-set equality enforced.

## M7 — Product layer

### T-26 Serve pipeline
- **Objective:** headless end-to-end inference function.
- **Files:** `src/emorec/serve/pipeline.py`, `tests/test_pipeline.py`, bundled 2 s fixture wav.
- **Depends on:** T-14 ckpt, T-18 (for explain=True path), faster-whisper.
- **Details:** `predict(audio, text=None, explain=False)`; resample via torchaudio/soxr to 16 k mono; whisper small int8 CPU only when text is None; k=0 context build; returns dict {probs, label, gate, transcript, attributions?}. Lazy-load models once (module-level singleton with explicit `load()`).
- **Done when:** fixture test green on CPU. · **Difficulty:** 2 · **Time:** 4 h
- **Commit:** `feat(serve): end-to-end predict pipeline with optional on-demand IG`
- **PR:** Headless pipeline, CPU-tested; demo becomes a thin UI over this.

### T-27 Gradio demo
- **Objective:** the interactive artifact.
- **Files:** `src/emorec/serve/app.py`, `scripts/demo.py`.
- **Depends on:** T-26, T-23 viz.
- **Details:** per DEPLOYMENT §2. Two tabs (live input / MELD sample). Explain button separate from predict. Editable transcript re-predict. UI copy caveats per DEPLOYMENT. Record the demo GIF here.
- **Done when:** TESTING manual demo checklist passes; GIF committed. · **Difficulty:** 2 · **Time:** 5 h
- **Commit:** `feat(serve): gradio demo — live mic + meld samples + attribution views`
- **PR:** Working demo; manual checklist evidence + GIF in PR.

### T-28 ONNX export + latency bench
- **Objective:** deployable inference artifact + honest numbers.
- **Files:** `src/emorec/serve/export_onnx.py`, `scripts/bench_latency.py`, `tests/test_onnx.py`.
- **Depends on:** T-14.
- **Details:** fusion head only (ED9), dynamic axes, opset ≥17; T-CRIT-2 parity on random AND 20 real inputs; latency per EVALUATION §8.
- **Common mistakes:** exporting with a fixed sequence length baked in (test parity at 2 different lengths); trying encoder export (forbidden, ED9).
- **Done when:** parity green; latency table in README. · **Difficulty:** 3 · **Time:** 4 h
- **Commit:** `feat(serve): onnx fusion-head export + parity + latency table`
- **PR:** ONNX path with mandatory parity; per-stage latency table added to README.

### T-29 Docker + HF Hub + README completion
- **Objective:** release.
- **Files:** `Dockerfile`, hub upload script, final README/model-card numbers.
- **Depends on:** T-27, T-28, all results tickets.
- **Details:** per DEPLOYMENT §1/§4/§6 checklist, executed literally, including the clean-venv stranger-reproducibility run.
- **Done when:** DEPLOYMENT release checklist fully ticked. · **Difficulty:** 2 · **Time:** 5 h
- **Commit:** `release: v1.0 — docker, hub weights, reproducibility-verified README`
- **PR:** v1.0 release. Checklist evidence attached (docker run screenshot, clean-venv transcript).

## M8 — Analysis & report

### T-30 Error analysis notebook (E9)
- **Objective:** failure-mode characterization.
- **Files:** `notebooks/03_error_analysis.ipynb` (+small helpers in package if needed).
- **Depends on:** T-15, T-22.
- **Details:** top confusion pairs with examples; laugh-track study: sample 50 errors, annotate contamination by listening (hand labeling is fine — say so), compare contamination rate errors-vs-correct; minority-class deep dive (fear/disgust); tie observations back to KNOWN_LIMITATIONS with "confirmed(E9)" updates.
- **Done when:** notebook committed, limitations updated. · **Difficulty:** 2 · **Time:** 5 h
- **Commit:** `analysis(E9): confusion pairs, laugh-track contamination study, minority classes`
- **PR:** Failure modes characterized with evidence; limitations doc updated.

### T-31 REPORT.md + model-card fill + final polish
- **Objective:** course deliverables + interview-ready close-out.
- **Files:** `docs/REPORT.md`, MODEL_CARD numbers, §17 talking-point→artifact map appended to REPORT.
- **Depends on:** everything.
- **Details:** REPORT structure: problem→architecture→experiments (E1–E10 with tables)→XAI methodology+results→LLM comparison→limitations→conclusions. Both course reports must be extractable sections. Measured numbers only.
- **Done when:** M8 exit gate; repo public. · **Difficulty:** 2 · **Time:** 8 h
- **Commit:** `docs: final report, completed model card, talking-point artifact map`
- **PR:** Project close-out. All gates met; both course deliverables extractable.

---

## ⚖ Judgment Registry (every place engineering judgment will be needed, with decision guidance)

- **J-01 MELD source mirror** (T-03): if the official download is dead, prefer (in order): the official GitHub-linked mirror → HF datasets for text/labels + any archived audio tarball whose per-file checksums you then freeze. Whatever you pick, freeze checksums immediately and note the source in an ADR. Never mix audio and CSVs from different mirrors without verifying utt_id alignment.
- **J-02 Loudness-norm gain clamp** (T-04): ±20 dB default. If the audit shows many clips clamped, prefer widening to ±30 dB over disabling norm; log the clamp count either way.
- **J-03 Encoder dropout in Stage 1** (T-07): encoders in eval() (dropout off) is the frozen-feature-extractor convention and matches the cache. If someone proposes train-mode encoders for regularization: rejected — it would also invalidate the cache. ADR territory.
- **J-04 Where B3 pools from** (T-11): decided — post-projection (256-d) streams, so E3 isolates the xattn+gate mechanism. The alternative (pool raw 768-d) answers a different, less useful question. Don't reopen.
- **J-05 Attention-score aggregation** (T-19): default = mean over heads, sum incoming-attention mass per token/frame, blended with pool weights multiplicatively. There is genuine freedom here; any defensible formula is fine PROVIDED it's (a) documented in the docstring, (b) held fixed across all analyses, (c) reported as one point in the agreement study — attention aggregation ambiguity is itself part of why attention-as-explanation is unreliable; say that in the report.
- **J-06 Break-even amortization assumptions** (T-25): assume self-host = one cloud T4-class instance at a stated $/h and a stated utilization; put both constants in the config and the table footnote. The point is a transparent model, not a precise one.
- **J-07 Hyperparameter drift** (T-12/T-15): you may tune lr ∈ [5e-5, 5e-4], epochs ≤20, warmup ∈ [5–15%] on DEV only; anything beyond (architecture dims, heads, blocks, losses) is ADR territory. Log every tried config's run dir; report the search honestly in REPORT (a small, honest search beats a hidden large one).
- **J-08 If M loses to B3 significantly** (T-15): do NOT iterate architecture chasing the win (that's post-hoc fishing). One permitted diagnostic pass: check gate saturation (g_a→0 everywhere suggests the audio stream is noise-dominated; report it) and INV1 again. Then write the negative result. The blueprint pre-authorizes this outcome.
- **J-09 Occlusion window sizes** (T-20): 3 tokens / 1 s are defaults; if audit shows median cur-span <6 tokens, drop to 2-token windows. Fixed once chosen.
- **J-10 Whisper model size** (T-26): `small` default; if demo latency on CPU >5 s for 10 s audio, drop to `base` and note the WER cost in the UI caveat. Never `large` (CPU demo).
- **J-11 Gemini subset fallback trigger** (T-25): projected cost > $12 (leave $3 headroom of the $15 cap) → 1K subsample. Mechanical rule, no judgment at run time.
- **J-12 Time-pressure scope cuts:** already decided — ROADMAP cut order. The judgment call is WHEN to invoke it: if any milestone slips >1 week past plan, invoke the next cut immediately rather than compressing quality. Quality of what ships > quantity of what's attempted.
