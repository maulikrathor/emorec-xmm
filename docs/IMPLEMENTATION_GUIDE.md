# IMPLEMENTATION_GUIDE.md

**Why this document exists:** the step-by-step "how," in dependency-true order, with the known traps pre-flagged at the exact step where they bite. Read the section for the module you're building at the start of the session; don't read ahead and don't skip ahead.

## Global order (from BLUEPRINT §11 — binding)

1 scaffold/CI/config/seeding → 2 download/preprocess/manifests/audit → 3 dataset+collator → 4 encoders+audio-cache → 5 metrics+losses → 6 trainer → 7 baselines B1→gate→B2→B3 → 8 fusion+INV1 test→train M→E2 → 9 [conditional] LoRA → 10 IG→attention→occlusion → 11 faithfulness→dominance→viz → 12 Gemini (cache layer FIRST) → 13 demo→ONNX→Docker → 14 model card→README→REPORT→error analysis.

Rule: item N's required tests green before item N+1 starts.

## Step notes and traps

### 1. Scaffold
pyproject with pinned deps; extras: `[xai]` captum, `[bench]` google-genai, `[serve]` gradio+faster-whisper+onnxruntime, `[dev]` pytest+ruff. CI = ruff + pytest on push, CPU, <5 min. Write `seed_everything` (torch, numpy, random, cudnn deterministic) + config loader first — everything depends on them.

### 2. Data pipeline
- MELD raw = CSVs + MP4 clips per split. ffmpeg: `-ac 1 -ar 16000`. pyloudnorm to −23 LUFS (guard: skip norm if clip <0.4 s or silent — pyloudnorm raises on very short input; log these).
- **Trap (R4):** broken/missing/zero-length clips are NORMAL in MELD. Exclusion list with counts goes in the audit report; excluded utterances dropped from manifests, counts reported per split per class.
- Manifest row: `{utt_id, dialogue_id, turn_idx, speaker, text, label, wav_path, duration}`. JSONL, one per split.
- Run `audit.py` immediately: class dist per split, duration histogram, k=2 token-length histogram (validates max_len 160 → else ADR), corrupt list. Commit the report.

### 3. Dataset & collator
- Context builder = pure function; golden-string unit test on a synthetic 3-turn dialogue BEFORE wiring to tokenizer. Handle dialogue-start (t<k: fewer context turns, no padding turns).
- Add `<cur>`/`</cur>` via `tokenizer.add_special_tokens` and record their ids in the dataset object (XAI needs them to find the span).
- Collator pads text (tokenizer pad) and audio (zero-pad + boolean mask) and returns BOTH masks. Audio mask must be downsampled to frame resolution later — do it in the encoder wrapper via `model._get_feature_vector_attention_mask` (HF provides this for wav2vec2-family models; WavLM inherits it). **Do not hand-roll the frame-mask math.**
- Truncation test: assert the kept audio is the END of the clip (ED5).

### 4. Encoders + cache
- `resize_token_embeddings` after adding special tokens; then freeze everything EXCEPT the two new embedding rows (implement via a gradient hook or by keeping embeddings trainable and zeroing grads for all rows but the two new ids — the hook is cleaner; test INV3 either way).
- Audio cache: iterate manifests once, save fp16 `(T_frames,768)` + frame mask per utt to `artifacts/cache/{sha1(ckpt)[:8]}/{utt_id}.npz`. Dataset gains `use_audio_cache` flag. Verify: cached vs online forward identical (fp16 tolerance).

### 5–6. Metrics, losses, trainer
- Metrics pure functions, tested against sklearn. Bootstrap CI: resample utterance indices 1,000×; percentile 2.5/97.5. McNemar: statsmodels or hand-rolled exact binomial on the discordant pairs.
- Trainer smoke test: 32-sample tiny subset must overfit (train loss → ~0) in <50 steps with encoders frozen. If it can't, the wiring is broken — fix before real training.

### 7. Baselines
Train B1 first. **GATE: B1 dev wF1 ≥ 60.** If below: the bug is almost certainly in (a) context builder, (b) label mapping, or (c) class weights — in that order of likelihood. Do not proceed. Then B2 (gate ≥ 38), then B3. 1 seed each for the gate check; 3 seeds for B1/B3 after.

### 8. Fusion
- Write INV1 (mask-invariance) test FIRST, red→green against the untrained model: same example padded into a batch of long sequences vs a batch of short ones ⇒ identical logits (atol 1e-5). If it fails: you are attending to padding — check both attention directions and the pooling masks.
- Then train M, 3 seeds; then E2 (k ∈ {0,1,2,4}); then E3 comparison table with CI + McNemar vs B3.

### 9. LoRA (only if M4 gate passes — see ROADMAP)
peft, r=8 α=16, q/v of top-4 layers each encoder, lr 1e-4. Audio cache is INVALID in this stage (encoder changes) — train online, expect 3–4× slower epochs; that's why this stage is conditional.

### 10. XAI — the Captum wiring pattern (do it this way, not another way)
Attribute ONE modality per pass, other held fixed:
```
def fwd_text(text_embeds):            # closes over: audio_hidden, masks, target
    return model.forward_from_embeds(text_embeds, audio_hidden, ...).logits
lig = LayerIntegratedGradients(...)   # or IG on embeds directly via forward_from_embeds
```
Requires the model to expose `forward_from_embeds(text_embeds, audio_hidden, ...)` — build that hook into `fusion.py` NOW (step 8), not retroactively. Baselines: pad-token embeddings (text), zero post-proj frames (audio). Always log convergence delta; n_steps 32→64 rule. Save `.npz` per utt (schema in MODULES.md). Attention extraction: return maps in `Output.xattn` during a no-grad pass. Occlusion: 500-sample stratified subset (`scripts/stratify_subset.py`, seeded).

### 11. Faithfulness & dominance
Deletion = pad-embedding (text) / zero-frame (audio) — consistent with IG baselines by construction; state this in the report. Curves at {5,10,20,50}%; AUC per method × modality. Sanity: parameter-randomization on the classifier head — attributions must degrade; if not, report it as a method failure. Dominance: three signals per §18.6, per-modality max-normalize IG before mass ratios (otherwise scale differences fake dominance).

### 12. Gemini runner
Build the cache + budget-cap + resumability skeleton with a MOCK client and tests BEFORE the first real API call. Prompt frozen in `bench/prompts/v1.txt`; any change ⇒ `v2.txt`, never edit v1. Two conditions: audio+text, text-only. Record token counts, wall latency, invalid-rate. Price-sheet snapshot into the report at run time.

### 13. Serving
Pipeline first (testable, headless), UI second. Whisper `small` via faster-whisper (CPU int8). ONNX: export fusion head with dynamic axes (T_t, T_a); parity test INV5 MANDATORY; then latency script (batch 1, 10 warmup, median of 100). Docker: CPU-only, weights from HF Hub at start.

### 14. Docs & analysis
Model card from MODEL_CARD.md skeleton (fill measured numbers only). Error-analysis notebook: laugh-track contaminated sample study, top confusion pairs, 6 qualitative attribution case studies (2 text-dominant / 2 audio-helpful / 2 method-disagreement).
