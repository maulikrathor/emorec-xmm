# Engineering Blueprint — Explainable Multimodal Emotion Recognition (Speech + Text)

**Document status:** FROZEN — v1.0 — 2026-07-19
**Author role:** Principal AI Architect / Technical Lead (design phase)
**Implementer:** Successor engineer (Claude Opus 4.8 or human). This document assumes no further contact with the author. Every decision needed to build the system is recorded here. Where judgment calls remain, the decision rule is written down.

**Project owner:** Maulik — 4th-year CS (AI/ML), VIT. This project must simultaneously satisfy two courses (Speech and Language Processing; Explainable AI) and function as a flagship AI-engineering portfolio piece for 2026–27 internship/placement interviews. When course requirements and portfolio quality conflict, portfolio quality wins as long as course rubrics are still met — both courses are satisfied by strict subsets of this blueprint.

---

## 0. Final assumption challenge and scope freeze

Every assumption was re-examined one last time before freezing. Results:

| Assumption | Verdict | Rationale |
|---|---|---|
| MELD dataset | **KEEP** | Standard benchmark, published leaderboard, dialogue structure (needed for SLP course), free, manageable size (~13.7K utterances). Its known weakness (noisy audio, text dominance) is *reframed as the research question*, not a flaw. IEMOCAP rejected: license friction, acted (not spontaneous-ish) data, no better story. |
| RoBERTa-base text encoder | **KEEP** | Strong, standard, well-understood, fits compute. DeBERTa-v3 gains ~0.5–1 F1 but complicates tokenizer/attribution work for no narrative gain. |
| Wav2Vec2 audio encoder | **CHANGED → WavLM-base-plus (default), Wav2Vec2-base-960h (ablation)** | This is the single architectural change made at freeze time. WavLM-base-plus consistently outperforms Wav2Vec2-base on paralinguistic/emotion tasks (SUPERB ER benchmark) and is a drop-in replacement: same HuggingFace `AutoModel` interface, same feature-extractor pattern, same output shape family. The audio encoder is therefore specified as an *interface* (see §4.2); the checkpoint is a config string. Keeping Wav2Vec2 as a documented ablation preserves the original plan as an experiment rather than discarding it. Cost of change: zero. Benefit: likely +1–2 audio-branch F1 and a ready-made ablation table row. |
| Simple concatenation fusion | **REJECTED (already decided) → bidirectional gated cross-attention** | Concat is retained only as baseline B3 in the experiment grid. See §4.3 and §15. |
| Integrated Gradients as sole XAI method | **REJECTED → IG + attention-based attribution + occlusion, with quantitative faithfulness evaluation** | A single unevaluated attribution method fails the XAI course's spirit and produces no defensible claims. See §18. |
| Fine-tune both encoders end-to-end | **REJECTED as default → frozen encoders + trainable fusion, with LoRA escalation path** | Compute is unconfirmed (see §13 A1). The architecture must produce a complete, publishable result on a single 16 GB GPU (Colab Pro T4 / Kaggle P100/T4x2). Full fine-tuning is an *optional* stage 2, not a dependency. |
| Frontier LLM baseline | **KEEP — Gemini 2.5 Flash with native audio input** | Owner already has Gemini API experience (Prysom internship). One model is sufficient; a second (GPT-4o-audio) is a stretch goal, not scope. |
| Gradio demo + ONNX + model card | **KEEP** | This is what separates "student repo" from "engineer repo." Non-negotiable scope. |

**SCOPE IS NOW FROZEN.** Any change to the items above after this point requires writing an ADR (Architecture Decision Record, see §14) into `docs/adr/` explaining what new information invalidated this document.

Out of scope (explicitly): video modality, real-time streaming inference, training audio encoders from scratch, multilingual support, deployment to paid cloud infrastructure, any dataset beyond MELD (except as future work, §16).

---

## 1. Final problem statement

Emotion recognition in conversation (ERC) systems in production are overwhelmingly text-only, because text models are cheap, strong, and easy to deploy — yet human emotion is substantially carried by prosody (tone, pitch, energy, rhythm) that text discards. The open engineering questions are: **(a)** when and how much does the audio channel actually improve emotion recognition over strong text baselines in realistic, noisy conversational data; **(b)** can we build a fusion architecture that exploits audio where it helps without degrading the text signal where it doesn't; and **(c)** can post-hoc explainability methods *reliably* tell us, per prediction, which modality and which tokens/frames drove the decision — and how do we know the explanations themselves are faithful rather than decorative?

This project builds a complete, deployable multimodal ERC system on MELD (7-class: neutral, joy, sadness, anger, surprise, fear, disgust) that answers all three questions quantitatively, and benchmarks the resulting specialist model against a prompted frontier multimodal LLM on accuracy, cost, and latency — the exact build-vs-buy tradeoff facing AI engineering teams in 2026.

**Central honest hypothesis (write this into the README from day one):** text will dominate globally on MELD; audio will contribute measurably on prosody-heavy classes (anger, surprise) and on utterances that are textually ambiguous (short utterances, interjections, sarcasm-adjacent cases). The XAI layer's job is to *detect and quantify* this pattern, and the faithfulness evaluation's job is to show the detection is trustworthy. If the hypothesis is falsified in either direction, that is a reportable finding, not a failure.

## 2. Final project objectives

1. **O1 — Strong reproducible baselines.** Text-only, audio-only, and concat-fusion baselines on MELD with weighted-F1 within the published range for comparable model classes (text-only RoBERTa-base + context: target ≥ 62 weighted-F1; audio-only: target ≥ 41; these targets are sanity gates, not success criteria).
2. **O2 — Fusion that earns its complexity.** Gated cross-attention fusion that beats the concat baseline by a statistically meaningful margin (bootstrap CI, see §18.4) OR, if it does not, a documented analysis of why (this outcome is acceptable and reportable).
3. **O3 — Evaluated explainability.** Per-prediction attributions from ≥ 2 independent method families (gradient-based IG; attention-based; occlusion as arbiter), with faithfulness measured by comprehensiveness/sufficiency deletion curves, and cross-method agreement analysis.
4. **O4 — Modality dominance quantification.** A per-class and per-utterance modality contribution score computed three independent ways (attribution mass, modality ablation delta, gate activation statistics), with agreement analysis among the three.
5. **O5 — Frontier LLM benchmark.** Gemini 2.5 Flash (audio+text, few-shot) evaluated on the identical test split; comparison table: weighted-F1, macro-F1, per-class F1, cost per 1K utterances, p50/p95 latency.
6. **O6 — Product-grade delivery.** Public GitHub repo with CI, tests, model card, Gradio demo (mic → Whisper → prediction + live attribution visualization), ONNX-exported inference path with latency table, Dockerfile, and a written technical report.
7. **O7 — Course deliverables.** SLP course: dialogue-context modeling, ASR integration, prosodic feature analysis (§18.6). XAI course: everything in O3/O4 plus the faithfulness methodology writeup.

## 3. Why this project matters in 2026

Three converging industry realities make this project timely rather than academic:

**The build-vs-buy question is the defining AI engineering decision of 2026.** Every team now chooses between prompting a frontier multimodal LLM and training/hosting a specialist model. The tradeoff is accuracy vs. cost vs. latency vs. controllability vs. explainability. Almost no student portfolios contain an honest, measured instance of this comparison. This project produces one, with numbers.

**Voice AI is production-mainstream.** Voice agents (customer support, sales QA, healthcare triage, in-car assistants) exploded through 2025–26. Emotion/affect signals are a core analytics layer in these products (escalation detection, QA scoring, coaching). The system built here is a realistic component of that stack, and the demo proves it end-to-end (microphone → ASR → emotion + explanation).

**Explainability is becoming a compliance requirement, not a nice-to-have.** The EU AI Act's high-risk provisions and equivalent emerging frameworks push affect-recognition systems specifically into heightened-scrutiny categories. "We can show which input drove which prediction, and we validated that the explanation is faithful" is a sentence regulated deployers need. The faithfulness-evaluation layer (rare even in industry) is the differentiator.

Secondary rationale: the specialist-model skillset (training, evaluation rigor, ONNX deployment) differentiates the owner from the 2026 cohort of LLM-wrapper-only graduates, while the LLM benchmark connects the project to his existing LLM orchestration experience — the portfolio tells one coherent story: *knows when to train, knows when to prompt, can measure the difference.*

---

## 4. Final architecture

### 4.0 Architecture summary (one paragraph)

Two frozen pretrained encoders — RoBERTa-base over the context-augmented transcript and WavLM-base-plus over the raw 16 kHz waveform — produce token-level and frame-level hidden states. Both are linearly projected into a shared d=256 space. A 2-layer bidirectional gated cross-attention block lets each modality attend to the other; attention-pooled summaries of both enriched streams are combined through a learned scalar gate (per-example, per-modality) and passed to a 2-layer MLP classifier over 7 classes, trained with class-weighted cross-entropy. Only projections, fusion, gate, and classifier train in Stage 1 (~5–8M trainable params). Stage 2 (optional, compute-permitting) adds LoRA adapters (r=8) to the top 4 layers of each encoder. Explainability operates on the trained model post-hoc (IG, attention attribution, occlusion) plus intrinsically (the gate values themselves are a modality-dominance signal).

### 4.1 Text branch (SLP-relevant design)

- **Checkpoint:** `roberta-base` (125M params, frozen in Stage 1).
- **Input construction (dialogue context):** for utterance *u_t* in dialogue *D*, the input string is:
  `[spk(t-2)]: u_(t-2) </s> [spk(t-1)]: u_(t-1) </s> <cur> [spk(t)]: u_t </cur>`
  where `[spk]` are speaker-name literals (MELD provides them), `</s>` is RoBERTa's separator, and `<cur>` / `</cur>` are two added special tokens (resize embeddings; the two new rows ARE trainable even in Stage 1). Context window k=2 prior utterances; k is a config value; k ∈ {0,1,2,4} is experiment E2.
- **Why current-utterance-last with markers:** (a) recency bias of transformers favors the classified utterance; (b) the `<cur>` span boundary lets the attribution module cleanly separate "context tokens" from "target tokens" — required for honest text attribution (otherwise IG credit assigned to context inflates apparent text dominance).
- **Max length:** 160 tokens (covers >99% of k=2 inputs on MELD; verify in data audit, M1).
- **Output used:** full last-hidden-state sequence (B, T_txt, 768) — NOT just CLS — because cross-attention needs the token sequence.

### 4.2 Audio branch

- **Interface contract:** the audio encoder is any HF model exposing `(B, T_frames, H_a)` last-hidden-state from raw 16 kHz mono waveform input. Checkpoint is a config string.
- **Default checkpoint:** `microsoft/wavlm-base-plus` (95M, frozen). **Ablation checkpoint:** `facebook/wav2vec2-base-960h`.
- **Preprocessing:** MELD ships MP4 clips → extract mono WAV @16 kHz via ffmpeg (one-time script, cached to disk). Loudness-normalize to −23 LUFS (pyloudnorm) — MELD clip volumes vary wildly. Clip/pad to max 10.0 s (covers ~97% of utterances; truncate from the *start*, keeping the end — utterance-final prosody carries the most emotion signal; this is a deliberate, documented choice).
- **Frame rate:** both candidate encoders emit ~49–50 frames/s → ≤ ~500 frames per utterance.
- **Known hazard (design around it, do not discover it):** MELD audio contains laugh track, music, overlapping speakers. Do NOT attempt source separation (scope creep). Instead: (a) loudness normalization, (b) report results as-is, (c) the error-analysis notebook (E9) specifically samples laugh-track-contaminated clips to characterize the failure mode. Honesty > heroics.

### 4.3 Fusion module (the architectural centerpiece)

```
text hidden   (B, T_t, 768) ──proj_t──▶ (B, T_t, 256) ─┐
                                                        ├─▶ CrossModalBlock ×2 ─▶ enriched_t (B,T_t,256), enriched_a (B,T_a,256)
audio hidden  (B, T_a, 768) ──proj_a──▶ (B, T_a, 256) ─┘
enriched_t ──attn-pool──▶ z_t (B,256) ─┐
                                        ├─▶ gate g = σ(W[z_t;z_a]+b) ∈ (0,1)^2 (per-modality scalar, softmax-normalized)
enriched_a ──attn-pool──▶ z_a (B,256) ─┘
fused = g_t·z_t ⊕ g_a·z_a  (⊕ = concat of gated vectors, dim 512)
fused ──MLP(512→256→7)──▶ logits
```

- **CrossModalBlock (one block):** pre-LN; MHA-1: text queries attend audio keys/values (8 heads, d=256); MHA-2: audio queries attend text keys/values; each followed by a per-stream FFN (256→512→256, GELU) with residuals. Padding masks respected in both directions (this is the #1 silent-bug risk in the whole system — see §12 R6).
- **Attention pooling:** learned query vector per modality attends over its enriched sequence → single vector. (Rejected: mean-pool — loses localization; CLS-token — audio has none.)
- **The gate is load-bearing for XAI:** `g_t, g_a` are logged for every prediction and become modality-dominance signal #3 (§18.5). This makes dominance analysis partly *intrinsic* to the architecture, not purely post-hoc — a genuine talking point.
- **Parameter budget:** projections ~0.4M, 2 fusion blocks ~1.6M, pools+gate+MLP ~0.3M → ~2.3M trainable + 2 embedding rows. Trains comfortably on 16 GB with batch 8 + gradient accumulation ×4.

### 4.4 Classifier, loss, training

- **Loss:** cross-entropy with class weights = inverse-sqrt frequency, normalized to mean 1 (full inverse-frequency over-corrects on MELD's extremes — fear/disgust have only a few hundred train samples). Focal loss (γ=2) is ablation E5, not default.
- **Optimizer:** AdamW, lr 2e-4 (fusion params), weight decay 0.01, cosine decay, warmup 10%, 15 epochs max, early stop on dev weighted-F1 (patience 3). Seed everything; 3 seeds for headline numbers (§18.4).
- **Stage 2 (optional):** LoRA r=8, α=16 on q,v projections of top-4 layers of each encoder; lr 1e-4; only if Stage 1 complete AND compute confirmed ≥ T4-16GB with ≥ 6 h/session. Decision gate written into roadmap M5.

### 4.5 Explainability subsystem (post-hoc + intrinsic)

Operates on a frozen trained checkpoint. Three attribution families + one intrinsic signal:

1. **Integrated Gradients (Captum), per modality.** Text: attribute w.r.t. embedding-layer output (LayerIntegratedGradients), baseline = pad-token embeddings; report token-level scores restricted to the `<cur>` span, with context-token attribution reported separately. Audio: attribute w.r.t. the *projected* audio frames (post-`proj_a`) — NOT raw waveform (waveform-level IG is noise); baseline = zero vectors; aggregate frames into 0.5 s bins for visualization. n_steps=32 (raise to 64 if convergence delta > 5%; log convergence delta always).
2. **Attention-based attribution:** cross-attention weight rollout from the fusion blocks (text→audio and audio→text maps), plus attention-pool weights. Cheap, intrinsic-ish, known-imperfect — that imperfection is the point of the agreement analysis.
3. **Occlusion (arbiter):** mask token spans / 1 s audio windows, measure Δ logit of predicted class. Slow → run on stratified 500-utterance test subsample only.
4. **Intrinsic gate signal:** `(g_t, g_a)` per example, free at inference.

All four feed the evaluation methodology in §18. The demo visualizes #1 and #4 live.

### 4.6 LLM benchmark subsystem

- **Model:** `gemini-2.5-flash` via API, native audio input (upload WAV) + transcript + k=2 context in prompt.
- **Protocol:** fixed few-shot prompt (7 examples, one per class, drawn from train set, frozen in a versioned prompt file); temperature 0; constrained JSON output `{"emotion": <label>}` with retry-on-parse-failure (max 2 retries, then mark invalid; report invalid rate). Full MELD test set (2,610 utterances). Cache every response to disk keyed by utterance-ID + prompt-hash (idempotent, resumable, protects against rate limits and re-run costs).
- **Measurements:** weighted/macro/per-class F1, per-call cost from token counts, wall-clock p50/p95 latency, invalid-output rate. Also run a *text-only* Gemini condition (same prompt minus audio) — this isolates whether the LLM itself extracts audio signal, a genuinely interesting result either way.
- **Budget cap:** hard cap ~US$15. If projected cost exceeds cap, fall back to a stratified 1,000-utterance test subsample (stratification script in `scripts/`), and report accordingly.

### 4.7 Serving & demo subsystem

- **Gradio app**, two input paths: (a) microphone/file → faster-whisper `small` ASR → transcript (editable by user) + audio → model; (b) pick a MELD test sample. Output: predicted emotion + probability bar chart, token-level text attribution (colored spans), audio attribution timeline (0.5 s bins aligned under a waveform plot), and the gate values (g_t vs g_a dial). Note in UI: demo has no dialogue context for live input (k=0 path); MELD-sample path uses full context.
- **ONNX export:** projections + fusion + classifier exported as one graph ("fusion head"); encoders run via HF/PyTorch (encoder ONNX export is a stretch goal, not scope — WavLM export has known operator issues; do not burn time). Latency table: PyTorch-CPU vs PyTorch-GPU vs ONNX-CPU (fusion head), plus end-to-end (encoders included) on both devices.
- **Docker:** single CPU-only image running the Gradio app; model weights pulled from HF Hub at build or mounted. GPU serving out of scope.

---

## 5. Component interaction diagram

```mermaid
graph TB
    subgraph DATA["data layer"]
        RAW[MELD raw CSVs + MP4s] --> PREP[preprocess: ffmpeg 16kHz WAV,\nloudness norm, manifest builder]
        PREP --> MAN[(manifests: train/dev/test.jsonl\n+ wav cache)]
    end
    subgraph CORE["model core"]
        MAN --> DS[MeldDataset + collator\n(dialogue context builder,\npad masks both modalities)]
        DS --> TE[TextEncoder\nroberta-base, frozen]
        DS --> AE[AudioEncoder\nwavlm-base-plus, frozen\n(checkpoint = config string)]
        TE --> FU[GatedCrossAttentionFusion\nproj → xattn×2 → pool → gate]
        AE --> FU
        FU --> CL[MLP classifier → 7 logits]
    end
    subgraph TRAIN["training"]
        CL --> TR[Trainer: weighted CE, AdamW,\nearly stop on dev wF1, 3 seeds]
        TR --> CKPT[(checkpoints + run logs)]
    end
    subgraph XAI["explainability"]
        CKPT --> IG[IG attribution\n(Captum, per modality)]
        CKPT --> AT[attention rollout]
        CKPT --> OC[occlusion arbiter\n(500-sample subset)]
        FU -. gate values g_t,g_a .-> DOM[modality dominance analysis]
        IG --> FAITH[faithfulness eval:\ncomprehensiveness/sufficiency,\nmethod agreement]
        AT --> FAITH
        OC --> FAITH
        IG --> DOM
    end
    subgraph BENCH["LLM benchmark"]
        MAN --> GEM[Gemini 2.5 Flash runner\n(few-shot, cached, cost/latency meters)]
        GEM --> CMP[comparison report:\nF1 / cost / latency]
        CKPT --> CMP
    end
    subgraph SERVE["serving"]
        CKPT --> ONNX[ONNX fusion-head export\n+ latency bench]
        CKPT --> APP[Gradio demo\nmic → whisper → predict\n+ live attributions + gate dial]
    end
```

## 6. Data flow

**Training-time flow (per batch):**
1. Manifest row → context builder assembles `k=2` text input with speaker tags and `<cur>` markers; audio path → cached WAV loaded, feature-extracted.
2. Collator pads both modalities; produces `text_ids, text_mask, audio_values, audio_mask, label, meta(utt_id, dialogue_id, speaker)`.
3. Frozen encoders (no_grad, eval mode, under autocast) → hidden sequences.
4. Fusion (grad-enabled) → logits → weighted CE → backward touches only fusion/classifier/new-embedding params.
5. Logged per step: loss, lr; per epoch: dev weighted-F1, macro-F1, per-class F1, mean gate values per class.

**Performance note for the implementer:** because encoders are frozen, encoder outputs for the *audio* branch MAY be precomputed once and cached as fp16 tensors on disk (~manageable size at ≤500×768 per clip), cutting epoch time ~3–4×. Text branch must stay online (context window k is an experiment variable). Build the audio-feature cache in M2; it is the single highest-leverage engineering optimization in the project. Invalidate cache on checkpoint-string change (hash the checkpoint name into the cache path).

**Inference-time flow (demo):** mic → 16 kHz mono resample → faster-whisper transcript → (k=0 context build) → encoders → fusion → softmax; in parallel on demand: IG for both modalities; gate values always returned.

**XAI-time flow:** load frozen checkpoint → iterate test manifest → per method, per example: attribution tensors saved to `artifacts/attributions/{method}/{utt_id}.npz` → evaluation scripts consume these files (attribution computation and attribution evaluation are decoupled on disk — enables re-running evals without recomputing IG).

**LLM-benchmark flow:** manifest → prompt builder (versioned template + few-shot block) → cache check → API call with audio upload → JSON parse/retry → response cache → aggregate metrics script.

---

## 7. Repository structure

Repo name suggestion: `emorec-xmm` (explainable multimodal). Single Python package, src-layout, uv or pip-tools for pinned deps.

```
emorec-xmm/
├── README.md                  # product-grade: banner, demo GIF, results tables, quickstart
├── LICENSE                    # MIT
├── pyproject.toml             # deps pinned; [project.optional-dependencies]: xai, bench, serve, dev
├── Dockerfile                 # CPU-only Gradio serving image
├── .github/workflows/ci.yml  # lint (ruff) + unit tests on push; no GPU in CI
├── configs/
│   ├── base.yaml              # every default in this blueprint, single source of truth
│   ├── text_only.yaml         # baseline B1 (overrides)
│   ├── audio_only.yaml        # baseline B2
│   ├── concat.yaml            # baseline B3
│   ├── xattn.yaml             # main model M
│   └── xattn_lora.yaml        # stage-2 (optional)
├── src/emorec/
│   ├── data/
│   │   ├── download.py        # MELD fetch + checksum verify
│   │   ├── preprocess.py      # ffmpeg extraction, loudness norm, manifest build
│   │   ├── dataset.py         # MeldDataset, context builder, collator
│   │   └── audit.py           # data audit report (M1 deliverable)
│   ├── models/
│   │   ├── encoders.py        # TextEncoder, AudioEncoder wrappers (freeze logic, caching hooks)
│   │   ├── fusion.py          # CrossModalBlock, AttnPool, ModalityGate, full model class
│   │   ├── baselines.py       # text-only / audio-only / concat heads (share encoder wrappers)
│   │   └── registry.py        # config-string → model factory
│   ├── training/
│   │   ├── trainer.py         # loop, amp, grad-accum, early stop, seed control
│   │   ├── losses.py          # weighted CE, focal (ablation)
│   │   └── metrics.py         # wF1/mF1/per-class, bootstrap CI, McNemar
│   ├── xai/
│   │   ├── ig.py              # LayerIG both modalities, convergence-delta logging
│   │   ├── attention.py       # cross-attn rollout extraction
│   │   ├── occlusion.py       # span/window occlusion, subset runner
│   │   ├── faithfulness.py    # comprehensiveness/sufficiency curves, AUC
│   │   ├── dominance.py       # 3-way modality dominance scores + agreement
│   │   └── viz.py             # token heatmaps, audio timeline plots (shared w/ demo)
│   ├── bench/
│   │   ├── gemini_runner.py   # cached, resumable, cost+latency metered
│   │   ├── prompts/v1.txt     # frozen prompt + few-shot block (versioned)
│   │   └── compare.py         # comparison tables
│   ├── serve/
│   │   ├── app.py             # Gradio UI
│   │   ├── pipeline.py        # whisper → model → attribution glue
│   │   └── export_onnx.py     # fusion-head export + latency bench
│   └── utils/                 # seeding, io, logging, config loading (OmegaConf)
├── scripts/                   # thin CLI entrypoints: prepare_data, train, evaluate,
│                              # run_xai, run_faithfulness, run_gemini, export, stratify_subset
├── tests/                     # see §9 module table for required tests
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_results_analysis.ipynb
│   └── 03_error_analysis.ipynb   # incl. laugh-track failure-mode study
├── docs/
│   ├── BLUEPRINT.md           # this document, committed verbatim
│   ├── adr/                   # any post-freeze decision changes, numbered
│   ├── MODEL_CARD.md
│   └── REPORT.md              # course/technical report source
└── artifacts/                 # gitignored: wavs, caches, checkpoints, attributions
```

Rules: `src/emorec` never imports from `scripts/` or `notebooks/`. Notebooks import the package; they contain zero logic that results depend on. Every figure in README/REPORT is regenerated by a script or notebook committed to the repo.

## 8–9. Every module and its responsibilities

| Module | Responsibility (contract) | Key interfaces | Required tests |
|---|---|---|---|
| `data/download.py` | Fetch MELD, verify checksums, unpack to `artifacts/raw/` | `download(dest) -> None`, idempotent | checksum-mismatch raises |
| `data/preprocess.py` | MP4→16kHz mono WAV, −23 LUFS norm, ≤10 s end-keeping truncation; build `{split}.jsonl` manifests with utt_id, dialogue_id, turn_idx, speaker, text, label, wav_path, duration | `build_manifests(raw, out)` | manifest row count matches CSV; a known clip's duration/sr correct |
| `data/dataset.py` | Context builder (k configurable, speaker tags, `<cur>` markers), tokenization, audio loading, collation with dual masks | `MeldDataset(manifest, cfg)`, `collate(batch) -> Batch` | context string exact-match for a synthetic 3-turn dialogue; k=0 path; mask shapes; truncation keeps utterance-final audio |
| `data/audit.py` | One-command data audit: class distribution per split, duration histogram, token-length histogram (validates max-len 160 claim), missing/corrupt file list | `audit(manifests) -> report.md` | runs on toy manifest |
| `models/encoders.py` | Wrap HF models; freeze/unfreeze; expose `(hidden, mask)`; audio-feature disk cache keyed by checkpoint-hash; special-token embedding resize (trainable rows) | `TextEncoder(cfg)`, `AudioEncoder(cfg)` | frozen params have `requires_grad=False`; cache hit returns identical tensors; new tokens trainable |
| `models/fusion.py` | §4.3 exactly; returns logits AND `(g_t, g_a)` AND cross-attn weights (for XAI) in a structured output object | `MultimodalERC(cfg).forward(batch) -> Output` | padding-mask invariance test (see §12 R6 — MANDATORY); output shapes; gate ∈ (0,1), sums to 1 |
| `models/baselines.py` | B1 text-only (attn-pool → MLP), B2 audio-only, B3 concat(z_t,z_a)→MLP; reuse encoder wrappers so comparisons differ ONLY in fusion | same Output contract | shape tests |
| `models/registry.py` | `build_model(cfg)` dispatch on `cfg.model.name` | — | unknown name raises |
| `training/trainer.py` | Full loop: amp autocast, grad-accum, clip 1.0, early stop patience 3 on dev wF1, checkpoint best+last, seed control, CSV/JSONL run logs, resumable | `Trainer(cfg).fit()` | 2-step overfit-tiny-batch smoke test (loss decreases) |
| `training/losses.py` | Weighted CE (inv-sqrt, mean-normalized), focal | `build_loss(cfg, class_counts)` | weights math vs hand-computed |
| `training/metrics.py` | wF1, mF1, per-class F1, confusion matrix, bootstrap CI (1,000 resamples over utterances), McNemar paired test | pure functions on `(y_true, y_pred)` | vs sklearn on toy data |
| `xai/ig.py` | LayerIG text (embedding layer) + audio (post-proj), pad/zero baselines, n_steps 32→64 rule, convergence delta logged, save `.npz` per utt | `run_ig(ckpt, manifest, out)` | attribution shape == input shape; completeness axiom approx holds on toy |
| `xai/attention.py` | Extract & aggregate cross-attn maps + pool weights → per-token/frame scores, same `.npz` schema as IG | `run_attn(...)` | rows sum to 1 pre-aggregation |
| `xai/occlusion.py` | Token-span (3-token windows) and audio (1 s windows) occlusion Δlogit; stratified 500-sample subset | `run_occlusion(...)` | occluding everything ≈ baseline-input prediction |
| `xai/faithfulness.py` | Comprehensiveness & sufficiency curves at {5,10,20,50}% deletion/retention per method per modality; AUC; method-agreement (Spearman over shared examples) | `evaluate(attr_dir, ckpt)` | curve monotonic-ish on a synthetic perfectly-faithful attribution |
| `xai/dominance.py` | Per-utterance dominance via (a) IG mass ratio within `<cur>` span vs audio frames — normalized per modality first, (b) ablation delta: Δlogit when replacing one modality with baseline, (c) gate values; per-class aggregation; 3-way agreement (pairwise Spearman + majority-vote label) | `analyze(...) -> tables+plots` | ablation of both modalities → near-uniform logits |
| `xai/viz.py` | Token heatmap HTML/matplotlib, audio timeline plot aligned to waveform, gate dial; shared by notebooks and demo | pure rendering | golden-image smoke test optional |
| `bench/gemini_runner.py` | §4.6 protocol: cache-first, resumable, JSON-constrained, retry ≤2, cost/latency meters, budget-cap abort | `run(manifest, prompt_version)` | cache round-trip; parse-failure path; cap-abort |
| `bench/compare.py` | Merge our metrics + Gemini metrics → markdown tables (F1/cost/latency), including Gemini text-only condition | — | table math |
| `serve/pipeline.py` | mic-or-file → resample → faster-whisper small → k=0 context → predict → optional IG → dict for UI | `predict(audio, text=None)` | end-to-end on a bundled 2 s test wav |
| `serve/app.py` | Gradio UI per §4.7; MELD-sample picker path | `main()` | manual |
| `serve/export_onnx.py` | Export fusion head (proj+xattn+pool+gate+MLP) with dynamic axes; parity check vs PyTorch (atol 1e-4); latency table script | — | parity test MANDATORY |
| `utils/*` | seeding (torch/numpy/random + cudnn deterministic flag), OmegaConf load/merge, JSONL logging | — | seed reproducibility test |

## 10. Milestone roadmap (12 weeks, ~10–12 h/week; owner also has capstone + TopCoder — buffers are real)

**M0 — Week 1: Skeleton & environment.** Repo scaffold, CI (ruff + pytest), configs, seeding utils, download script. *Exit:* CI green, MELD downloaded, checksums verified.

**M1 — Week 2: Data pipeline + audit.** Preprocess, manifests, dataset/collator with context builder, audit report. *Exit:* audit notebook committed; token-length claim (§4.1) verified or max_len ADR'd; all corrupt clips listed and excluded with counts.

**M2 — Weeks 3–4: Baselines B1/B2/B3.** Trainer, losses, metrics, audio-feature cache, three baselines trained (1 seed each first, then 3 seeds for B1/B3). *Exit gate:* B1 ≥ 60 wF1 and B2 ≥ 38 wF1 on dev. If B1 < 60: debug data pipeline before proceeding (most likely: context builder or label mapping bug). **Do not build fusion on top of broken baselines.**

**M3 — Week 5: Main fusion model.** `fusion.py` + mandatory mask-invariance test; train M (3 seeds); k-ablation E2. *Exit:* M vs B3 comparison with bootstrap CI + McNemar, whatever the result.

**M4 — Week 6: Buffer / Stage-2 decision.** Catch-up week (it will be needed). Decision gate: if M2–M3 on schedule AND compute ≥ T4/6 h sessions → run LoRA stage-2 (E6); else skip permanently, note in ADR. *Never let LoRA jeopardize M5–M7.*

**M5 — Weeks 7–8: XAI subsystem.** IG, attention, occlusion, faithfulness, dominance. *Exit:* faithfulness AUC table for ≥2 methods × 2 modalities; 3-way dominance agreement table; ≥6 qualitative case-study figures (2 text-dominant, 2 audio-helpful, 2 method-disagreement cases).

**M6 — Week 9: Gemini benchmark.** Runner + both conditions (audio+text, text-only) + comparison tables. *Exit:* full comparison table incl. cost & latency; response cache committed-adjacent (artifacts) for reproducibility.

**M7 — Weeks 10–11: Demo, ONNX, packaging.** Gradio app, whisper path, ONNX export + parity + latency table, Dockerfile, model card, README with demo GIF and all result tables, HF Hub weight upload. *Exit:* `docker run` → working demo; a stranger can reproduce headline numbers from README in ≤ 5 commands.

**M8 — Week 12: Report + error analysis + polish.** REPORT.md, error-analysis notebook (laugh-track study, confusion-pair analysis: neutral↔joy? anger↔disgust?), interview-prep pass over §17. *Exit:* both course deliverables extractable from REPORT.md; repo public.

## 11. Exact implementation order (dependency-true sequence)

1. Repo scaffold → CI → config system → seeding utils
2. `download.py` → `preprocess.py` → manifests → `audit.py` (run audit NOW)
3. `dataset.py` + collator (+ its context-builder tests before anything consumes it)
4. `encoders.py` (freeze + special tokens) → audio feature cache
5. `metrics.py` + `losses.py` (pure, testable, no model needed)
6. `trainer.py` + overfit-tiny smoke test
7. `baselines.py`: B1 → train → **gate check** → B2 → B3
8. `fusion.py` + mask-invariance test → train M (3 seeds) → E2 k-ablation
9. [conditional] LoRA stage-2
10. `xai/ig.py` → `xai/attention.py` → `xai/occlusion.py` (in this order; IG first because faithfulness needs at least one method to develop against)
11. `xai/faithfulness.py` → `xai/dominance.py` → `xai/viz.py` + case studies
12. `bench/gemini_runner.py` (cache layer FIRST, API calls second) → `compare.py`
13. `serve/pipeline.py` → `app.py` → `export_onnx.py` (+ parity) → Dockerfile
14. Model card → README → REPORT → error-analysis notebook

Rule for the implementer: never start item N+1 while item N's required tests are red. The order above is chosen so every item is verifiable in isolation the moment it's built.

---

## 12. Risks (ranked by expected damage × likelihood)

| # | Risk | Likelihood | Impact | Mitigation (already designed in) |
|---|---|---|---|---|
| R1 | **Compute shortfall** (no reliable GPU ≥16 GB) | Medium | High | Frozen-encoder architecture + audio feature cache means Stage 1 trains on T4 in ~1–2 h/run. LoRA is optional behind a decision gate (M4). If even T4 is unreliable: batch 4 + accum 8, and drop 3-seed to 2-seed with a documented note. |
| R2 | **Fusion doesn't beat concat / audio adds ~nothing** | Medium-High | Medium (reputational only if unhandled) | Reframed as a finding from day one (§1 hypothesis). Deliverables O3–O5 are result-agnostic. The per-class analysis (anger/surprise) is where the positive story most likely lives; the dominance analysis is interesting *especially* if text dominates. |
| R3 | **Timeline collision** (capstone, TopCoder, placements) | High | High | 12-week plan has an explicit buffer week (M4) and a skip-able stage (LoRA). Minimum viable portfolio = M0–M3 + M5 + M7-lite (demo without ONNX). Cut order if forced: LoRA → occlusion→ ONNX → Gemini text-only condition → E-ablations. NEVER cut: baselines, faithfulness eval, demo, README. |
| R4 | **MELD audio download/extraction pain** (large files, broken clips) | Medium | Medium | Checksum verification, corrupt-clip exclusion list with counts in audit, preprocessing is one-time and cached. Budget a full day for this in M1; it historically takes it. |
| R5 | **Gemini cost/rate-limit surprises** | Low-Medium | Low | Cache-first resumable runner, hard budget cap with stratified-subsample fallback, temperature 0 + JSON constraint minimizes retries. |
| R6 | **Silent padding-mask bug in cross-attention** (model attends to pad frames; metrics look plausible but attributions are garbage) | Medium | High (poisons the entire XAI layer) | MANDATORY test: same example padded to two different lengths within differently-sized batches must produce identical logits (atol 1e-5). This test is written before fusion training begins. This is the single most important test in the repo. |
| R7 | **IG on audio is uninterpretable noise** | Medium | Medium | Already mitigated by design: attribute at post-projection frames not waveform; 0.5 s binning; occlusion as arbiter; if IG-audio faithfulness AUC is poor, that becomes a reported comparative finding, not a blocker. |
| R8 | **Whisper transcript mismatch at demo time** (model trained on gold transcripts, demo uses ASR) | High (it will mismatch) | Low-Medium | Documented openly in model card as train/serve skew; demo lets the user edit the transcript; optional experiment E8 quantifies gold-vs-ASR degradation on test set — a genuinely good SLP-course result. |
| R9 | **Reproducibility drift** (HF version bumps change numbers) | Medium | Medium | Pinned deps, seeds, committed configs, run logs in artifacts, weights on HF Hub. |

## 13. Assumptions (each with a validation step)

- **A1 — Compute:** owner has intermittent access to a single 16 GB-class GPU (Colab Pro / Kaggle). *Validate in M0 week 1: run a 100-step B1 training probe, record it/s. If unavailable → R1 mitigation path, no architecture change needed.*
- **A2 — MELD availability:** official MELD download remains up with published checksums. *Validate M0. Fallback: HF datasets mirror for text/labels + original audio archive.*
- **A3 — Token budget:** k=2 context fits in 160 RoBERTa tokens for >99% of samples. *Validated by audit in M1; else raise to 192 via ADR.*
- **A4 — Gemini 2.5 Flash still exposes native audio input at comparable pricing in the M6 window.* Validate at M6 start; fallback: any current Gemini flash-tier audio model; the runner is model-string-parameterized.*
- **A5 — Owner time:** ~10–12 h/week average. The plan degrades gracefully (R3 cut order) if this halves.
- **A6 — MELD standard splits are used unmodified** (train 9,989 / dev 1,109 / test 2,610 utterances) so numbers are leaderboard-comparable. Never re-split.
- **A7 — Labels are taken as ground truth** despite known annotator noise on MELD; label noise discussion belongs in the report's limitations, not the pipeline.

## 14. Engineering decisions (the "why" record — implementer must not silently reverse these)

- **ED1 Frozen encoders in Stage 1.** Compute-robustness (A1/R1), 3–4× faster iteration via audio cache, and clean scientific comparison (all models share identical representations; only fusion differs). Cost: a few F1 points vs full fine-tune — accepted; LoRA gate recovers some if compute allows.
- **ED2 WavLM-base-plus default.** Zero-cost swap, better paralinguistic prior (see §0). Wav2Vec2 preserved as ablation to honor the original plan and fill the ablation table.
- **ED3 d=256 shared fusion space.** Small enough to train fast and resist overfitting on 10K samples; large enough for 8-head attention. Not worth an experiment; fixed.
- **ED4 Per-modality scalar gate (not per-dimension).** Interpretability > expressiveness here: a scalar pair is directly readable as a dominance signal, which the XAI layer exploits. Per-dimension gating rejected as uninterpretable.
- **ED5 Truncate audio keeping the END of utterances.** Emotion prosody concentrates utterance-finally; documented, testable, defensible.
- **ED6 Class-weighted CE (inv-sqrt) over focal as default.** Simpler, one fewer hyperparameter, historically sufficient on MELD; focal is ablation E5.
- **ED7 Attribution artifacts decoupled on disk from evaluation.** Enables cheap re-analysis, protects against the (slow) IG/occlusion runs being repeated.
- **ED8 Cache-first LLM runner.** Idempotency and budget safety are non-negotiable when API calls cost money and rate limits exist. (This mirrors production practice — say so in interviews.)
- **ED9 ONNX scope = fusion head only.** Encoder export (esp. WavLM) has known operator friction; the latency story is told honestly with a mixed-runtime table. Time saved: ~1 week of debugging.
- **ED10 Gold transcripts for train/eval, ASR only in demo.** Keeps benchmark comparability (all MELD literature uses gold text); skew handled via R8/E8.
- **ED11 OmegaConf YAML configs, no CLI arg soup.** Every experiment is a config file diff; the experiment grid in §19 maps 1:1 to `configs/`.
- **ED12 Bootstrap CI + McNemar for model comparisons.** Single-number F1 deltas of <1 point on 2,610 test utterances are noise; the repo must never claim a win without a paired test. This discipline is itself a portfolio signal.

## 15. Rejected alternative designs (and why)

- **Concat-only fusion (original plan).** Kept only as baseline B3. Rejected as centerpiece: no interview surface, no intrinsic interpretability, no architecture to defend.
- **Full pivot to call-center analytics product (ASR + emotion + LLM coaching).** More "2026-flashy," but: no benchmark dataset → no credible numbers; scope explosion; weaker fit to both course rubrics. Rejected on risk.
- **Fine-tuning an audio-capable LLM (e.g., open audio-LLM SFT).** Compute-infeasible under A1; benchmark-prompting Gemini captures 70% of the narrative value at 2% of the cost.
- **MulT / full pairwise-crossmodal transformers (Tsai et al. style).** 3× the fusion parameters and complexity for marginal expected gain at this data scale; harder to explain cleanly. Our 2-block bidirectional design is the right size.
- **Graph-based ERC (DialogueGCN/DAG-ERC family).** Strong MELD numbers but a completely different project (dialogue-graph modeling), poor fit to the multimodal-XAI story, heavy implementation.
- **Video modality (MELD ships it).** Tripled preprocessing burden, face-detection dependency chain, minimal reported gains on MELD. Future work (§16).
- **SHAP as the second attribution family.** KernelSHAP on 500-frame audio inputs is computationally prohibitive; occlusion delivers the same "perturbation-family arbiter" role at tractable cost. (If a reviewer asks: occlusion IS the degenerate single-coalition case of SHAP; say that.)
- **Source separation / laugh-track removal.** Unbounded audio-engineering rabbit hole; replaced by honest failure-mode analysis (E9).
- **Speaker-embedding branch.** Speaker identity leaks label priors on MELD (main-cast emotion distributions differ); adding it inflates numbers for the wrong reason and pollutes the modality-dominance question. Deliberately excluded; mention in report limitations.
- **Streamlit instead of Gradio.** Gradio's native audio/mic components and HF Spaces path win for this use case.

## 16. Future extensions (ordered by value/effort)

1. **E-ext1 ASR-robustness training:** train on Whisper transcripts (or gold+ASR mix) to close the R8 skew → directly production-relevant.
2. **E-ext2 HF Spaces deployment** of the demo (free tier, CPU) → shareable link on resume.
3. **E-ext3 IEMOCAP transfer evaluation** (zero-shot + light fine-tune) → generalization story.
4. **E-ext4 Video third modality** with the same gated fusion pattern (architecture already generalizes: add a third stream + 3-way gate).
5. **E-ext5 Distillation:** distill the fused model into a single small audio-text student for edge latency.
6. **E-ext6 Calibration study** (ECE, temperature scaling) — pairs naturally with the XAI layer for a "trustworthy ML" narrative.
7. **E-ext7 LLM-generated natural-language explanations** grounded in the attribution outputs (bridge to owner's RAG/agent background).

## 17. Interview talking points (rehearse these; each maps to an artifact in the repo)

1. **Build-vs-buy with numbers:** "My fine-tuned 2.3M-trainable-param fusion model vs Gemini 2.5 Flash: here's the F1/cost/latency table. Here's where each wins and the break-even volume where self-hosting pays." (→ `bench/compare.py` output)
2. **Explanations you can trust:** "I didn't just generate saliency maps — I measured their faithfulness with deletion/insertion curves and cross-method agreement. Attention and gradients disagreed on X% of cases; occlusion arbitrated." (→ faithfulness tables)
3. **Intrinsic + post-hoc interpretability:** "The fusion gate makes modality dominance a first-class model output, and I validated it against two independent post-hoc estimates." (→ dominance agreement table)
4. **Negative-result maturity:** "Text dominates MELD globally — I predicted that in the README before training, and built the project to quantify *where* audio helps instead of overselling multimodality." 
5. **The mask-invariance test story:** "The scariest bug class in cross-attention is attending to padding — metrics stay plausible, attributions turn to garbage. I wrote an invariance test before training." (→ `tests/`; this is a *great* systems-thinking answer)
6. **Statistical discipline:** "No model-comparison claim in the repo lacks a bootstrap CI and a McNemar test; sub-point F1 deltas on 2.6K test items are noise."
7. **Performance engineering:** "Frozen encoders let me cache audio features to disk — 3–4× epoch speedup, which is what made 3-seed runs feasible on Colab."
8. **Production skew awareness:** "Training on gold transcripts but serving through Whisper is train/serve skew; I documented it, made transcripts user-editable in the demo, and measured the degradation." (→ E8)
9. **Cost-safe API engineering:** "The Gemini runner is cache-first, resumable, and budget-capped — the same idempotency pattern I used in my internship's agent pipelines."
10. **Scope discipline:** "I rejected video, source separation, and speaker embeddings deliberately — here's the rejected-alternatives section of my design doc." (→ this document, §15 — committing the blueprint itself is a talking point)

## 18. Evaluation methodology

**18.1 Task metrics.** Primary: weighted-F1 (MELD convention). Always reported alongside: macro-F1 (imbalance honesty), per-class F1, confusion matrix. Dev for selection, test touched only for final reported runs.

**18.2 Comparisons.** Fixed grid: B1 (text), B2 (audio), B3 (concat), M (xattn), M+LoRA (conditional), Gemini-audio+text, Gemini-text-only. All on identical test split A6.

**18.3 Seeds.** B1, B3, M: 3 seeds, report mean ± std. B2 and ablations: 1 seed (resource triage, documented).

**18.4 Statistical tests.** Bootstrap 95% CI on wF1 (1,000 utterance-level resamples). McNemar's test for paired model comparisons (M vs B3 is the headline pair). Claim threshold: CI-nonoverlap or p<0.05, else report "no significant difference."

**18.5 XAI evaluation (the course-critical part).**
- *Faithfulness:* comprehensiveness (delete top-{5,10,20,50}% attributed tokens/frames → mean drop in predicted-class prob; higher = more faithful) and sufficiency (keep only top-k% → prob retained; lower drop = better) per method × modality; summarized as AUC over the deletion fractions. Deletion = replace with pad-embedding (text) / zero post-proj frames (audio) — consistent with IG baselines by construction.
- *Agreement:* pairwise Spearman correlation of per-token/frame rankings between IG, attention, occlusion on the 500-sample occlusion subset; report distribution, not just mean.
- *Sanity checks:* parameter-randomization test (Adebayo-style) on IG — attributions must degrade under a randomized classifier head; if they don't, the method is broken for this model and that is reported.

**18.6 Modality dominance protocol.** Per utterance: (a) IG mass ratio = Σ|attr_audio| / (Σ|attr_audio| + Σ|attr_text-in-<cur>|) after per-modality max-normalization; (b) ablation dominance = Δlogit(pred class) when audio replaced by baseline vs when text replaced; (c) gate ratio g_a/(g_a+g_t). Aggregate per class with CIs; report 3-way agreement (pairwise Spearman + fraction of utterances where all three name the same dominant modality). SLP-course tie-in: correlate audio-dominance scores with prosodic features (pitch variance, energy — via librosa) to show *why* audio dominates where it does.

**18.7 LLM benchmark metrics.** Per §4.6: F1 family, invalid-output rate, $/1K utterances (from token+audio pricing at run time — record the price sheet snapshot in the report), p50/p95 latency. Fairness note in report: Gemini is zero-training but per-call-expensive; our model is train-once, cheap-serve; the break-even volume calculation goes in `compare.py` output.

**18.8 Serving metrics.** Latency table (§4.7): per-stage (ASR / encoders / fusion / IG-on-demand) on CPU and GPU, PyTorch vs ONNX where applicable, batch=1, median of 100 runs after 10 warmups.

## 19. Experiment roadmap (each is a config diff + a results-table row)

| ID | Question | Config delta | Milestone |
|---|---|---|---|
| E1 | Baseline grid B1/B2/B3 | `text_only/audio_only/concat.yaml` | M2 |
| E2 | Context window value | `xattn.yaml` × k ∈ {0,1,2,4} | M3 |
| E3 | Headline: M vs B3 | `xattn.yaml`, 3 seeds, McNemar | M3 |
| E4 | Audio encoder: WavLM vs Wav2Vec2 | checkpoint string swap | M3/M4 |
| E5 | Focal vs weighted CE | loss swap | M4 (if time) |
| E6 | LoRA stage-2 gain | `xattn_lora.yaml` | M4 (conditional) |
| E7 | Full XAI battery | §18.5–18.6 | M5 |
| E8 | Gold vs Whisper transcripts at test time | eval-time transcript source swap | M7 (stretch) |
| E9 | Failure-mode study: laugh-track/overlap clips; confusion pairs | error-analysis notebook | M8 |
| E10 | Gemini two conditions + break-even | prompt/audio-flag | M6 |

Priority if time collapses (R3): E1 → E3 → E7 → E10 → E2 → E4 → rest.

## 20. Deployment plan

1. **Weights:** best checkpoint (M, seed-averaged selection by dev wF1) → HF Hub model repo with the model card (`docs/MODEL_CARD.md` mirrored). Card must include: intended use, MELD-domain limitation, class-imbalance behavior, train/serve ASR skew, laugh-track failure mode, and an explicit "not for surveillance/consequential decisions about individuals" use restriction — affect recognition is a sensitive category and saying so is both ethically required and a maturity signal.
2. **Demo:** Gradio app runnable via `docker run -p 7860:7860 emorec-xmm`; CPU-only image; weights pulled from Hub at container start (keeps image slim). HF Spaces deployment is E-ext2 (post-scope, cheap).
3. **Inference artifact:** ONNX fusion head + parity test + `scripts/bench_latency.py` producing the §18.8 table committed to README.
4. **Repro contract in README:** ≤5 commands from clone to (a) demo running with downloaded weights, (b) headline eval reproduced from cached predictions, (c) full retrain (documented as GPU-required).
5. **CI:** lint + unit tests on every push (CPU, <5 min); training/XAI jobs are never in CI.

---

## Appendix A — Handover notes to the implementing engineer

1. Read §0, §12 R6, and §11 before writing any code. The mask-invariance test is the hill to die on.
2. The audit (M1) exists to falsify this document's assumptions early. If audit contradicts a number in §4 (token lengths, durations, class counts), write an ADR and adjust the config — do not silently edit this file.
3. Expected pain ranking, from experience: MELD audio extraction (R4) > padding masks (R6) > Captum LayerIG wiring for dual-input models > everything else. Budget accordingly.
4. For Captum on a two-input model: wrap `MultimodalERC` in a forward function that takes (text_embeds, audio_proj) as the attributed inputs and closes over masks/ids; attribute each input in a separate LayerIG pass with the other held fixed. Do not try to attribute both simultaneously through one call.
5. When numbers disappoint, the deliverable is the analysis, not the number. The project was designed so that every plausible empirical outcome is publishable. Hold that line.
6. The owner's stated working preference: direct honest feedback, root-cause understanding, readable code over clever code. Match it.

*End of blueprint. Frozen v1.0.*
