# DECISIONS.md

**Why this document exists:** the immutable record of engineering decisions (ED) and rejected alternatives (RA), plus the ADR process for any post-freeze change. Its purpose is to prevent silent reversals and re-litigation of settled questions — by humans or by AI assistants in future sessions. If you find yourself wanting to do something this file rejects, write an ADR first.

## Frozen engineering decisions (ED1–ED12) — from BLUEPRINT §14, binding

- **ED1** Frozen encoders Stage 1; LoRA optional behind M4 gate. (Compute robustness; 3–4× iteration speed via cache; clean scientific comparison.)
- **ED2** WavLM-base-plus default audio encoder; Wav2Vec2-base-960h as ablation E4. (Better paralinguistic prior; zero-cost swap.)
- **ED3** d=256 shared fusion space, 8 heads, 2 blocks. Fixed; not an experiment.
- **ED4** Per-modality scalar gate (not per-dimension): interpretability is the point — the gate is dominance signal #3.
- **ED5** Audio truncation keeps the END of utterances (utterance-final prosody).
- **ED6** Weighted CE (inv-sqrt, mean-normalized) default; focal γ=2 is ablation E5 only.
- **ED7** Attribution artifacts decoupled on disk from their evaluation.
- **ED8** Cache-first, resumable, budget-capped LLM runner. Non-negotiable for money-touching code.
- **ED9** ONNX scope = fusion head only. Encoder export is a known time sink — forbidden.
- **ED10** Gold transcripts for train/eval; ASR only in demo path; skew documented + optionally measured (E8).
- **ED11** OmegaConf YAML configs; experiments are config diffs; no hardcoded hyperparameters.
- **ED12** Bootstrap CI + McNemar required for every comparison claim. No exceptions, including in the README.

## Rejected alternatives (RA1–RA10) — from BLUEPRINT §15, do not re-propose without new information

RA1 concat-as-centerpiece (kept only as B3) · RA2 call-center product pivot (no benchmark, scope explosion) · RA3 fine-tuning an audio LLM (compute-infeasible) · RA4 MulT-style full pairwise crossmodal transformer (oversized for 10K samples) · RA5 graph-based ERC (different project) · RA6 video modality (cost≫gain; future work) · RA7 KernelSHAP (prohibitive on 500-frame audio; occlusion is the tractable perturbation-family arbiter) · RA8 source separation / laugh-track removal (unbounded rabbit hole; replaced by E9 failure-mode study) · RA9 speaker-embedding branch (label leakage via cast priors pollutes the dominance question) · RA10 Streamlit (Gradio wins on audio/mic + Spaces path).

## ADR process (for any post-freeze change)

File `docs/adr/NNNN-short-title.md`, numbered sequentially, using:

```
# ADR-NNNN: <title>
Date: YYYY-MM-DD
Status: accepted | superseded-by-NNNN
## Context — what NEW information invalidated the frozen decision?
## Decision — what changes, precisely (configs/files touched)
## Consequences — cost, risk, what downstream docs were updated
## Blueprint delta — which BLUEPRINT/ARCHITECTURE sections this supersedes
```

Rules: ADRs change decisions; they never rewrite history (BLUEPRINT.md and this file's ED/RA sections are edited only to add "superseded by ADR-NNNN" pointers). An AI assistant may DRAFT an ADR but the owner accepts it before implementation proceeds.

## ADR index

*(empty — no post-freeze changes yet)*
