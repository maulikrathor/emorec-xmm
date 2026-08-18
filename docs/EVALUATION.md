# EVALUATION.md

**Why this document exists:** the single authoritative statement of how everything is measured — task metrics, statistics, XAI faithfulness, modality dominance, LLM benchmark, and serving latency. Any number reported anywhere in the repo must be produced by the procedures here. If a procedure needs changing, that's an ADR.

## 1. Task metrics
Primary: **weighted-F1** (MELD convention, leaderboard-comparable). Always co-reported: macro-F1, per-class F1, confusion matrix. Dev = model selection only; test touched only for final reported runs. Splits: MELD standard, unmodified (A6), forever.

## 2. Model comparison grid
B1 text-only · B2 audio-only · B3 concat · M gated-xattn · [M+LoRA conditional] · Gemini audio+text · Gemini text-only. Identical test split for all.

## 3. Seeds & aggregation
B1, B3, M: seeds {13, 42, 2026}, report mean ± std. B2 + ablations: 1 seed (documented triage). Headline README numbers = 3-seed means.

## 4. Statistics (ED12 — binding)
- Bootstrap 95% CI on wF1: 1,000 utterance-level resamples, percentile method.
- McNemar exact test on paired predictions for any A-vs-B claim (headline pair: M vs B3).
- Claim rule: CI non-overlap or p<0.05 ⇒ may claim a difference; otherwise the repo says "no significant difference." Sub-point F1 deltas on 2,610 test items are noise — say so.

## 5. XAI faithfulness protocol
- **Comprehensiveness**: delete top-{5,10,20,50}% attributed tokens/frames → mean drop in predicted-class probability (higher = more faithful). **Sufficiency**: retain only top-k% → probability retained (smaller drop = better). Summary: AUC over the four fractions, per method × modality.
- Deletion operator: pad-token embedding (text) / zero post-proj frame (audio) — deliberately consistent with IG baselines; state this in the report.
- **Agreement**: pairwise Spearman of per-token/frame rankings (IG vs attention vs occlusion) on the 500-sample stratified occlusion subset; report the distribution (violin/percentiles), not just the mean.
- **Sanity**: parameter-randomization (Adebayo-style) on the classifier head — attributions must degrade; a non-degrading method is reported as broken for this model.

## 6. Modality dominance protocol (per utterance → per class)
Three independent signals: (a) **IG mass ratio** Σ|attr_audio| / (Σ|attr_audio|+Σ|attr_text-in-<cur>|) after per-modality max-normalization; (b) **ablation delta**: Δlogit(pred) replacing audio with baseline vs replacing text; (c) **gate ratio** g_a/(g_a+g_t). Aggregate per class with bootstrap CIs. Agreement: pairwise Spearman + fraction of utterances where all three name the same dominant modality. SLP tie-in: correlate audio-dominance with librosa prosodic features (pitch variance, energy) to explain WHERE audio dominates.

## 7. LLM benchmark metrics
wF1/mF1/per-class, invalid-output rate, $/1K utterances (from token+audio pricing; snapshot the price sheet at run time into the report), p50/p95 wall latency. Plus break-even served-volume calculation (API cost vs self-host amortization) in `compare.py`. Fairness framing: zero-training vs train-once-cheap-serve — both sides stated.

## 8. Serving latency
Per-stage table (Whisper / encoders / fusion / IG-on-demand), CPU and GPU, PyTorch vs ONNX where applicable, batch 1, 10 warmups, median of 100 runs.

## 9. Reporting rules
Empty is honest, fake is fatal: `–` until measured. Every table states seeds, split, and date. Negative results reported with the same prominence as positive ones.
