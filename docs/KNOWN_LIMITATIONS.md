# KNOWN_LIMITATIONS.md

**Why this document exists:** the honest, versioned list of what this system cannot do and why — pre-registered where possible, updated with evidence as experiments land. It feeds the model card, the report's limitations section, and interview answers (stating limitations crisply is a maturity signal). Never delete an entry; mark it "confirmed (E#)" or "not observed" as evidence arrives.

## Dataset-inherited
1. **Text dominance of MELD.** Pre-registered expectation: text carries most signal; multimodal gains small and class-concentrated. Status: pre-registered, pending E1/E3/E7.
2. **Noisy audio channel.** Laugh track, overlapping speakers, background music; no source separation by design (RA8). Failure mode to be characterized in E9. Status: pending.
3. **Severe class imbalance.** Fear/disgust have few hundred train samples; per-class F1 for these will be weak regardless of weighting. Status: pre-registered, pending E1.
4. **Label noise.** MELD emotion labels are annotator judgments of acted performances; inter-annotator disagreement is documented in the original paper. Taken as ground truth by policy (A7); discussed, not fixed.
5. **Domain narrowness.** Scripted US sitcom English, small recurring cast, 1990s-2000s conversational register. No claim of generalization beyond it (see MODEL_CARD restrictions).
6. **Speaker priors.** Main-cast emotion distributions differ; speaker features are deliberately excluded (RA9) but names appear in text context — residual leakage is possible. Status: acknowledged; quantification is future work.

## Model/architecture
7. **Frozen encoders (Stage 1)** cap achievable F1 below full-fine-tune SOTA; accepted trade (ED1). LoRA may close part of the gap (E6, conditional).
8. **k=0 at demo time**: live inputs get no dialogue context; demo accuracy < benchmark accuracy by construction. Stated in the UI.
9. **10 s audio cap** truncates ~3% of utterances (start-truncated by ED5); long-utterance behavior slightly degraded.
10. **7-class closed set**: no neutral-vs-ambiguous abstention, no mixed emotions, no intensity. Calibration unstudied (FUTURE_WORK E-ext6).

## Explainability
11. **Post-hoc attributions describe model behavior, not causal ground truth about emotion.** Faithfulness is measured, not assumed — and faithfulness ≠ human-plausibility.
12. **Attention-as-explanation is known-unreliable**; it is included precisely to measure its disagreement with IG/occlusion, not as a trusted method.
13. **IG-audio interpretability risk** (R7): frame-level attribution may be diffuse even post-projection; occlusion arbitrates; a negative faithfulness result for IG-audio is reported as a finding.
14. **Deletion-metric circularity caveat**: deletion operators match IG baselines by construction (documented choice in EVALUATION §5), which mildly favors IG in faithfulness comparisons; stated openly in the report.

## Serving/benchmark
15. **Train/serve ASR skew** (ED10): trained on gold transcripts, served through Whisper; degradation measured only if E8 runs.
16. **Latency numbers are single-machine, batch-1** — indicative, not a throughput study.
17. **Gemini benchmark is a point-in-time snapshot** (model version, pricing, prompt v1); results dated and price-sheet-snapshotted, not claimed as durable truths.
