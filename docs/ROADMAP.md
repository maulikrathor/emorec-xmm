# ROADMAP.md

**Why this document exists:** the living schedule. Every session starts by locating the current milestone here. Exit gates are binding: a milestone is not done until its gate is demonstrably met. The milestone log at the bottom is appended (never edited) as work completes — it is the project's flight recorder.

Planning basis: 12 weeks, ~10–12 h/week, competing with capstone + TopCoder + placements. Buffers are real; treat them as load-bearing.

## Milestones and exit gates

| MS | Weeks | Scope | EXIT GATE (binding) |
|---|---|---|---|
| M0 | 1 | Scaffold, CI, configs, seeding, download; **A1 compute probe** (100-step B1 run, record it/s) | CI green; MELD checksums verified; compute probe result logged below |
| M1 | 2 | Preprocess, manifests, dataset/collator, audit | Audit report committed; max_len-160 claim verified or ADR'd; corrupt-clip counts logged |
| M2 | 3–4 | Metrics/losses/trainer, audio cache, B1→B2→B3 | **B1 dev wF1 ≥ 60 AND B2 ≥ 38** (else debug data pipeline — do NOT proceed); 3-seed B1/B3 done |
| M3 | 5 | Fusion + INV1 test, train M (3 seeds), E2 k-ablation | M-vs-B3 table with bootstrap CI + McNemar committed (any outcome accepted) |
| M4 | 6 | **Buffer + decision gate** | Decision logged: LoRA (only if on schedule AND compute ≥ T4/6 h) or permanently skipped via ADR |
| M5 | 7–8 | Full XAI battery (IG, attention, occlusion, faithfulness, dominance) | Faithfulness AUC table (≥2 methods × 2 modalities); 3-way dominance agreement table; 6 case-study figures |
| M6 | 9 | Gemini benchmark, both conditions | Full comparison table (F1/cost/latency/invalid-rate); response cache preserved |
| M7 | 10–11 | Demo, ONNX+parity+latency, Docker, model card, README, HF Hub upload | `docker run` → working demo; stranger-reproducibility (≤5 commands) verified by actually doing it |
| M8 | 12 | REPORT.md, error-analysis notebook, interview-prep pass, repo public | Both course deliverables extractable; §17 talking points each mapped to an artifact |

## Cut order under schedule collapse (R3 — pre-agreed, do not renegotiate under stress)

Cut in this order: (1) LoRA/E6 → (2) occlusion (keep IG+attention; faithfulness runs on 2 methods) → (3) ONNX → (4) Gemini text-only condition → (5) E4/E5/E8 ablations.
**Never cut:** baselines with gates, INV1 test, faithfulness eval, demo, README, statistics discipline.
Minimum viable portfolio = M0–M3 + M5 + M7-lite (demo, no ONNX).

## Standing weekly rhythm

Session 1: build (per IMPLEMENTATION_GUIDE order). Session 2: build + tests. Session 3 (short): update EXPERIMENTS.md rows, append milestone log, sanity-check gate progress.

## Milestone log (append-only)

```
2026-07-19  Blueprint v1.0 frozen. Docs bundle generated. Implementation not started.
```
