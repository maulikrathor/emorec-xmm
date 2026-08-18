# EXPERIMENTS.md

**Why this document exists:** the experiment registry and status board. Every experiment is a config diff (ED11) plus one row here. This file is how a future session instantly knows what has been run, with what result, and what remains. Update the status column in the same commit that produces the result.

## Registry

| ID | Question | Config / delta | Milestone | Priority | Status | Result pointer |
|---|---|---|---|---|---|---|
| E1 | Baseline grid B1/B2/B3 | `text_only.yaml` / `audio_only.yaml` / `concat.yaml` | M2 | P0 | not started | – |
| E2 | Dialogue context value | `xattn.yaml` × `data.context_k ∈ {0,1,2,4}` | M3 | P1 | not started | – |
| E3 | **Headline:** M vs B3 (CI + McNemar) | `xattn.yaml`, 3 seeds | M3 | P0 | not started | – |
| E4 | WavLM vs Wav2Vec2 | `model.audio_ckpt` swap | M3/M4 | P2 | not started | – |
| E5 | Focal vs weighted CE | `loss.name=focal` | M4 | P3 | not started | – |
| E6 | LoRA Stage-2 gain | `xattn_lora.yaml` | M4 (conditional) | P3 | gated (M4 decision) | – |
| E7 | Full XAI battery (faithfulness, agreement, dominance) | scripts, per EVALUATION §5–6 | M5 | P0 | not started | – |
| E8 | Gold vs Whisper transcripts at eval | eval-time transcript source | M7 stretch | P3 | not started | – |
| E9 | Failure modes: laugh-track clips, confusion pairs | `notebooks/03_error_analysis` | M8 | P1 | not started | – |
| E10 | Gemini: audio+text vs text-only, cost, latency, break-even | `run_gemini.py`, prompt v1 | M6 | P0 | not started | – |

Priority meaning under time pressure (R3): run all P0, then P1; P2/P3 are the first casualties (cut order in ROADMAP.md).

## Per-experiment protocol
1. Create/verify config diff. 2. Run with logged seed(s), git SHA, config dump. 3. Compute metrics via `emorec.training.metrics` only. 4. Statistics per EVALUATION §4 for any comparison. 5. Fill the row: status=done, pointer to `artifacts/runs/<run_id>/` and the results table location. 6. If the result contradicts a blueprint assumption → note it and check whether an ADR is needed.

## Pre-registered expectations (written before any run; do not edit after)
- E1: B1 ≈ 60–64 wF1; B2 ≈ 38–45; B3 ≈ B1 + 0–2.
- E3: M ≥ B3 by 0.5–2 wF1; significance uncertain — a null result is acceptable and reportable.
- E2: k=2 > k=0 by ≥ 2 wF1; k=4 ≈ k=2 (diminishing).
- E7: text dominates globally (~70–90% of utterances); audio-dominant pockets in anger/surprise; IG–occlusion agreement > attention–occlusion agreement.
- E10: Gemini audio+text within a few F1 of M, at orders-of-magnitude higher per-call cost; Gemini text-only ≈ Gemini audio+text (LLM extracts little from noisy MELD audio) — genuinely uncertain, interesting either way.
