# PROJECT_CONTEXT.md

**Why this document exists:** this file eliminates the need to re-explain the project's purpose, constraints, and success criteria in any future chat or coding session. An AI assistant (or human) reading only this file should understand *why* every other document says what it says. It is the "who/why" file; `ARCHITECTURE.md` is the "what"; `IMPLEMENTATION_GUIDE.md` is the "how"; `CLAUDE.md` is the "how we work."

## The one-paragraph brief

Owner is a 4th-year CS (AI/ML) undergraduate at VIT (graduating 2027), entering Semester VII, building this across ~12 weeks at ~10–12 h/week alongside a medical-AI capstone, competitive-programming exam prep (TopCoder, placement-critical), and campus placements. The project must simultaneously (1) satisfy a **Speech and Language Processing** course, (2) satisfy an **Explainable AI** course, and (3) serve as a flagship **AI-engineering portfolio piece** for 2026–27 internship/placement interviews. When goals conflict: portfolio quality wins, provided course rubrics stay satisfied (they are strict subsets of the full scope).

## Why this project (career logic)

- Owner's resume already carries a strong **LLM orchestration** story (internship: multi-agent LangGraph system, Gemini 2.5 Flash, deterministic verification gates, production handover doc). What it lacks is evidence of **training, evaluating, and shipping a specialist model**. This project supplies exactly that missing half.
- The Gemini benchmark deliberately *bridges* both halves: "knows when to train, knows when to prompt, can measure the difference" is the portfolio's single coherent narrative. Protect that narrative in every writeup.
- Target roles: AI Engineer / LLM Engineer / ML Engineer internships and placements, 2026–27 cycle in India + remote.

## Course-mapping (what each course grades)

- **SLP course draws on:** dialogue-context modeling (k-utterance windows, speaker conditioning), ASR integration (Whisper in the demo, E8 gold-vs-ASR study), prosodic feature correlation with audio dominance (§18.6 of blueprint), the MELD conversational structure itself.
- **XAI course draws on:** IG/attention/occlusion attribution, faithfulness evaluation (comprehensiveness/sufficiency), sanity checks (parameter randomization), modality-dominance quantification with 3-way agreement, the model card's transparency sections.
- Both course reports are extracted from `docs/REPORT.md` at M8; never write course material that isn't also repo material.

## Hard constraints

1. **Compute (assumption A1):** design for a single intermittent 16 GB GPU (Colab Pro / Kaggle). Everything must complete in Stage 1 (frozen encoders). LoRA is optional behind the M4 gate. Validate A1 in week 1 with a 100-step training probe.
2. **Budget:** Gemini benchmark hard-capped at ~US$15 (cache-first runner, stratified-subsample fallback).
3. **Time:** the R3 cut-order in `ROADMAP.md` is the pre-agreed answer to schedule collapse. Never cut: baselines, faithfulness eval, demo, README. Cut first: LoRA, then occlusion, then ONNX, then Gemini text-only condition, then extra ablations.
4. **Scope freeze:** architecture frozen at BLUEPRINT v1.0 (2026-07-19). Changes require an ADR in `docs/adr/` (template in `DECISIONS.md`).

## Success criteria (in priority order)

1. A stranger reproduces the headline numbers from the README in ≤5 commands.
2. Every model-comparison claim carries a bootstrap CI and (for pairs) a McNemar test.
3. The XAI layer has *evaluated* faithfulness, not just pretty saliency maps.
4. The demo works end-to-end from a microphone.
5. Interview talking points 1–10 (BLUEPRINT §17) are each backed by a concrete artifact.
6. Both course rubrics satisfied from REPORT.md.

## Pre-registered scientific stance

Text will likely dominate MELD globally; audio should help on anger/surprise and ambiguous short utterances. **Every plausible empirical outcome is a reportable finding.** Negative results are written up, not hidden. This stance is itself a differentiator — keep it visible in the README from day one.

## Owner working preferences (match these)

Direct, honest feedback over reassurance. Root-cause understanding over surface fixes. Readable code over clever code. English, C++/Python background, strong DSA, comfortable with transformers/HF, has production LangGraph + FastAPI experience, first serious Captum/ONNX project (budget learning time there).
