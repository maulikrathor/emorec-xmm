# emorec-xmm — Explainable Multimodal Emotion Recognition (Speech + Text)

> **Status: pre-implementation.** This README is the v0 committed before code exists. Sections marked `<!-- RESULTS -->` are filled at milestones M2–M7. Do not fill them with placeholder numbers — empty is honest, fake is fatal.

**Why this document exists:** the README is the product surface of the repo — the first (often only) thing recruiters, interviewers, and other engineers read. It must let a stranger (a) understand the project in 60 seconds, (b) run the demo in ≤5 commands, (c) find the headline results and the design doc. Everything else lives in `docs/`.

---

## What this is

A complete, deployable emotion-recognition-in-conversation system for the MELD dataset (7 emotions) that:

1. Fuses a **RoBERTa** text branch (with dialogue context) and a **WavLM** audio branch through **gated cross-modal attention**.
2. Doesn't just *generate* explanations — it **evaluates** them: Integrated Gradients, attention attribution, and occlusion, scored with faithfulness (deletion/insertion) curves and cross-method agreement.
3. Quantifies **modality dominance** (does audio or text drive each prediction?) three independent ways — including an intrinsic learned gate — and checks whether they agree.
4. Benchmarks the trained specialist against **Gemini 2.5 Flash** (native audio, few-shot) on **accuracy, cost per 1K utterances, and latency** — the 2026 build-vs-buy question, with numbers.
5. Ships like a product: Gradio demo (microphone → Whisper → prediction + live attribution view), ONNX-exported inference path, Docker image, model card, CI.

**Stated up front (pre-registered hypothesis):** on MELD, text is expected to dominate globally; audio is expected to help on prosody-heavy classes (anger, surprise) and textually ambiguous utterances. This project is designed so that *quantifying that pattern* — whichever way it comes out — is the deliverable.

## Quickstart

```bash
# 1. install
pip install -e ".[serve]"
# 2. run the demo with released weights (pulled from HF Hub)
python scripts/demo.py
# 3. reproduce headline eval from cached predictions
python scripts/evaluate.py --config configs/xattn.yaml --from-cache
# 4. full data prep (one-time, ~needs MELD download)
python scripts/prepare_data.py
# 5. retrain (GPU required)
python scripts/train.py --config configs/xattn.yaml
```

## Results

<!-- RESULTS: filled at M2 (baselines), M3 (fusion), M5 (XAI), M6 (Gemini) -->

| Model | Weighted F1 | Macro F1 | Cost / 1K utt | p50 latency |
|---|---|---|---|---|
| B1 text-only (RoBERTa + context) | – | – | – | – |
| B2 audio-only (WavLM) | – | – | – | – |
| B3 concat fusion | – | – | – | – |
| **M gated cross-attention (ours)** | – | – | – | – |
| Gemini 2.5 Flash (audio+text, few-shot) | – | – | – | – |
| Gemini 2.5 Flash (text-only) | – | – | – | – |

Faithfulness and modality-dominance tables: see `docs/EVALUATION.md` outputs in `artifacts/reports/` after M5.

## Repository map

```
configs/        every experiment = one YAML diff
src/emorec/     data / models / training / xai / bench / serve / utils
scripts/        thin CLIs (no logic lives here)
tests/          incl. the MANDATORY mask-invariance test
docs/           BLUEPRINT (frozen design), all engineering docs
notebooks/      audit, results analysis, error analysis (no load-bearing logic)
```

## Design documents

The full frozen engineering blueprint is at [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md). Start there. The rejected-alternatives section (§15) explains why this repo looks the way it does.

## License & intended use

MIT. **Use restriction (see `docs/MODEL_CARD.md`):** research/portfolio system trained on scripted TV dialogue; not suitable for surveillance, hiring, or any consequential decision about individuals.
