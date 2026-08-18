# MODEL_CARD.md — emorec-xmm (gated cross-attention ERC on MELD)

**Why this document exists:** the transparency artifact for the released model — intended use, training data, measured performance, limitations, and use restrictions. It is a portfolio signal (responsible-AI maturity) and an XAI-course deliverable. Rule: only measured numbers enter this card; sections marked *(fill at MX)* stay empty until then.

## Model details
- **Architecture:** frozen RoBERTa-base (text, k=2 dialogue context, speaker tags) + frozen WavLM-base-plus (16 kHz audio) → d=256 projections → 2× bidirectional gated cross-attention (8 heads) → attention pooling → per-modality scalar gate → MLP → 7 classes. ~2.3 M trainable parameters (Stage 1). *(If LoRA stage shipped: note here — fill at M4.)*
- **Task:** 7-class emotion recognition in conversation (neutral, joy, sadness, anger, surprise, fear, disgust).
- **Training data:** MELD (Poria et al., 2019) — 13.7K utterances from the TV series *Friends*; standard splits, unmodified. English, scripted, acted, laugh-track-contaminated audio.
- **Version / date / license:** v1.0 · *(fill at M7)* · MIT (code); MELD's own license governs data.

## Intended use
Research and portfolio demonstration of multimodal fusion + evaluated explainability;education/coursework; benchmarking against LLM baselines. Interactive demo for illustration.

## Out-of-scope / use restrictions
**Not for:** surveillance, employment/hiring decisions, medical or psychological assessment, or any consequential decision about individuals. Affect recognition is a sensitive, error-prone capability; this model is trained on scripted TV dialogue and does not generalize to real-world consequential contexts. Not evaluated on: non-English speech, children's speech, telephony-band audio, non-North-American accents beyond the MELD cast.

## Performance *(fill at M2/M3/M6 — measured only)*
- Headline table (wF1/mF1/per-class, 3 seeds mean±std): –
- Comparison vs baselines and Gemini 2.5 Flash (incl. cost/latency): –

## Explainability *(fill at M5)*
Attributions available per prediction: IG (text tokens, audio 0.5 s bins), cross-attention maps, intrinsic modality gate. Faithfulness measured (comprehensiveness/sufficiency AUC): – . Method-agreement summary: – . **Explanations are model-behavior descriptions, not ground truth about human emotion.**

## Known limitations (summary — full list in KNOWN_LIMITATIONS.md)
Text-dominant dataset; noisy audio (laugh track, overlap, music) with characterized failure mode *(E9, fill at M8)*; severe class imbalance (fear/disgust weakest — see per-class table); train/serve ASR skew (trained on gold transcripts, demo uses Whisper; degradation measured in E8 if run: –); label noise inherent to MELD annotation; speaker/cast priors in the data (speaker features deliberately excluded from the model to avoid leakage).

## Ethical considerations
Emotion labels are annotator judgments of acted performances, not ground truth of internal states. Misclassification risk is unevenly distributed across classes (minority emotions worst). The use restrictions above are load-bearing, not boilerplate.
