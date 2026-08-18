# FUTURE_WORK.md

**Why this document exists:** the parking lot that protects scope. Anything valuable-but-out-of-scope lands here instead of leaking into the milestones. Ordered by value/effort; each entry says what it would prove and what it costs. Also the source of "how would you extend this?" interview answers.

1. **ASR-robustness training** (train on Whisper or gold+ASR-mixed transcripts). Proves: closing a real train/serve skew — directly production-relevant. Cost: ~1 week (transcribe train set once, retrain). Natural first extension; pairs with E8 measurements.
2. **HF Spaces deployment** of the demo (CPU free tier). Proves: shareable live artifact for the resume. Cost: ~half a day (Docker already exists).
3. **IEMOCAP transfer** (zero-shot + light fine-tune). Proves: generalization beyond MELD; addresses limitation #5. Cost: ~1–2 weeks incl. license/data wrangling.
4. **Video third modality** with the same gated-fusion pattern (3-stream, 3-way gate — architecture already generalizes). Proves: the fusion design scales in modality count. Cost: high (face pipeline); rejected from scope as RA6 for good reason.
5. **Distillation to a single small student** (audio+text → one compact encoder). Proves: edge-latency engineering. Cost: ~2 weeks.
6. **Calibration study** (ECE, temperature scaling, selective prediction/abstention). Proves: trustworthy-ML depth; pairs elegantly with the XAI layer. Cost: ~3–4 days. Addresses limitation #10.
7. **LLM-generated natural-language explanations grounded in attributions** (feed IG spans + gate + prosody stats to an LLM that writes the "why" in prose, with citations back to tokens/frames). Proves: bridges this project to the owner's RAG/agent background; the explainability story becomes user-facing. Cost: ~1 week. Highest interview-narrative value of the list.
8. **Speaker-leakage quantification** (mask speaker names in context; measure Δ). Proves: rigor on limitation #6. Cost: 1–2 days; cheap paper-cut worth doing early if time appears.
9. **Streaming/incremental inference** (chunked audio, partial-utterance predictions). Proves: real-time systems capability. Cost: high; architectural changes to pooling.
10. **Multilingual extension** (XLM-R + multilingual audio encoder on a non-English ERC set). Proves: breadth. Cost: high; new data problems.
