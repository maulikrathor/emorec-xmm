# TRIBAL_KNOWLEDGE.md — Final transfer of undocumented knowledge

**Why this document exists:** everything the departing tech lead knows that hasn't landed in any other document — implementation folklore, design intuition, and advice about how this project actually succeeds. This is the last communication from the original architect. §Session Notes at the bottom is append-only and becomes the NEW tribal knowledge as implementation proceeds.

## 1. Implementation folklore (specific, hard-won, not in the guides)

1. **The wav2vec2-family attention-mask trap (worth repeating because it's silent):** `facebook/wav2vec2-base-960h` was pretrained without attention masks and its feature extractor defaults `return_attention_mask=False`; `microsoft/wavlm-base-plus` expects masks. If you switch encoders for E4 and blindly reuse extractor defaults, one config runs maskless and your E4 comparison is corrupted while both runs "work." Always construct masks explicitly from raw lengths; never trust extractor defaults.
2. **`_get_feature_vector_attention_mask` is the only correct way** to get frame-level masks — the conv feature extractor's downsampling arithmetic (stride stack 5-2-2-2-2-2-2) is easy to get off-by-one by hand, and an off-by-one frame mask passes every shape test while leaking one pad frame into attention.
3. **MELD CSVs contain smart quotes and stray Windows-1252 bytes** in utterance text. If tokenizer inputs look subtly wrong, it's encoding, not your code.
4. **MELD utt/dialogue IDs are per-split** — `dia0_utt0` exists in every split. Namespace utt_ids with the split (`train/dia0_utt0`) in manifests and every artifact path, or caches will collide the day you cache dev.
5. **pyloudnorm raises on very short/silent audio** and MELD has both. The <0.4 s guard in T-04 is not paranoia; you will hit it in the first thousand clips.
6. **Captum + frozen embeddings:** the input embeds you pass to `forward_from_embeds` must have `requires_grad_(True)` set INSIDE the attribution wrapper — the embedding parameter being frozen is irrelevant, but people conflate the two and get all-zero attributions with no error.
7. **`nn.MultiheadAttention` returns averaged attention weights by default** (`average_attn_weights=True`); if you want per-head maps for J-05 aggregation experiments, set it False explicitly at extraction time.
8. **matplotlib in headless environments:** set `Agg` backend at module import in viz.py, not in notebooks, or the demo server will die trying to open a display.
9. **faster-whisper first call is slow** (model load + warmup). Load at app start, not per request, and warm it with the fixture wav — otherwise the first demo interaction looks broken.
10. **Colab/Kaggle survival:** persist `artifacts/` to Drive/Datasets between sessions; the audio cache (T-08) is the thing you must never lose (hours to rebuild). Copy checkpoints out the moment a run finishes; assume the VM dies at the worst moment. Keep `pip install -e .` in the first cell — session resets silently un-install you.
11. **Gradient accumulation + early epochs:** with accum 4 and batch 8, one epoch of MELD train ≈ 312 optimizer steps; warmup 10% ≈ 31 steps — sanity-check these printed numbers on run 1; a 4× discrepancy means the classic scheduler bug (T-10) shipped anyway.
12. **The dev set has 1,109 utterances; per-epoch full-dev eval is cheap.** Never subsample dev to save time — early stopping on a noisy subsample is how you select a worse model while feeling faster.

## 2. Design intuition (the "why" behind the shape of the project, one level deeper than the blueprint says)

13. **The gate is the project's soul.** Concat asks "can two vectors help?"; the gate asks "how much does the model *choose* to trust each sense, per utterance?" — and makes that choice inspectable, trainable, and cross-validatable against post-hoc methods. Almost every distinctive analysis (dominance triangulation, gate-vs-IG agreement, per-class trust profiles) flows from that one scalar pair. If scope collapses catastrophically, protect the gate analyses above almost everything else.
14. **Why triangulation everywhere:** single-method XAI results are unfalsifiable decoration; two methods that agree are evidence; three signals with measured agreement is a methodology. The same pattern (never trust one estimator of a latent quantity) is the transferable engineering lesson of this project — name it in interviews.
15. **Why the pre-registered expectations file matters more than it looks:** it converts every possible result into a story about calibration ("here's what I predicted, here's what happened, here's why the delta"). That's the difference between a student reporting numbers and an engineer demonstrating judgment. Guard EXPERIMENTS.md's no-edit-after rule jealously.
16. **The honest-negative-result posture is a strategy, not a consolation.** Interviewers see hundreds of repos claiming wins; a repo that predicted, found, quantified, and explained a *limit* of multimodality is memorable. Fight the instinct to spin.
17. **Why frozen encoders is the right call even if compute appears:** beyond cost, it makes the E-grid a controlled experiment (identical representations everywhere) and makes the audio cache possible, which makes 3 seeds possible, which makes the statistics honest. Full fine-tuning would trade scientific cleanliness for a couple of F1 — a bad trade for THIS project's goals. LoRA behind a gate is the compromise that keeps the story clean.

## 3. Process advice for the humans

18. **The gates exist to stop sunk-cost momentum.** B1 < 60 and the temptation will be "train longer" — but the base-rate cause is a data bug, and training longer on a data bug wastes a week. Trust the debug order in T-12.
19. **Timebox debugging at 90 minutes before writing the problem down** (rubber-duck into TRIBAL §Session Notes). Half of the hard bugs in this stack (masks, encodings, caches) yield to being described precisely.
20. **Do T-24's mock tests on a day you feel impatient, and the real API run on a day you don't.** Money-touching code and impatience are a bad pairing.
21. **Record the demo GIF early in M7, not last** — it forces the demo to actually work mid-milestone and it's the single highest-ROI artifact for the resume.
22. **For the owner specifically:** your instinct toward root-cause depth is right for this repo, but placements season punishes perfectionism. The cut order exists so that "smaller but impeccable" is always available. A finished M0–M3+M5+M7-lite with honest stats beats an unfinished everything.
23. **Interview prep is an M8 deliverable, not an afterthought:** rehearse the ten talking points against the actual artifacts, out loud, once. The mask-invariance story and the build-vs-buy table are the two openers; lead with whichever fits the interviewer (systems vs product).

## 4. Things that will tempt you and must be resisted (final warnings)

- Adding "just one more" architecture variant after E3 (J-08 exists for a reason).
- Encoder ONNX export "since we're at it" (ED9 — it eats a week).
- Filling a README cell with a number you remember instead of one you can trace.
- Letting the demo drift from `pipeline.py` (UI-only logic is untestable logic).
- Re-splitting or "just peeking at" test (A6 — this one is unrecoverable).

## Session Notes (append-only; every session adds anything future tickets should know)

```
2026-07-19  Handover complete. No implementation sessions yet.
```
