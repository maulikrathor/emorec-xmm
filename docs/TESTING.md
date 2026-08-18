# TESTING.md

**Why this document exists:** defines the test strategy, the mandatory tests (with IDs referenced across all docs), and the CI contract. Tests here are part of each module's definition of done; an AI session must run `pytest -q` before and after changes and never build on red.

## Philosophy
Unit-test contracts, not implementations. Heavy things (training, IG runs, API calls) are NOT in CI; their correctness is protected by smoke tests, invariants, mocks, and gates. Test the places where silent corruption is possible — masks, caches, context strings, statistics — hardest.

## Critical tests (IDs are canonical; referenced from other docs)
- **T-CRIT-1 Mask invariance (INV1).** Same example padded to different lengths in different batches ⇒ identical logits (atol 1e-5), fusion model. Written BEFORE fusion training (IMPLEMENTATION_GUIDE §8). Failure = attending to padding = every downstream attribution is garbage while metrics look plausible. Non-negotiable.
- **T-CRIT-2 ONNX parity (INV5).** Fusion head ONNX vs PyTorch, random + real inputs, atol 1e-4.
- **T-CRIT-3 Freeze correctness (INV3).** Exactly the fusion params + 2 new embedding rows trainable in Stage 1.
- **T-CRIT-4 Context builder golden strings (INV4).** Synthetic 3-turn dialogue: exact expected string for k∈{0,1,2}, incl. dialogue-start (t<k) handling and `<cur>` placement.
- **T-CRIT-5 Cache equivalence.** Cached audio features ≡ online forward (fp16 tolerance).
- **T-CRIT-6 Budget-cap + cache round-trip** on the mocked Gemini client (no real API in tests, ever).

## Standard test suites (per MODULES.md required-tests column)
data (counts, sr/duration, truncation-keeps-end, mask shapes) · models (shapes, gate simplex INV2, registry) · training (loss weight math vs hand-calc; metrics vs sklearn; bootstrap CI on synthetic; tiny-subset overfit smoke ≤50 steps) · xai (attr shapes; IG approximate completeness on toy; attention rows sum to 1; occlude-all ≈ baseline; faithfulness curve monotonic on synthetic perfect attribution; dual-ablation → near-uniform logits) · serve (pipeline end-to-end on bundled 2 s wav) · utils (seed reproducibility).

## CI contract
GitHub Actions on every push: `ruff check` + `pytest -q`, CPU-only, target <5 min. Tests requiring model downloads use tiny fixtures/mocks (a 2-layer random "roberta-like" stub config for shape tests; the real checkpoints are integration-only, run locally). No GPU, no network beyond pip in CI.

## Manual test checklists
- **Demo (M7):** mic path records → transcript appears and is editable → prediction + heatmap + timeline + gate dial render → MELD-sample path shows context; a wrong-transcript edit changes the prediction (demonstrates text sensitivity).
- **Stranger-reproducibility (M7 gate):** on a clean machine/venv, follow README's 5 commands literally; any deviation is a README bug.

## Regression rule
Every bug that survives to a milestone gate gets a regression test named `test_regression_<issue>.py` before the fix is merged.
