# CLAUDE.md — AI-assisted development operating manual

**Why this document exists:** this is the file the implementing model (Opus 4.8 or successor) reads at the start of EVERY coding session. It defines read order, conventions, guardrails, commands, and the definition of done. It exists so that no future chat ever needs to re-explain the project. Keep it short enough to actually be read every time; push detail into the referenced docs.

## Session start protocol

1. Read this file fully.
2. Read `docs/ROADMAP.md` → find the current milestone and its exit gate.
3. Read the relevant section of `docs/IMPLEMENTATION_GUIDE.md` for the module you're about to touch.
4. Check `docs/adr/` for any post-freeze decisions newer than your knowledge of the repo.
5. Run `pytest -q` before writing anything. Never build on red.

Deeper context on demand: `PROJECT_CONTEXT.md` (why), `docs/ARCHITECTURE.md` (what), `docs/BLUEPRINT.md` (full frozen design — authoritative on any conflict), `docs/DECISIONS.md` (why-not).

## The ten commandments

1. **The blueprint is frozen.** You may not change architecture, dataset, splits, metrics, or scope without writing an ADR (`docs/DECISIONS.md` template) stating what new information invalidated the freeze. Silent reversals of ED1–ED12 are forbidden.
2. **Implementation order is dependency-true** (`IMPLEMENTATION_GUIDE.md` §order). Never start item N+1 while item N's required tests are red.
3. **The mask-invariance test (TESTING.md T-CRIT-1) is written and green BEFORE any fusion training.** This is the hill to die on.
4. **Test split discipline:** MELD standard splits, unmodified, ever (A6). Dev for selection; test only for final reported runs. No peeking, no re-splitting, no "quick check on test."
5. **No claim without statistics.** Any model-vs-model statement in any doc/README requires bootstrap CI; paired comparisons require McNemar. If not significant, write "no significant difference."
6. **Configs are the only experiment interface.** New experiment = new YAML in `configs/` + row in `EXPERIMENTS.md`. No hardcoded hyperparameters, no CLI arg soup, no notebook-only results.
7. **Notebooks contain zero load-bearing logic.** Every figure/table in README or REPORT regenerates from a script or a notebook that only calls package code.
8. **Money-touching code is cache-first and resumable** (Gemini runner). Never call a paid API in a loop without a disk cache keyed by (input-id, prompt-hash) and a budget cap.
9. **Empty is honest, fake is fatal.** Never fill results placeholders with invented, estimated, or "example" numbers — in README, REPORT, MODEL_CARD, anywhere. `–` until measured.
10. **When numbers disappoint, the deliverable is the analysis.** Do not "fix" a negative result by quietly changing eval; write it up (KNOWN_LIMITATIONS.md / REPORT.md).

## Code conventions

- Python ≥3.11, src-layout package `emorec`. `src/emorec` never imports from `scripts/` or `notebooks/`.
- Style: ruff (config in pyproject) — line length 100, type hints on public functions, docstrings state the *contract* not the obvious.
- Readable > clever (owner preference). No premature optimization except the audio-feature cache (ED-perf, it's designed-in).
- Config: OmegaConf; `configs/base.yaml` is the single source of default truth; experiment YAMLs are minimal overrides.
- Every training/eval entrypoint: `--config path.yaml` + optional dotlist overrides; logs seeds, git SHA, config dump into the run dir.
- Randomness: `emorec.utils.seed_everything(seed)` at every entrypoint. Headline runs: seeds {13, 42, 2026}.
- Artifacts: everything heavy under `artifacts/` (gitignored). Attribution files: `artifacts/attributions/{method}/{utt_id}.npz` with schema documented in `MODULES.md`.
- Errors: fail loud and early; no silent `except Exception: pass` anywhere, ever.

## Commands (keep current as scripts land)

```bash
pip install -e ".[dev]"          # dev install
pytest -q                        # unit tests (<2 min, CPU)
ruff check src tests scripts     # lint
python scripts/prepare_data.py   # download + preprocess + manifests + audit
python scripts/train.py --config configs/<name>.yaml
python scripts/evaluate.py --config configs/<name>.yaml
python scripts/run_xai.py --method ig|attention|occlusion --ckpt <path>
python scripts/run_faithfulness.py --attr-dir artifacts/attributions
python scripts/run_gemini.py --prompt-version v1 [--subset stratified1k]
python scripts/export_onnx.py && python scripts/bench_latency.py
python scripts/demo.py
```

## Definition of done (per module)

Code + its required tests (MODULES.md table) green + docstrings + config keys added to base.yaml with comments + one-line entry in the milestone log at the bottom of ROADMAP.md. For result-producing steps: the results table row filled in the relevant doc, with CI.

## Known dragon locations (budget extra care/time)

1. MELD audio extraction: broken/missing clips are normal — exclude with logged counts (audit).
2. Padding masks through cross-attention, both directions (R6).
3. Captum LayerIG on a dual-input model — use the wrapper pattern in IMPLEMENTATION_GUIDE.md §XAI; do not attribute both modalities in one call.
4. WavLM/Wav2Vec2 feature-extractor sampling-rate mismatches — assert sr==16000 at load.
5. ONNX export: fusion head only (ED9). Do not attempt encoder export; it's a known time sink.

## What to do when stuck

Root-cause before workaround (owner preference). If a fix requires deviating from the blueprint: ADR first, code second. If a result contradicts a blueprint assumption (A1–A7): that's expected — follow the assumption's written validation/fallback path.
