# emorec-xmm

Explainable multimodal (text + audio) emotion recognition on the MELD dialogue
dataset: a gated cross-attention fusion model over frozen RoBERTa (text) and
WavLM (audio) encoders, evaluated against unimodal/concat baselines and a
Gemini benchmark, with an integrated-gradients / attention / occlusion
explainability layer.

**Status:** Review-1 sprint, M0 in progress.

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest -q
```

## Data pipeline

MELD isn't bundled with the repo. Point the tools at a local copy of the raw
MELD mirror (CSVs + `.mp4` media) via, in priority order: `--meld-root`,
the `MELD_ROOT` environment variable, or `data.meld_root` in
`configs/base.yaml`. The exact mirror layout (which subdirectory holds which
CSV/videos) is discovered at runtime, not assumed.

```bash
# Stage 1+2 only: validate metadata and inventory media, no audio written.
python scripts/inspect_meld.py --meld-root /path/to/MELD.Raw

# Smoke run: a handful of dev utterances, written to dev_smoke.jsonl.
python scripts/prepare_data.py --stage smoke --meld-root /path/to/MELD.Raw

# Full run over one or all splits.
python scripts/prepare_data.py --stage full --split all --meld-root /path/to/MELD.Raw
```

Both scripts read/write only under `artifacts/` (gitignored): extracted WAVs
under `artifacts/wav/`, manifests under `artifacts/manifests/`, and
validation/inventory/exclusion reports under `artifacts/reports/`.
