# DEPLOYMENT.md

**Why this document exists:** the complete serving/release plan — demo, ONNX, Docker, weight hosting, and the reproducibility contract — so that M7 is executed from a checklist rather than improvised.

## 1. Weight release (HF Hub)
- Best checkpoint of M (selection: highest dev wF1 among the 3 seeds; report all three on test).
- HF model repo contains: weights (safetensors), config.yaml used, tokenizer additions (`<cur>`,`</cur>`), MODEL_CARD.md mirrored, and the exact git SHA of the training code.
- Also upload: the fusion-head ONNX file.

## 2. Gradio demo
- Entry: `python scripts/demo.py` (wraps `serve/app.py`).
- Paths: (a) mic/file → resample 16 kHz mono → faster-whisper `small` (CPU, int8) → editable transcript → k=0 predict; (b) MELD test-sample picker → full-context predict.
- Displays: predicted emotion + probability bars; token heatmap with `<cur>` span emphasized; audio attribution timeline (0.5 s bins) aligned under waveform; gate dial (g_t vs g_a). "Explain" is a button (IG on demand — it's the slow part), prediction itself is instant.
- UI copy must state: demo path has no dialogue context; explanations describe model behavior, not human emotional truth.

## 3. ONNX inference path (ED9: fusion head ONLY)
- Export proj_t/proj_a → xattn×2 → pools → gate → MLP as one graph, dynamic axes (B, T_t, T_a), opset ≥17.
- T-CRIT-2 parity (atol 1e-4) blocks release.
- `scripts/bench_latency.py` → table per EVALUATION §8, committed into README.

## 4. Docker
- Single CPU-only image: `python:3.11-slim`, ffmpeg installed, `pip install -e ".[serve]"`, weights pulled from HF Hub at container start (slim image; document the env var for a local-weights mount for offline use).
- `docker run -p 7860:7860 emorec-xmm` → demo at localhost:7860. GPU serving explicitly out of scope.

## 5. Reproducibility contract (README, verified at M7 gate)
≤5 commands from clean clone to: (a) running demo with released weights; (b) headline eval reproduced from cached predictions (`--from-cache` path — commit predictions JSONs to artifacts release, not git); (c) full retrain documented as GPU-required with expected wall time from the M0 compute probe.

## 6. Release checklist (M7)
[ ] weights + ONNX on Hub · [ ] parity green · [ ] latency table in README · [ ] docker build + run verified · [ ] stranger-reproducibility executed literally on a clean venv · [ ] model card filled with measured numbers only · [ ] README results tables complete with CI · [ ] demo GIF recorded · [ ] repo public · [ ] (post-scope, cheap) HF Spaces deployment = FUTURE_WORK E-ext2.
