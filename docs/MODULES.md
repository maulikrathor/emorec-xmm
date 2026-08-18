# MODULES.md

**Why this document exists:** the per-module contract sheet — responsibilities, interfaces, required tests, and shared data schemas. When building or modifying a module, this file states what it must do and what may depend on it. Update it in the same commit as any interface change.

## Dependency graph (imports flow downward only)

```
utils ← data ← models ← training
                 ↑          ↑
                xai ← (checkpoints)
                bench (independent of models except metrics)
                serve ← models, xai.viz
```
`scripts/` and `notebooks/` sit above everything and are never imported.

## Contracts

### src/emorec/utils/
`seed_everything(seed)`, `load_config(path, overrides)`, `JsonlLogger`, `sha_short()`. Tests: seed reproducibility (two seeded runs of a tiny model → identical losses).

### src/emorec/data/
- `download.py` — fetch MELD + checksum verify; idempotent. Test: bad checksum raises.
- `preprocess.py` — MP4→WAV(16k mono)→LUFS norm→manifest JSONL; corrupt-clip exclusion with counts. Tests: manifest counts match CSV minus exclusions; known clip sr/duration.
- `dataset.py` — `MeldDataset(manifest, cfg)`; `build_context(dialogue, t, k) -> str` (pure); `collate(batch) -> Batch{text_ids, text_mask, audio_values, audio_mask, label, meta}`. Tests: golden context string (incl. t<k and k=0); mask shapes; END-keeping truncation.
- `audit.py` — one-command audit → `artifacts/reports/data_audit.md`. Test: runs on toy manifest.

### src/emorec/models/
- `encoders.py` — `TextEncoder(cfg)`, `AudioEncoder(cfg)` returning `(hidden, frame_mask)`; freeze logic; special-token resize with only-new-rows-trainable; audio disk cache (checkpoint-hashed). Tests: INV3; cache≡online.
- `fusion.py` — `MultimodalERC(cfg)`; `forward(batch) -> Output`; `forward_from_embeds(text_embeds, audio_hidden, ...) -> Output` (XAI hook, built up front). Tests: INV1 (critical), INV2, shapes.
- `baselines.py` — B1/B2/B3 sharing encoder wrappers, same `Output` contract (gate/xattn=None where n/a). Tests: shapes.
- `registry.py` — `build_model(cfg)`. Test: unknown name raises.

### src/emorec/training/
- `losses.py` — `build_loss(cfg, class_counts)`: weighted CE (inv-sqrt, mean-normalized), focal(γ=2). Test: weight math vs hand calc.
- `metrics.py` — wF1/mF1/per-class/confusion; `bootstrap_ci(y, ŷ, n=1000)`; `mcnemar(ŷ_a, ŷ_b, y)`. Tests: vs sklearn; CI on synthetic known-variance data.
- `trainer.py` — fit loop per ARCHITECTURE (AMP, accum, clip, early stop, best+last ckpt, run-dir logging: config dump + git SHA + seeds). Test: tiny-subset overfit smoke.

### src/emorec/xai/
Shared attribution artifact schema — `artifacts/attributions/{method}/{utt_id}.npz`:
```
text_attr (T_t,), text_tokens (T_t,) [str], cur_span (2,) [start,end],
audio_attr (T_frames,), frame_times (T_frames,),
pred (int), label (int), probs (7,), extras: convergence_delta / gate
```
- `ig.py` — LayerIG both modalities via `forward_from_embeds`; baselines per ARCHITECTURE; delta logging; n_steps rule. Test: attr shape == input shape; approximate completeness on toy.
- `attention.py` — cross-attn map extraction/aggregation to per-token/frame scores; same schema. Test: pre-aggregation rows sum to 1.
- `occlusion.py` — 3-token spans / 1 s windows Δlogit; subset runner. Test: occlude-everything ≈ baseline prediction.
- `faithfulness.py` — comprehensiveness/sufficiency curves {5,10,20,50}%, AUC; Spearman agreement; randomization sanity. Test: monotonic curve on synthetic perfectly-faithful attribution.
- `dominance.py` — 3 signals + per-class aggregation + agreement tables. Test: dual-modality ablation → near-uniform logits.
- `viz.py` — token heatmap, audio timeline (waveform-aligned, 0.5 s bins), gate dial; shared with demo.

### src/emorec/bench/
- `gemini_runner.py` — cache-first (`artifacts/gemini/{prompt_hash}/{utt_id}.json`), resumable, temp 0, JSON-constrained, ≤2 retries then invalid-marked, cost+latency meters, budget-cap abort, mock-client injectable. Tests (on mock): cache round-trip; parse-failure path; cap-abort.
- `prompts/v1.txt` — frozen prompt + 7 few-shot examples (one per class, from train). Immutable; changes create v2.
- `compare.py` — merged comparison tables incl. break-even volume calc. Test: table math.

### src/emorec/serve/
- `pipeline.py` — `predict(audio, text=None, explain=False) -> dict`; resample→whisper(if text None)→k=0 build→model→(optional IG). Test: end-to-end on bundled 2 s wav.
- `app.py` — Gradio UI (mic/file path + MELD-sample path). Manual test checklist in TESTING.md.
- `export_onnx.py` — fusion-head export, dynamic axes, INV5 parity, latency bench. Test: parity MANDATORY.
