# ARCHITECTURE.md

**Why this document exists:** the implementable, condensed statement of the frozen system design. `BLUEPRINT.md` is the full rationale; this file is what you keep open while coding. On any conflict, BLUEPRINT.md §4–§6 wins; file an ADR to resolve the discrepancy.

## System at a glance

```
MELD (13.7K utterances, 7 classes, dialogues)
   │ preprocess: ffmpeg→16kHz mono WAV, −23 LUFS, ≤10s (keep END), manifests
   ▼
┌───────────────┐            ┌──────────────────┐
│ RoBERTa-base  │            │ WavLM-base-plus   │   both FROZEN (Stage 1)
│ (text+context)│            │ (raw waveform)    │   checkpoints = config strings
└──────┬────────┘            └────────┬─────────┘
 (B,T_t,768)                   (B,T_a,768)
   proj_t → 256                  proj_a → 256
       └────────┬─────────────────────┘
        2 × CrossModalBlock (bidirectional, 8 heads, pre-LN, masked both ways)
       ┌────────┴─────────────────────┐
 attn-pool → z_t (256)          attn-pool → z_a (256)
       └───► gate g = softmax(σ-linear[z_t; z_a]) ∈ Δ¹ ───┐
             fused = [g_t·z_t ; g_a·z_a]  (512)           │ g logged per example
             MLP 512→256→7 → logits ◄─────────────────────┘
```

Trainable in Stage 1: proj_t, proj_a, 2 fusion blocks, 2 attn-pools, gate, MLP, 2 new token-embedding rows (`<cur>`, `</cur>`). ≈2.3 M params. Stage 2 (conditional, M4 gate): + LoRA r=8, α=16 on q,v of top-4 layers of both encoders.

## Text input contract

```
[spk(t-2)]: u_(t-2) </s> [spk(t-1)]: u_(t-1) </s> <cur> [spk(t)]: u_t </cur>
```
k=2 default (config `data.context_k`), max_len 160 (validated by audit; ADR if violated). Current utterance LAST, wrapped in `<cur>` markers → attribution can separate target-tokens from context-tokens. Speaker names are MELD literals. k=0 path must work (demo uses it).

## Audio input contract

16 kHz mono float32, loudness-normalized −23 LUFS, ≤10.0 s **truncated from the start** (utterance-final prosody kept — ED5). Encoder frame rate ≈49–50 fps → ≤~500 frames.

## Model output contract (all models, incl. baselines)

```python
@dataclass
class Output:
    logits: Tensor            # (B, 7)
    gate: Tensor | None       # (B, 2) = (g_t, g_a); None for unimodal baselines
    xattn: dict | None        # cross-attention maps for XAI; None if not applicable
```
Baselines B1/B2/B3 reuse the SAME encoder wrappers and differ ONLY in the fusion stage — this is what makes comparisons scientific.

## Loss / optimization (Stage 1)

Class-weighted CE, weights = inverse-sqrt train frequency normalized to mean 1. AdamW lr 2e-4, wd 0.01, cosine + 10% warmup, grad-clip 1.0, AMP autocast, batch 8 × accum 4, ≤15 epochs, early stop patience 3 on dev weighted-F1. Seeds {13, 42, 2026} for headline models.

## Performance design

Frozen encoders ⇒ audio hidden states precomputed ONCE to fp16 disk cache, keyed by hash(checkpoint string); cache path invalidates on encoder change. Text stays online (context_k is an experiment variable). Expected epoch speedup 3–4×; this is what makes 3-seed runs feasible on Colab.

## XAI attachment points

1. IG-text: LayerIntegratedGradients at RoBERTa embedding output, baseline = pad embeddings; report `<cur>`-span and context-span separately.
2. IG-audio: at post-proj_a frames (NOT waveform), baseline = zeros; visualize in 0.5 s bins. n_steps 32 (→64 if convergence delta >5%; always log delta).
3. Attention attribution: fusion-block cross-attn maps + pool weights (exposed via `Output.xattn`).
4. Occlusion: 3-token text spans / 1 s audio windows; stratified 500-sample test subset only.
5. Intrinsic: gate values, free at inference — modality-dominance signal #3.

## Benchmark & serving (interfaces only; detail in DEPLOYMENT.md)

Gemini 2.5 Flash runner: few-shot prompt file `bench/prompts/v1.txt` (frozen, versioned), temperature 0, JSON-constrained, cache-first, budget-capped. Demo: mic/file → faster-whisper `small` → k=0 predict → probs + token heatmap + audio timeline + gate dial. ONNX scope = fusion head only (proj→…→MLP), parity atol 1e-4 vs PyTorch.

## Invariants (test-enforced)

- INV1: logits invariant to padding amount (same example, different batch paddings; atol 1e-5). ← the critical one
- INV2: gate components ∈ (0,1) and sum to 1.
- INV3: frozen encoder params have requires_grad=False; exactly the ~2.3M fusion params + 2 embedding rows are trainable in Stage 1.
- INV4: dataset context builder is a pure function of (dialogue, t, k) — golden-string tested.
- INV5: ONNX fusion head matches PyTorch within atol 1e-4 on random and real inputs.
