# Dialogue-context ablation (k=2 vs k=0)

Full MELD train (9,988), dev 1,108, seed 42, frozen encoders, max_epochs=10.
Best dev weighted-F1 per run (model selection on dev, as everywhere else).

| model | k=2 best dev wF1 | k=0 best dev wF1 | delta (k=2 - k=0) |
|---|---|---|---|
| xattn (M)   | 0.6155 (ep5)  | 0.6083 (ep5) | +0.0072 |
| text_only (B1) | 0.5548 (ep10) | 0.5928 (ep7) | -0.0380 |

## Reading (one seed, no significance test — do not over-claim)

1. For the multimodal model, k=2 dialogue context contributes +0.0072 weighted-F1.
   This is NOT a meaningful improvement. We do not claim dialogue context helps.

2. For the frozen text-only baseline, REMOVING context IMPROVED weighted-F1 by
   0.0380. Reported as a negative result.

Likely cause: with a frozen RoBERTa and max_text_len=160, a k=2 context string
spends most of the token budget on previous turns (MELD utterances average ~8
whitespace tokens). The current utterance occupies a small fraction of a mostly
-context sequence, and a frozen encoder with only a trainable pooling head
cannot learn to down-weight the context span. The <cur>/</cur> markers delimit
the target span but the frozen encoder was never trained to exploit them. The
multimodal model is less affected because cross-attention provides a second
route to the target signal.

## Consequence for the headline comparison

The strongest text-only baseline on dev is k=0 at 0.5928, not k=2 at 0.5548.
The M-vs-B1 dev gap therefore narrows from +0.0607 to +0.0227.
The held-out TEST comparison is unaffected: both test-evaluated models were
trained with k=2, and that comparison stands (M 0.6173 vs B1 0.6015,
McNemar p=0.021). The k=0 variants were NOT evaluated on test, by design —
test is touched once, for the pre-registered final runs only (A6).
