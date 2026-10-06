# Explainable multimodal emotion recognition on MELD — final report

**Scope.** One full-train run per model (seed 42) of a text-only baseline (B1), an audio-only baseline (B2) and
a gated cross-attention fusion model (M), evaluated on the official MELD dev and test splits, plus an
explainability study of M (gate analysis, modality ablation, Integrated Gradients and its faithfulness).
Every number below is copied from a file under `results/`; the file is named in an HTML comment beside it
(`<!-- src: ... -->`, visible in the raw markdown). Differences are plain subtraction of two cited values.
Anything that was not measured is marked "–" or stated as not measured. Architecture and scope follow
`docs/ARCHITECTURE.md` and `docs/DECISIONS.md` (ED1–ED12); no deviation was made except the documented
IG-audio attachment point (§7d).

**What was NOT done** (so nothing here should be read as covering it): multiple seeds, the concat baseline B3,
LoRA/Stage 2, occlusion and attention attribution, the Gemini benchmark, ONNX export and the Gradio demo.

---

## 1. Problem and task definition

MELD is a multi-party sitcom (Friends) dialogue corpus with one emotion label per utterance. The task is 7-class
emotion recognition in conversation (ERC): neutral, joy, sadness, anger, surprise, fear, disgust. The official
splits are used unmodified: 9,989 train / 1,109 dev / 2,610 test labelled utterances.
<!-- src: results/reports/meld_metadata.md (Row counts) -->

The usable dev set is **1,108**, not 1,109: one labelled dev utterance has no media file (`MEDIA_MISSING`; it
is evidently a neutral utterance, since the dev CSV has 470 neutral and the dev evaluation has 469). One train utterance failed ffmpeg
extraction (`FFMPEG_FAILED`), so **9,988** train utterances were written; all 2,610 test utterances were written.
<!-- src: results/reports/excluded.json; results/reports/media_inventory.md (dev: labeled_rows 1109, matched 1108, missing_media 1); results/reports/meld_metadata.md (R1 dev neutral 470); results/runs/*/classification_report.txt (dev neutral support 469) -->

Class distribution (labelled rows):

| class | train | dev | test |
|---|---|---|---|
| neutral | 4710 (47.15%) | 470 (42.38%) | 1256 (48.12%) |
| joy | 1743 (17.45%) | 163 (14.7%) | 402 (15.4%) |
| sadness | 683 (6.84%) | 111 (10.01%) | 208 (7.97%) |
| anger | 1109 (11.1%) | 153 (13.8%) | 345 (13.22%) |
| surprise | 1205 (12.06%) | 150 (13.53%) | 281 (10.77%) |
| fear | 268 (2.68%) | 40 (3.61%) | 50 (1.92%) |
| disgust | 271 (2.71%) | 22 (1.98%) | 68 (2.61%) |

<!-- src: results/reports/meld_metadata.md (R1) -->

Severe imbalance (fear and disgust are about 2–3% of the data) is the central difficulty, which is why weighted-F1
(the MELD convention) is reported together with macro-F1, per-class F1 and the all-neutral baseline.

## 2. Data pipeline

**Media reconciliation.** The Kaggle mirror holds 15,908 `.mp4` files: 13,848 real media plus 2,060 macOS
AppleDouble resource-fork files (`._*.mp4`), all of the latter in the test directory. Of the 13,848 real files,
13,707 matched a labelled row, 141 are orphans (4 in dev, 137 in test; counted, never labelled, never deleted)
and 1 labelled row has no media. The arithmetic reconciles exactly per split.
<!-- src: results/reports/media_inventory.md (Per split; Conclusion) -->

**Split-scoped lookup.** 2,068 file stems (e.g. `dia0_utt0`) occur in more than one split directory, so
media are matched per split and every `utt_id` is split-namespaced (`dev/dia0_utt0`); a global lookup would
silently attach the wrong audio.
<!-- src: results/reports/media_inventory.md (Cross-directory duplicate stems: 2068) -->

**Preprocessing.** ffmpeg extraction to 16 kHz mono WAV, loudness normalisation to −23 LUFS, and audio capped at
10 s **keeping the END of the utterance (ED5)** because utterance-final prosody carries emotion. Clips shorter than
0.4 s are not loudness-normalised (the normaliser cannot measure them). Normalisation counters:
train 212 / dev 28 / test 68 too short; 91 / 7 / 23 gain-clamped; 63 / 5 / 9 peak-limited; 1 silent train clip.
<!-- src: docs/ARCHITECTURE.md (audio contract); configs/base.yaml (min_lufs_duration_s 0.4); results/reports/excluded.json (normalisation) -->

**Text.** Only encoding artefacts are repaired; punctuation and case are kept (they carry emotion).

**Dialogue context (k = 2).** The model sees the two preceding turns plus the current one, current turn last and
wrapped in two new special tokens:
`[spk(t-2)]: u_(t-2) </s> [spk(t-1)]: u_(t-1) </s> <cur> [spk(t)]: u_t </cur>`.
At the start of a dialogue fewer turns are used and nothing is padded. This format is test-enforced by five
golden strings. Run configuration: `context_k 2`, `max_text_len 160`, `max_audio_sec 10.0`.
<!-- src: docs/ARCHITECTURE.md (text input contract); results/runs/*/run_meta.json (config.data) -->

**Data caveats found by the pipeline.** Five (Season, Episode, StartTime, EndTime) keys appear in two splits, and
156 (Season, Episode) pairs occur in more than one split; MELD's official splits are drawn dialogue-wise from
overlapping episodes, so the same cast and episodes recur across splits.
<!-- src: results/reports/meld_metadata.md (R4, R5) -->

## 3. Architecture (frozen)

- **Encoders (both frozen):** RoBERTa-base for the text with its context string, and WavLM-base-plus for the
  waveform. Only two new embedding rows (`<cur>`, `</cur>`) are trainable in the text encoder, through a gradient
  hook. Both encoders stay in `eval()` always.
- **Audio features are cached once** (fp16, per utterance, keyed by a hash of the checkpoint string), so training
  never re-runs WavLM; text is encoded online.
- **Fusion (M):** both streams are projected to d = 256; two bidirectional cross-modal blocks (8 heads, pre-LN,
  masked in both directions) let text attend to audio and audio attend to text; a masked learned-query attention
  pool summarises each stream into `z_t` and `z_a`; a scalar gate `g = softmax(Linear([z_t; z_a])) = (g_text, g_audio)`
  weights them; `fused = [g_text·z_t ; g_audio·z_a]` goes through an MLP 512 → 256 → 7.
  Order matters for §7: **the gate is applied after the cross-modal blocks.**
- **Baselines** reuse the same encoder wrappers and differ only in the fusion stage: B1 text-only (pooled text → MLP),
  B2 audio-only (pooled audio → MLP). (B3, concat, is implemented and tested but was not trained.)
- **Invariants (test-enforced):** logits are invariant to padding amount (INV1, the critical one), the gate is a
  simplex (INV2), frozen encoder parameters have `requires_grad=False` and only the two embedding rows train (INV3).
<!-- src: docs/ARCHITECTURE.md; docs/DECISIONS.md (ED1–ED5); results/runs/*/config.yaml (d_model 256, n_heads 8, n_fusion_blocks 2, num_classes 7) -->

**Trainable parameters.** `docs/ARCHITECTURE.md` and `docs/CLAUDE_CODE_HANDOFF.md` state "≈2.3 M". Counting the
implemented model (a throwaway instantiation with the real dimensions; the count is not stored in `results/`)
gives **3,688,969** trainable parameters: 3,687,433 in the fusion side (two projections 196,864 each, two cross-modal
blocks 3,159,040 together, pools 256 each, gate 1,026, MLP 133,127) plus 1,536 for the two embedding rows. The
cross-modal blocks' 4× feed-forward layers account for the difference from the documented figure. The documented
2.3 M should be read as an outdated estimate; it does not affect any result.

## 4. Experimental setup

| item | value |
|---|---|
| loss | class-weighted cross-entropy, weight ∝ 1/√(train class count), mean-normalised to 1 (ED6) |
| optimiser | AdamW, lr 2e-4, weight decay 0.01 |
| schedule | cosine with 10% warmup, stepped on optimiser steps |
| batch | 8 × gradient accumulation 4 (effective 32), grad-clip 1.0, AMP |
| epochs | at most **10** (the run configs override the base default of 15) |
| early stopping | patience 3 on dev weighted-F1; the best-dev checkpoint is kept |
| seed | 42 — **one seed per model** |
| train data | full train split (9,988 utterances), `use_cache: true` |

<!-- src: results/runs/*/config.yaml and run_meta.json (train.*, max_epochs 10); configs/base.yaml (max_epochs 15, patience 3); docs/ARCHITECTURE.md (loss) -->

| model | epochs run | best epoch (dev wF1) | wall time |
|---|---|---|---|
| B1 text-only | 10 | 10 | 503 s |
| B2 audio-only | 10 | 10 | 367 s |
| M gated fusion | 8 | 5 | 779 s |

<!-- src: results/runs/*/run_meta.json (epoch_count, wall_time_s) and results/runs/*/metrics.json (per-epoch dev_weighted_f1; best epoch = argmax) -->

B1 and B2 stopped at the 10-epoch cap with dev weighted-F1 still rising slightly (B1: 0.5529 at epoch 8, 0.5525 at 9,
0.5548 at 10; B2: 0.4002, 0.3996, 0.4012). M peaked at epoch 5 (0.6155) and then overfitted while training loss kept
falling (epoch 5 → 8: train loss 1.045 → 0.560, dev wF1 0.6155 → 0.5958).
<!-- src: results/runs/*/metrics.json -->

Test reports and predictions exist once per final checkpoint (`classification_report_test.txt`,
`predictions_test.jsonl`); the checkpoint, stopping epoch and every design choice were made on dev only.

## 5. Results

### 5.1 Headline tables

**Dev (n = 1,108).** Bootstrap 95% CI (1,000 utterance-level resamples, percentile method).

| model | accuracy [CI] | macro-F1 [CI] | weighted-F1 [CI] |
|---|---|---|---|
| all-neutral baseline | 0.4233 | 0.0850 | 0.2518 |
| B1 text-only | 0.5776 [0.5478, 0.6065] | 0.3760 [0.3500, 0.4021] | 0.5548 [0.5230, 0.5855] |
| B2 audio-only | 0.4558 [0.4242, 0.4865] | 0.2339 [0.2103, 0.2576] | 0.4012 [0.3687, 0.4342] |
| **M gated fusion** | **0.6363** [0.6074, 0.6634] | **0.4670** [0.4251, 0.5057] | **0.6155** [0.5832, 0.6448] |

<!-- src: results/reports/statistics.md (Bootstrap CI table); results/reports/final_results.md (trivial baseline) -->

**Test (n = 2,610).**

| model | accuracy [CI] | macro-F1 [CI] | weighted-F1 [CI] |
|---|---|---|---|
| all-neutral baseline | 0.4812 | – | – |
| B1 text-only | 0.6199 [0.6008, 0.6372] | 0.3948 [0.3734, 0.4151] | 0.6015 [0.5815, 0.6193] |
| B2 audio-only | 0.4797 [0.4594, 0.4966] | 0.2222 [0.2063, 0.2357] | 0.4273 [0.4052, 0.4457] |
| **M gated fusion** | **0.6387** [0.6195, 0.6559] | **0.4276** [0.4003, 0.4534] | **0.6173** [0.5973, 0.6367] |

<!-- src: results/reports/statistics.md; all-neutral test accuracy = neutral share 48.12% from results/reports/meld_metadata.md (R1 test). The all-neutral test macro-F1 / weighted-F1 were not computed in results/, hence "–". -->

Two honest reads of these tables:
- Every model beats the all-neutral baseline on dev weighted-F1 (0.2518) and macro-F1 (0.0850), but **B2's test accuracy
  (0.4797) is slightly below the all-neutral accuracy (0.4812)**: audio alone does not beat "always neutral" on test
  accuracy. (The all-neutral test weighted-/macro-F1 were not computed, so the test F1 comparison is not made.)
- M's gain over B1 (differences of the table values): dev weighted-F1 **+0.0607**, macro-F1 +0.0910, accuracy +0.0587;
  test weighted-F1 **+0.0158**, macro-F1 +0.0328, accuracy +0.0188.

### 5.2 Statistical comparison: overlapping CIs, significant paired test

Exact two-sided McNemar test on paired per-utterance correctness. `n01` = the first-named model right and the
second wrong; `n10` = the reverse.

| comparison | split | n01 | n10 | p |
|---|---|---|---|---|
| M vs B1 | dev | 141 | 76 | 1.212e-05 |
| M vs B2 | dev | 287 | 87 | 4.600e-26 |
| B1 vs B2 | dev | 270 | 135 | 1.813e-11 |
| M vs B1 | test | 240 | 191 | 2.067e-02 |
| M vs B2 | test | 630 | 215 | 5.550e-48 |
| B1 vs B2 | test | 643 | 277 | 3.104e-34 |

<!-- src: results/reports/statistics.md (McNemar table). Reproduced exactly from results/runs/*/predictions_*.jsonl with emorec.training.metrics.bootstrap_ci / mcnemar / align_predictions (see §9). -->

**Both facts are reported.** On test, the weighted-F1 CIs of M [0.5973, 0.6367] and B1 [0.5815, 0.6193] overlap,
**and** the paired McNemar test is significant (p = 0.021). These do not contradict each other. The two bootstrap
intervals are marginal: each is computed for one model as if the other did not exist. M and B1 are evaluated on the
same 2,610 utterances and make strongly correlated errors, and for correlated predictors overlapping marginal CIs
are a conservative criterion for "no difference". McNemar conditions on the utterances where the two models
disagree (240 + 191 = 431) and asks whether the split between "M right / B1 wrong" and the reverse is 50/50; it is
not (net +49 for M). Under the claim rule of ED12, p < 0.05 permits the statement "M is significantly more accurate
than B1 on test", and the overlap of marginal CIs is reported alongside it as a caution about effect size: +0.0158
weighted-F1 on test is small. Note that McNemar tests **correctness (accuracy)**, not weighted-F1; a paired
bootstrap of the weighted-F1 difference was not computed.

### 5.3 The dev-versus-test gap (+0.0607 vs +0.0158)

M's advantage over B1 shrinks from +0.0607 weighted-F1 on dev to +0.0158 on test. The pattern of the per-model
scores matters:

| model | dev wF1 | test wF1 | test − dev |
|---|---|---|---|
| B1 | 0.5548 | 0.6015 | +0.0467 |
| B2 | 0.4012 | 0.4273 | +0.0261 |
| M | 0.6155 | 0.6173 | +0.0018 |

<!-- src: results/reports/statistics.md -->

The checkpoint, the stopping epoch and the choice of what to report were all decided on dev, and M's dev score is the
maximum of eight evaluated epochs (0.4945 … 0.6155; the later epochs 6–8 score 0.5802, 0.5825, 0.5958), a selection
effect that inflates a dev number by construction. This is the most plausible contributor to the large dev gain,
and the +0.0158 test figure is the held-out estimate to quote. Two cautions apply. First, M's own test score is not
lower than its dev score, so the selection optimism is **not visible as a drop for M**; what the table shows is that
the baselines score *higher* on test than on dev, i.e. B1 is relatively stronger on the test split. Second, with one
seed and splits of different composition (e.g. neutral 42.4% of dev vs 48.1% of test) the data cannot separate
selection optimism from split difficulty. The dev bootstrap CIs are also wide (M [0.5832, 0.6448], B1 [0.5230, 0.5855]).
<!-- src: results/runs/xattn-seed42-*/metrics.json (per-epoch dev_weighted_f1); results/reports/meld_metadata.md (R1) -->

## 6. Per-class analysis

Per-class F1 of each model's best checkpoint (two decimals, as written by the classification reports; support in
brackets).

| class | dev B1 | dev B2 | dev **M** | test B1 | test B2 | test **M** |
|---|---|---|---|---|---|---|
| neutral | 0.73 (469) | 0.64 | **0.78** | 0.76 (1256) | 0.66 | **0.78** |
| joy | 0.53 (163) | 0.18 | **0.60** | 0.58 (402) | 0.21 | **0.60** |
| sadness | **0.38** (111) | 0.18 | 0.31 | **0.31** (208) | 0.14 | 0.22 |
| anger | 0.38 (153) | 0.38 | **0.47** | 0.49 (345) | 0.34 | **0.50** |
| surprise | 0.57 (150) | 0.26 | **0.66** | 0.53 (281) | 0.21 | **0.55** |
| fear | 0.04 (40) | 0.00 | **0.25** | 0.10 (50) | 0.00 | **0.21** |
| disgust | 0.00 (22) | 0.00 | **0.21** | 0.00 (68) | 0.00 | **0.14** |

<!-- src: results/runs/{text_only,audio_only,xattn}-seed42-*/classification_report.txt (dev) and classification_report_test.txt (test). Test four-decimal values (e.g. sadness 0.3128 → 0.2166, fear 0.0952 → 0.2056, disgust 0.0000 → 0.1412) are in results/reports/final_results.md -->

> **Provenance note.** The per-class table in `results/reports/final_results.md` is headed "best checkpoint, dev" but its
> values equal the **test** classification reports (e.g. M's seven values average to the test macro-F1 0.4276, not the
> dev 0.4670). Dev per-class numbers above are therefore taken from the per-run `classification_report.txt` files, and
> the four-decimal values quoted below are test values.

**Where fusion helps.** Fear and disgust, the two rarest classes, gain most: test fear F1 0.10 → 0.21 and disgust
0.00 → 0.14 (four-decimal: 0.0952 → 0.2056 and 0.0000 → 0.1412); on dev fear 0.04 → 0.25 and disgust 0.00 → 0.21.
Neutral and joy improve by 0.02–0.07 on both splits. Surprise and anger gain clearly on dev (+0.09 each) but only
marginally on test (+0.02 and +0.01), so those two should not be presented as robust gains. Audio alone cannot detect
fear or disgust at all (B2 F1 = 0.00 on both splits), so the fusion gain on these classes is not "audio recognises
fear"; it is a gain of the combination. Per-class supports are small (dev disgust n = 22, fear n = 40) and no per-class
confidence intervals were computed, so individual per-class differences carry a lot of sampling noise.

**Where fusion regresses: sadness.** M is worse than the text-only baseline on sadness on both splits: dev 0.38 → 0.31
and test 0.31 → 0.22 (four-decimal test: 0.3128 → 0.2166). The regression is a recall loss: sadness recall falls from
0.34 to 0.22 on dev and from 0.27 to 0.14 on test, while precision rises (0.43 → 0.53 on dev, 0.37 → 0.43 on test).
M is more conservative about predicting sadness. The cause was not investigated.
<!-- src: results/runs/*/classification_report*.txt (precision/recall columns) -->

Confusion matrices for every model and split are in `results/runs/*/confusion_matrix*.png`.

## 7. Explainability

All explainability results use M's best checkpoint on the dev split.
<!-- src: results/reports/gate_analysis.md, ablation.md, faithfulness.md (checkpoint xattn-seed42-20261006T070547Z, manifest dev) -->

### 7a. Gate analysis — the gate is text-saturated

On all 1,108 dev utterances the learned gate puts almost all weight on text:

| statistic | g_audio |
|---|---|
| mean | 0.0013 |
| median | 0.0003 |
| std | 0.0039 |
| p05 / p95 | 0.0001 / 0.0047 |
| fraction audio-dominant (> 0.5) | 0.0000 |
| fraction saturated toward text (< 0.05) | 0.9991 |

The largest `g_audio` on dev is 0.0692. Per true class the mean g_audio is 0.0005 (neutral), 0.0008 (joy), 0.0024
(sadness), 0.0030 (anger), 0.0005 (surprise), 0.0045 (fear), 0.0029 (disgust); by correctness 0.0011 (correct, n = 705)
vs 0.0016 (incorrect, n = 403). The gate is reported exactly as produced; nothing was rescaled (J-08).
<!-- src: results/reports/gate_analysis.md; results/reports/gates.csv; figure results/reports/figures/final_gate_distribution.png -->

Read alone, the gate says "M ignores audio" — yet M beats the text-only baseline (§5), most on the rare classes.

### 7b. Modality ablation — audio is load-bearing despite g_audio ≈ 0.001

Same trained weights; only the inference-time input changes. `zero_audio` zeroes the encoder frames (mask unchanged),
`zero_text` replaces the text embeddings by pad embeddings, `shuffle_audio` gives each utterance another utterance's
audio from the same batch (a seeded derangement).

| mode | accuracy | macro-F1 | weighted-F1 [95% CI] | ΔwF1 | Δmacro | McNemar vs control (n01 / n10, p) |
|---|---|---|---|---|---|---|
| none (control) | 0.6363 | 0.4670 | 0.6155 [0.5832, 0.6448] | – | – | – |
| zero_audio | 0.5848 | 0.4325 | 0.5782 [0.5480, 0.6069] | **−0.0373** | −0.0345 | 93 / 36, **p = 5.43e-07** |
| shuffle_audio | 0.6119 | 0.4324 | 0.5860 [0.5551, 0.6177] | **−0.0296** | −0.0345 | 71 / 44, **p = 0.015** |
| zero_text | 0.4097 | 0.0991 | 0.2637 [0.2348, 0.2957] | **−0.3519** | −0.3679 | 319 / 68, **p = 5.7e-40** |

<!-- src: results/reports/ablation.md; results/reports/ablation.csv; figure results/reports/figures/final_ablation.png. n01 = control right, ablated wrong. McNemar is on correctness. -->

- Removing the **audio** costs 0.0373 weighted-F1 (p = 5.43e-07): the model uses audio even though the gate gives it
  0.0013.
- Removing the **text** is catastrophic (−0.3519 weighted-F1, macro-F1 0.0991), so text is the dominant modality
  (consistent with MELD's known text dominance and with B1 vs B2).
- **Shuffled audio** hurts significantly too (−0.0296, p = 0.015). If audio acted only as a presence/bias signal, a
  shuffled stream (same shape and statistics, wrong utterance) would leave accuracy unchanged. It does not, which is
  consistent with the model using utterance-specific audio content. Whether zeroing and shuffling differ from each
  other was not tested (their weighted-F1 CIs overlap).
- Per class (small supports, so indicative only), `shuffle_audio` takes fear from 0.2466 to 0.0923 and
  `zero_audio` takes anger from 0.4739 to 0.3951 and surprise from 0.6598 to 0.5869.
<!-- src: results/reports/ablation.md (Per-class F1) -->

### 7c. Synthesis — why the gate and the ablation disagree

The gate is applied **after** the bidirectional cross-modal blocks. By the time the pooled streams are gated, the
text stream has already attended to the audio frames, so audio-derived information is already mixed into `z_t`. The
gate therefore weights the pooled audio *summary* `z_a`, not audio *information*: a near-zero `g_audio` is compatible
with a model that relies heavily on audio that has entered through the text stream. The modality ablation is the
instrument that detects this, because it removes the audio before the cross-attention rather than reading a
downstream weight, and the shuffle control rules out audio acting merely as a presence signal.

State plainly: **this is a measured failure of an intrinsic interpretability signal to reflect information flow.**
`g_audio` = 0.0013 does not mean the audio is unused; the ablation shows a statistically significant reliance on it.
The practical consequence is that a gate value must not be reported as a modality-importance measure for this
architecture without an ablation or attribution check beside it.

Scope of the claim: what is *measured* is the disagreement (g_audio ≈ 0.001, yet zeroing or shuffling audio
significantly lowers accuracy). The *mechanism* — audio entering the text stream through cross-attention — follows from
the architecture and is the explanation consistent with the data; it was not isolated by a separate experiment
(for instance, ablating only the audio→text cross-attention direction, or measuring the size of `g_audio·z_a` against
the rest of the MLP input). A small-but-nonzero gated audio summary amplified by the MLP is an alternative route that
these experiments do not exclude.

### 7d. Integrated Gradients and its faithfulness

**Method.** Captum IG, one modality per pass, target = the predicted class's logit, text baseline = pad embeddings,
audio baseline = zeros. **Documented deviation:** IG-audio is attributed at the encoder's frame-level output
(pre-`proj_a`, 768-d), not the post-`proj_a` point named in `docs/ARCHITECTURE.md`; both are frame-aligned, so the
temporal reading is identical and only the feature basis differs (project-lead decision, no ADR).
Faithfulness uses 40 class-stratified dev examples, IG n_steps = 64, fractions {0.05, 0.1, 0.2, 0.5}, and a random-
attribution control (mean of 5 seeded draws per example). Comprehensiveness deletes the top-k% attributed positions
(higher is more faithful); sufficiency keeps only the top-k% and reports the drop from the full input (lower is more
faithful). Values are mean drops in the predicted-class probability; AUC is the unnormalised trapezoid area.
<!-- src: results/reports/faithfulness.md (header); results/reports/xai_examples.md (header) -->

| modality | curve | IG AUC [95% CI] | random AUC [95% CI] | IG − random [paired 95% CI] | IG better? |
|---|---|---|---|---|---|
| text | comprehensiveness | 0.1490 [0.1184, 0.1779] | 0.0825 [0.0651, 0.0987] | +0.0665 [+0.0451, +0.0889] | yes |
| text | sufficiency | 0.0809 [0.0496, 0.1100] | 0.1394 [0.1104, 0.1678] | −0.0585 [−0.0815, −0.0359] | yes |
| audio | comprehensiveness | 0.0556 [0.0430, 0.0680] | 0.0023 [−0.0031, 0.0073] | +0.0534 [+0.0409, +0.0640] | yes |
| audio | sufficiency | −0.0383 [−0.0604, −0.0189] | 0.0089 [−0.0067, 0.0229] | −0.0472 [−0.0584, −0.0364] | yes |

<!-- src: results/reports/faithfulness.md (AUC table); curves in results/reports/faithfulness.csv and figures/faithfulness_curves.png -->

**IG beats the random control on all four curves**, with paired per-example confidence intervals that exclude zero in
the faithful direction. A negative audio-sufficiency AUC means that keeping only the top-attributed audio frames
*raises* the predicted-class probability above that of the full input (the drop is negative on average).

**Randomisation sanity check (Adebayo-style).** The classifier head (`model.mlp`) of a copy was re-initialised, IG
recomputed for the same target, and the per-example Spearman correlation with the original attributions taken
(20 examples):

| modality | mean Spearman (signed) | mean Spearman (magnitudes) | verdict |
|---|---|---|---|
| text | −0.1432 | 0.2944 | attributions degrade |
| audio | −0.0798 | 0.3272 | attributions degrade |

<!-- src: results/reports/faithfulness.md (sanity table). The BROKEN threshold (signed Spearman ≥ 0.5) is this repo's own choice. -->

Both modalities pass: after the head is randomised the attributions are essentially uncorrelated with the originals,
so IG is sensitive to what the model learned and is not a data-only saliency map.

**Illustrative examples** (full list of 20 with figures in `results/reports/xai_examples.md`). `dev/dia56_utt9`
(Joey: "…I gotta say it quick cause this is killing me", true and predicted sadness, p = 0.645) has the largest gate
on dev (g_audio 0.0692); the five tokens with the largest |score| ("killing" +0.3611, "is" +0.2556, "quick" −0.2347, "say"
+0.2329, "this" +0.2082) all lie inside the current utterance. `dev/dia10_utt8` ("How do you figure that?", neutral, p = 0.615) has g_audio 0.0000, and the
top tokens are "figure" (+0.6147) and "do" (+0.5397).
<!-- src: results/reports/xai_examples.md (examples 1 and 3) -->

**Caveat on convergence.** Mean |IG convergence delta| over the 80 faithfulness passes (40 text + 40 audio) is **0.284**
at n_steps = 64, and none was re-run (the re-run rule only fires below 64 steps). In the 20 examples of
`xai_examples.md` the text deltas range from 8.2e-03 to 1.2 in magnitude while the audio deltas are at most ~3e-06,
so the large deltas come from the text pass. Completeness is therefore only approximately satisfied for text:
**individual token scores should not be over-interpreted**; the aggregate faithfulness result above is the claim
that is supported.
<!-- src: results/reports/faithfulness.md (IG convergence line); results/reports/xai_examples.md (per-example "convergence delta") -->

## 8. Limitations

1. **One seed.** Every number is a single run (seed 42); there is no seed-to-seed variance. The dev/test CIs are
   bootstrap CIs over utterances, not over training runs. The 3-seed protocol in `docs/EVALUATION.md` was not run.
2. **Frozen encoders cap absolute performance** (ED1): LoRA/Stage 2 was not run, so how much of the gap to
   fine-tuned systems it would close is unknown.
3. **IG convergence.** Mean |delta| = 0.284 at 64 steps (text-driven, §7d): token-level scores are noisy;
   only the aggregate faithfulness comparison is supported.
4. **Faithfulness uses deletion operators identical to the IG baselines** (pad embedding / zero frame), which mildly
   favours IG in any cross-method comparison (`docs/KNOWN_LIMITATIONS.md` 14). Only IG was evaluated, so the bias did not
   enter a method comparison here, but it limits how strongly "IG is faithful" can be stated. 40 examples were used.
5. **Ablation uses out-of-distribution inputs.** The model never saw zeroed audio or all-pad text in training, so the
   size of an ablation drop measures reliance under the intervention, not how much information a modality holds.
6. **The mechanism in §7c is not isolated** by a dedicated experiment (see its scope paragraph).
7. **Sadness regression:** M is worse than B1 on sadness on both splits (dev 0.38 → 0.31, test 0.31 → 0.22), a recall loss.
8. **Disgust remains weak** (M test F1 0.14, dev 0.21; dev n = 22, test n = 68) and fear is low (0.21 test); both
   rest on very few examples and have no per-class confidence intervals.
9. **The dev advantage is an upper estimate** (selection on dev, §5.3); the test gain (+0.0158 weighted-F1) is small.
   McNemar tests accuracy, not weighted-F1.
10. **Baselines hit the 10-epoch cap** with dev weighted-F1 still creeping up (§4); B3 (concat) and the E-series
    ablations were not run, so the M-vs-B3 comparison named in `docs/EVALUATION.md` is missing.
11. **Domain:** MELD is scripted US sitcom English with a small recurring cast, and episodes/speakers recur across
    splits (156 episode pairs, §2); no generalisation beyond it is claimed.
12. **Provenance:** the three runs record git SHA `efb2451…` (the code at training time); the ablation and
    faithfulness scripts were added afterwards and do not change the model. No script that wrote
    `results/reports/statistics.md` is committed; its numbers are reproducible exactly with the committed functions (§9).
13. **Documentation inaccuracy:** the "≈2.3 M trainable parameters" in the architecture docs is outdated; the
    implemented model has 3,688,969 (§3).
14. **Not built:** occlusion, attention attribution, the Gemini benchmark, ONNX export and the demo UI.

## 9. Reproduction

Everything below runs from the repository root. `results/` is a copy of the corresponding `artifacts/` outputs
(runs, reports, attributions) of the Kaggle session that produced this report.

```bash
# 0. Environment
pip install -e ".[dev,xai]"
export MELD_ROOT=/path/to/MELD.Raw          # or pass --meld-root to each data script

# 1. Data: validate metadata + media, extract 16 kHz WAVs, write manifests (artifacts/manifests/*.jsonl)
python scripts/inspect_meld.py  --meld-root "$MELD_ROOT"
python scripts/prepare_data.py  --stage full --split all --meld-root "$MELD_ROOT"

# 2. Audio feature cache (once per encoder checkpoint; test only because the final test evaluation is run)
for s in train dev test; do
  python scripts/build_cache.py --manifest artifacts/manifests/$s.jsonl --config configs/base.yaml
done

# 3. Train B1, B2, M (seed 42, at most 10 epochs, best-dev checkpoint kept)
for m in text_only audio_only xattn; do
  python scripts/train.py --config configs/$m.yaml \
      --train-manifest artifacts/manifests/train.jsonl --dev-manifest artifacts/manifests/dev.jsonl \
      --seed 42 --epochs 10 --device auto
done                                        # -> artifacts/runs/<model>-seed42-<UTC timestamp>/

# 4. Dev and test evaluation of each best checkpoint (writes predictions_<split>.jsonl and reports in the run dir)
RUN=artifacts/runs/<model>-seed42-<timestamp>
python scripts/evaluate.py --ckpt $RUN/best.pt --config configs/<model>.yaml \
    --manifest artifacts/manifests/test.jsonl --split test --allow-test     # test: once per final model
python scripts/summarize_runs.py            # artifacts/reports/pilot_results.md (= results/reports/final_results.md)

# 5. Explainability on M (dev)
XA=artifacts/runs/xattn-seed42-<timestamp>/best.pt
python scripts/run_xai.py          --ckpt $XA --config configs/xattn.yaml        # gate_analysis.md, gates.csv, xai_examples.md, attributions
python scripts/run_ablation.py     --ckpt $XA --config configs/xattn.yaml        # ablation.md/.csv, per-mode predictions
python scripts/run_faithfulness.py --ckpt $XA --config configs/xattn.yaml --n-examples 40 --n-steps 64

# 6. Presentation figures from the committed artifacts only
python scripts/make_final_figures.py --results-dir results
```

Bootstrap CIs and McNemar p-values (`results/reports/statistics.md`) are computed from the saved predictions; this
snippet reproduces them exactly (verified against the committed files):

```python
from emorec.training.metrics import bootstrap_ci, load_predictions, align_predictions, mcnemar
R = "results/runs/"                      # one directory per model: <model>-seed42-<timestamp>
_, y, pred, _ = load_predictions(R + "xattn-seed42-20261006T070547Z/predictions_test.jsonl")
bootstrap_ci(y, pred, "weighted_f1", n=1000, seed=42)            # {'point', 'lo', 'hi'} -> 0.6173 [0.5973, 0.6367]
y, a, b = align_predictions(R + "xattn-seed42-20261006T070547Z/predictions_test.jsonl",
                            R + "text_only-seed42-20261006T064248Z/predictions_test.jsonl")
mcnemar(y, a, b)                                                 # n01=240, n10=191, p=2.067e-02
```

The checkpoints (`best.pt`) and the audio cache are not committed (`.gitignore`); they must be regenerated or
copied from the Kaggle session. Training is seeded but not bit-exact across hardware (frozen WavLM's group norm makes
raw-audio batching composition-sensitive; the cache path avoids this).
<!-- src: results/runs/*/run_meta.json; docs/CLAUDE_CODE_HANDOFF.md section 4 -->

## 10. Course mapping

### Speech and Language Processing
- **Dialogue context:** the k = 2 context window with a `<cur>` span that makes the target utterance
  locatable by token id (§2); context turns are encoded jointly with the current one.
- **Audio preprocessing:** 16 kHz mono extraction, −23 LUFS loudness normalisation with short/silent clip guards,
  and the 10 s cap that keeps the end of the utterance because utterance-final prosody carries emotion (ED5) (§2).
- **Prosody / class findings:** audio alone is weak (B2 test weighted-F1 0.4273, F1 0.00 on fear and disgust) yet the
  fused model improves the rare classes (test fear 0.10 → 0.21, disgust 0.00 → 0.14); the ablation shows audio
  reliance that the gate does not (§6, §7b). The sadness regression is a documented counter-example.

### Deep Learning
- **Architecture:** frozen RoBERTa + WavLM, projections, bidirectional masked cross-attention blocks, attention pooling,
  scalar gate, MLP head; the padding-invariance invariant (INV1) and frozen-parameter invariant (INV3) are tested (§3).
- **Training:** weighted cross-entropy for 7-class imbalance, AdamW with cosine/warmup, accumulation, AMP, early stopping
  on dev weighted-F1, an audio-feature cache for speed (§4).
- **Baselines and comparison:** B1/B2/M share encoders and differ only in fusion; all are compared with the all-neutral
  baseline (§5).
- **Statistics:** bootstrap 95% CIs, exact McNemar on paired predictions, and the reasoning for why the paired test is the
  right instrument when marginal CIs overlap (§5.2); the dev-test gap and selection effect are discussed (§5.3).

### Explainable AI
- **Gate:** an intrinsic interpretability signal, found text-saturated (§7a).
- **Ablation:** shows the gate does not reflect information flow, with a shuffle control (§7b, §7c).
- **Integrated Gradients:** per-token and per-frame attributions with convergence diagnostics (§7d).
- **Faithfulness:** comprehensiveness and sufficiency against a random control, with paired CIs (§7d).
- **Sanity check:** parameter randomisation of the classifier head; attributions degrade (§7d).
- **Honest limits:** IG deltas, out-of-distribution ablation inputs, and the unisolated mechanism (§8).

## Figures

Regenerated by `scripts/make_final_figures.py` from the committed artifacts (all in `results/reports/figures/`):

| figure | content |
|---|---|
| `final_per_class_f1.png` | per-class F1, B1/B2/M, dev and test (§6) |
| `final_dev_vs_test_wf1.png` | weighted-F1 with bootstrap CIs, dev vs test, with the all-neutral dev line (§5) |
| `final_ablation.png` | modality ablation with the control highlighted, CIs and McNemar p (§7b) |
| `final_gate_distribution.png` | g_audio histogram on a log axis and per-class means (§7a) |

Produced earlier by the experiment scripts: `ablation.png`, `gate_distribution.png`, `faithfulness_curves.png`, and
per-example `dev_*_tokens.png` / `dev_*_audio.png` IG figures.
