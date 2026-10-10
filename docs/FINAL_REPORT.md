# An Explainable Deep Learning Framework for Multimodal Emotion Recognition

**Course:** BCSE419L, Speech and Language Processing
**Repository:** https://github.com/maulikrathor/emorec-xmm
**Task:** seven-class Emotion Recognition in Conversation (ERC) on MELD from speech and transcribed text

*Provenance convention.* Every number in this report is copied from a committed file under `results/` (or, where stated, from a committed document under `docs/`). The source is named in an HTML comment beside each table (visible in the raw Markdown). Differences between two cited values are plain subtraction. Anything that was not measured is stated as not measured.

---

## 1. Introduction

### 1.1 Emotion recognition in conversation as a speech and language task

Emotion Recognition in Conversation (ERC) assigns an emotion label to each utterance of a multi-party dialogue. It is a speech and language problem in three respects. First, the *linguistic content* of an utterance (lexical choice, negation, intensifiers, interjections) carries much of the affective signal. Second, the *conversational context* matters: the same words can be read as sarcastic, relieved or angry depending on the turns that precede them, and speakers react to one another. Third, the *acoustic realisation* of the utterance, its prosody and voice quality, carries affect that the words alone do not encode. Pitch level and range, intensity, speaking rate, and the shape of the final pitch contour are among the vocal correlates of emotion that have been studied for decades [12].

A transcript discards all of this third channel. A one-word turn such as "Okay!" or "Ow!" can express joy, irritation, pain or surprise, and the orthography preserves only a fraction of the distinction (an exclamation mark, a question mark). A text-only ERC system is therefore structurally blind to information that a listener uses. At the same time, it is an empirical question how much of that information a model can actually recover from real conversational audio, which in the MELD corpus [1] is sitcom speech with laugh tracks, overlapping speakers and background music.

### 1.2 Research questions

This project asks two questions.

**RQ1 (benefit of the speech channel).** Does adding the speech signal measurably improve ERC over a strong text baseline, when both are evaluated on MELD's official splits with all seven emotion classes and compared with a paired statistical test?

**RQ2 (per-utterance attribution of modality).** Can we identify, for an individual utterance, which channel (speech or text) drove the model's decision, and can the explanation methods used to answer this be shown to be faithful to the model rather than merely plausible?

### 1.3 Contribution

The contribution is a controlled reproduction and explainability audit of a published RoBERTa plus WavLM framework for multimodal emotion recognition [2], with the following properties:

1. The same pair of pretrained encoders (RoBERTa-base for text [5], WavLM-base-plus for speech [4]) is evaluated on **MELD's official train, dev and test splits with all seven classes**, under a protocol in which the test split is touched once, for the final models only.
2. The fusion stage is extended from concatenation to **bidirectional gated cross-attention**, so that each modality can attend to the other before pooling, and a scalar modality gate is exposed as an intrinsic interpretability signal.
3. The explanation layer is **measured, not assumed**: Integrated Gradients [6] is evaluated for faithfulness against a random-attribution control and with a parameter-randomisation sanity check [7], and the gate is tested against inference-time modality ablation.

The headline findings are proportionate. The speech channel produces a statistically significant but modest test-set improvement over the text baseline, concentrated in rare classes, with a regression on sadness. The model's intrinsic gate claims near-total reliance on text, and modality ablation shows that this claim is false: the gate does not reflect information flow in this architecture. The code, configurations, committed results and tests are in the project repository (Figure 1).

> **[SCREENSHOT 1]**
> **What to capture:** The GitHub repository landing page showing the file tree (`src/`, `scripts/`, `configs/`, `docs/`, `results/`, `tests/`) and the latest commit message.
> **Where to find it:** https://github.com/maulikrathor/emorec-xmm in a browser, branch `main`; switch the browser or GitHub theme to light mode and convert the capture to greyscale before inserting.
> **Caption to use:** Figure 1. Project repository on GitHub, showing the source package, experiment scripts, configurations, committed results and test suite.

---

## 2. Literature Review

### 2.1 The MELD corpus

**Poria et al. (2019), MELD [1].** MELD (Multimodal EmotionLines Dataset) extends the text-only EmotionLines corpus with the audio and video of each utterance, taken from the television series *Friends*. Each utterance is labelled with one of seven emotions (neutral, joy, sadness, anger, surprise, fear, disgust) and with sentiment, and the corpus is distributed with official dialogue-level train, dev and test splits. MELD is multi-party (more than two speakers per dialogue is common) and strongly imbalanced toward neutral.

*Relation to this project.* MELD is the corpus used throughout, with its official splits unmodified and all seven classes retained. The pipeline validates the distributed metadata and media against each other before any model is trained (Section 3.1), because the corpus as mirrored contains duplicated filename stems across splits and operating-system artefacts that would otherwise corrupt the audio lookup.

### 2.2 Modelling conversational context

**Majumder et al. (2019), DialogueRNN [13].** DialogueRNN models a conversation with recurrent units that track the global dialogue state, each party's state and the emotional state of the current speaker, and showed that speaker-aware context modelling improves ERC on MELD and IEMOCAP.

*Relation to this project.* This project uses a simpler, encoder-native form of context: the preceding k turns are concatenated into the text input with speaker tags and the current utterance is delimited by special marker tokens (Section 3.3). With frozen encoders, Section 4.3 finds that this form of context does not help, which is consistent with the view that context must be modelled, not merely concatenated.

### 2.3 Speech and text representations

**Chen et al. (2022), WavLM [4].** WavLM is a self-supervised speech model trained with masked speech prediction *and* a denoising objective in which training utterances are mixed with overlapping speech or noise. It was designed for "full stack" speech processing, including speaker-related and paralinguistic tasks, not only automatic speech recognition, and reported strong results on the SUPERB benchmark, which includes emotion recognition.

*Relation to this project.* WavLM-base-plus is the speech encoder. Its denoising pretraining is relevant to MELD, whose audio contains laugh tracks and overlapping speakers, and its paralinguistic orientation makes it a better fit for affect than an encoder optimised purely for phonetic content.

**Liu et al. (2019), RoBERTa [5].** RoBERTa is a robustly optimised BERT pretraining recipe (more data, longer training, dynamic masking, no next-sentence objective) with a byte-level BPE vocabulary. It is a standard strong text encoder for utterance classification.

*Relation to this project.* RoBERTa-base is the text encoder, applied to the dialogue-context string. Two new special tokens mark the current utterance.

**Scherer (2003) [12].** Scherer's review of vocal emotion communication summarises the acoustic correlates of emotion (fundamental frequency level, range and contour, intensity, speech rate, spectral energy distribution) and their relation to arousal and valence.

*Relation to this project.* This is the theoretical motivation for the speech channel and for the end-preserving audio truncation (Section 3.2). It also frames the per-class results: classes that differ mainly in arousal are the ones where acoustic evidence would be expected to help.

### 2.4 Multimodal fusion for emotion recognition

**Tsai et al. (2019), Multimodal Transformer [19].** MulT introduced directional cross-modal attention between unaligned modality sequences, letting each modality attend to the time steps of another without forced word-level alignment.

*Relation to this project.* The fusion blocks used here are bidirectional cross-modal attention in this spirit, applied to frame-level speech features and subword-level text features that are not time-aligned.

**Yun et al. (2024), TelME [3].** TelME is a teacher-leading fusion network in which a strong text teacher distils knowledge into the weaker audio and visual students before fusion. It reports a weighted-F1 of 67.4 on MELD under the standard seven-class protocol with official splits.

*Relation to this project.* TelME is the reference point for what the official MELD protocol supports with trainable models. It is not a like-for-like comparison: TelME fine-tunes its encoders and uses the visual channel, whereas this project keeps both encoders frozen and uses speech and text only.

**Pandey, Singh and Kaur (2025) [2], the base paper.** This work proposes an explainable multimodal transformer that combines RoBERTa text features with WavLM acoustic features, projects both into a shared latent space, fuses them by concatenation, and applies Integrated Gradients and occlusion to attribute predictions to tokens and prosodic regions. It reports an accuracy of 0.83 over five emotion categories, after removing the disgust and fear classes, on a random 80/20 split of the data rather than MELD's official splits. Its statistical comparison reports a Wilcoxon signed-rank p of 0.125 and a sign-test p of 0.25.

*Relation to this project.* The base paper fixes the encoder pair and the explainability intent that this project adopts. The differences are methodological, and they explain why the numbers are not directly comparable (Table 1). A random utterance-level split of a dialogue corpus can place turns of the same dialogue, scene and speaker in both training and evaluation data, whereas MELD's official splits are drawn dialogue-wise. Removing disgust and fear removes the two rarest and hardest classes. Neither reported test reaches the conventional 0.05 threshold. This project therefore keeps all seven classes and the official splits, and compares models with an exact McNemar test on paired per-utterance predictions over the full test split.

### 2.5 Explainability methods and their evaluation

**Sundararajan, Taly and Yan (2017), Integrated Gradients [6].** Integrated Gradients (IG) attributes a prediction to input features by integrating the gradient along a straight path from a baseline input to the actual input. It satisfies the *completeness* axiom: attributions sum to the difference between the model's output at the input and at the baseline. The numerical gap between the two (the convergence delta) is a diagnostic of how well the path integral has been approximated.

*Relation to this project.* IG is the post-hoc attribution method for both modalities, implemented with Captum [9] over the model's embedding-level forward pass (Section 3.8).

**Adebayo et al. (2018), Sanity Checks for Saliency Maps [7].** This work showed that some saliency methods produce nearly the same maps after the model's parameters are randomised, which means they reflect the input rather than what the model learned. It proposed the parameter-randomisation test as a minimum requirement.

*Relation to this project.* The test is applied to IG by re-initialising the classifier head and measuring the rank correlation between original and randomised attributions (Section 4.4).

**DeYoung et al. (2020), ERASER [8].** ERASER defined the *comprehensiveness* (deleting the most important features should lower confidence) and *sufficiency* (keeping only the most important features should preserve confidence) metrics for rationale faithfulness in NLP.

*Relation to this project.* Both metrics are computed as deletion curves for text tokens and speech frames, against a random-attribution control.

**Jain and Wallace (2019), Attention is not Explanation [14].** This work showed that attention weights frequently fail to correlate with gradient-based importance and that alternative attention distributions can yield the same predictions.

*Relation to this project.* The modality gate in this model is a learned weighting of the same kind. Section 4.4 tests it directly and finds a comparable failure: the gate value does not reflect how much the model depends on the speech channel.

### 2.6 Positioning

Table 1 summarises the methodological differences between the base paper and this project.

<!-- source: Pandey, Singh and Kaur (2025) [2] as published (base-paper column); results/reports/meld_metadata.md, results/reports/statistics.md, results/reports/faithfulness.md, results/reports/ablation.md (this-project column) -->

**Table 1.** Methodological comparison with the base paper.

| aspect | base paper [2] | this project |
|---|---|---|
| encoders | RoBERTa, WavLM | RoBERTa-base, WavLM-base-plus, both frozen |
| classes | 5 (disgust and fear removed) | 7 (all MELD classes) |
| data split | random 80/20 | MELD official train / dev / test |
| fusion | concatenation in a shared latent space | two bidirectional cross-modal attention blocks, then a scalar modality gate |
| headline metric | accuracy 0.83 | weighted-F1 (primary), with macro-F1, per-class F1, accuracy and confusion matrices |
| model comparison | Wilcoxon p = 0.125, sign test p = 0.25 | exact McNemar on paired predictions, plus bootstrap 95% CIs |
| explanation methods | IG, occlusion | IG, intrinsic gate, modality ablation |
| explanation evaluation | not compared in this report | comprehensiveness and sufficiency vs random control; parameter-randomisation check |

---

## 3. Methodology

### 3.1 Dataset and corpus validation

MELD distributes 13,708 labelled utterances across the official splits: 9,989 train, 1,109 dev and 2,610 test. Before any audio is extracted, the pipeline validates the CSV metadata (row counts, label vocabulary, identifier uniqueness) and reconciles every labelled row with a media file. Two utterances are lost in this process: one dev utterance has no media file, and one train utterance failed ffmpeg extraction. The usable sets are therefore 9,988 train, 1,108 dev and 2,610 test utterances (Table 2).

<!-- source: results/reports/meld_metadata.md (Row counts); results/reports/media_inventory.md (dev: labeled_rows 1109, matched 1108, missing_media 1); results/context_ablation/README.md (full MELD train 9,988, dev 1,108); results/runs/*/classification_report_test.txt (test support total 2610) -->

**Table 2.** Labelled and usable utterances per split.

| split | labelled utterances | usable after media validation | lost |
|---|---|---|---|
| train | 9,989 | 9,988 | 1 (ffmpeg extraction failure) |
| dev | 1,109 | 1,108 | 1 (no media file) |
| test | 2,610 | 2,610 | 0 |
| total | 13,708 | 13,706 | 2 |

**Media reconciliation.** The corpus mirror used contains 15,908 `.mp4` files, which is more than the number of labelled utterances. The reconciliation (Table 3) shows that 2,060 of these files, all in the test directory, are macOS AppleDouble resource-fork sidecars (files named `._*.mp4` that contain file-system metadata, not media). The remaining 13,848 are real media; 13,707 of them match a labelled row, 141 are orphans with no label (4 in dev, 137 in test), and they are counted but never labelled or deleted. The arithmetic closes exactly: 13,848 real files plus 2,060 sidecars equals 15,908.

<!-- source: results/reports/media_inventory.md (Per split; Conclusion) -->

**Table 3.** Media inventory reconciliation.

| quantity | train | dev | test | total |
|---|---|---|---|---|
| `.mp4` files found | 9,989 | 1,112 | 4,807 | 15,908 |
| AppleDouble sidecars | 0 | 0 | 2,060 | 2,060 |
| real media files | 9,989 | 1,112 | 2,747 | 13,848 |
| matched to a labelled row | 9,989 | 1,108 | 2,610 | 13,707 |
| orphan media (no label) | 0 | 4 | 137 | 141 |
| labelled rows without media | 0 | 1 | 0 | 1 |

**Split-scoped media lookup.** MELD names media files by dialogue and utterance index (for example `dia0_utt0`), and these indices restart in every split. 2,068 filename stems occur in more than one split directory; `dia0_utt0`, for instance, exists in train, dev and test. A global filename-to-file lookup would therefore attach the wrong speech recording to an utterance without raising any error. The pipeline matches media within each split directory only and namespaces every utterance identifier by split (`dev/dia0_utt0`).

<!-- source: results/reports/media_inventory.md (Cross-directory duplicate stems: 2068) -->

**Class distribution.** Table 4 gives the distribution of the seven emotion labels. Neutral accounts for 42 to 48 percent of each split, while fear and disgust each account for roughly 2 to 4 percent. This imbalance is the central difficulty of the task and is the reason weighted-F1 is always reported together with macro-F1 and per-class F1.

<!-- source: results/reports/meld_metadata.md (R1, per-split emotion distribution; labelled rows) -->

**Table 4.** Emotion class distribution of labelled utterances (count and share of split).

| class | train | dev | test |
|---|---|---|---|
| neutral | 4,710 (47.15%) | 470 (42.38%) | 1,256 (48.12%) |
| joy | 1,743 (17.45%) | 163 (14.70%) | 402 (15.40%) |
| sadness | 683 (6.84%) | 111 (10.01%) | 208 (7.97%) |
| anger | 1,109 (11.10%) | 153 (13.80%) | 345 (13.22%) |
| surprise | 1,205 (12.06%) | 150 (13.53%) | 281 (10.77%) |
| fear | 268 (2.68%) | 40 (3.61%) | 50 (1.92%) |
| disgust | 271 (2.71%) | 22 (1.98%) | 68 (2.61%) |

Utterances are short: the mean length is 7.96 whitespace tokens in train (median 6), 7.92 in dev and 8.22 in test, and dialogues average 9.62 turns in train. The metadata audit also found that 156 (Season, Episode) pairs occur in more than one split and that 5 (Season, Episode, StartTime, EndTime) keys appear in two splits. MELD's official splits are drawn dialogue-wise from overlapping episodes, so the same cast recurs across splits; this is a property of the corpus, not of this pipeline. The automated metadata report and the media reconciliation are shown in Figure 2 and Figure 3.

<!-- source: results/reports/meld_metadata.md (R2, R3, R4, R5) -->

> **[SCREENSHOT 2]**
> **What to capture:** The rendered MELD metadata report showing the row counts and the per-split emotion distribution (sections "Row counts" and "R1").
> **Where to find it:** `results/reports/meld_metadata.md`, opened on GitHub (https://github.com/maulikrathor/emorec-xmm/blob/main/results/reports/meld_metadata.md) in light mode; capture from the title to the end of the test distribution, convert to greyscale.
> **Caption to use:** Figure 2. Automated MELD metadata validation: row counts and per-split emotion distribution.

> **[SCREENSHOT 3]**
> **What to capture:** The "Per split" block for the test split and the "Conclusion" block of the media inventory report, showing 15,908 = 13,848 + 2,060.
> **Where to find it:** `results/reports/media_inventory.md` on GitHub (https://github.com/maulikrathor/emorec-xmm/blob/main/results/reports/media_inventory.md), light mode, greyscale.
> **Caption to use:** Figure 3. Media inventory reconciliation: 2,060 AppleDouble sidecar files and 141 orphan media files account exactly for the gap between files on disk and labelled utterances.

### 3.2 Speech processing pipeline

Every usable utterance is converted from its video container to a speech waveform suitable for a self-supervised speech encoder. Table 5 lists the parameters.

<!-- source: results/context_ablation/xattn-seed42-20261010T061528Z/config.yaml (data.sample_rate, data.target_lufs, data.min_lufs_duration_s, data.lufs_clamp_db, data.max_audio_sec); docs/ARCHITECTURE.md (audio input contract) -->

**Table 5.** Speech preprocessing parameters.

| step | setting |
|---|---|
| extraction | ffmpeg, audio stream to mono PCM WAV |
| sample rate | 16,000 Hz (the rate WavLM was pretrained on) |
| loudness target | -23 LUFS integrated loudness |
| short-clip guard | clips shorter than 0.4 s are not loudness-normalised |
| gain clamp | loudness correction limited to 20 dB |
| silent-clip guard | clips with no measurable loudness are not gain-adjusted |
| maximum duration | 10.0 s, truncated by keeping the **end** of the utterance |

**Resampling and channel reduction.** WavLM was pretrained on 16 kHz single-channel speech, and feeding it audio at a different rate would shift every spectral cue the encoder learned. The pipeline therefore downmixes the audio track of each MELD video file to mono and resamples it to 16 kHz once, at extraction time, and both the dataset loader and the speech encoder refuse audio at any other rate.

**Loudness normalisation.** Recording level in a sitcom varies with microphone placement and scene, and raw level is a confound: a model could learn that "loud recording" means "anger" even when the speaker is not shouting. Each clip is normalised to -23 LUFS, the integrated-loudness target of the EBU R 128 broadcast recommendation [15], so that *relative* intensity within an utterance remains while absolute recording level is removed. Two guards make this safe on real data. Integrated loudness is measured over 400 ms gating blocks, so for clips shorter than 0.4 s the measurement is undefined and the clip is left unnormalised rather than receiving an arbitrary gain. A clip with no measurable loudness (silence) is likewise left unchanged, and any correction is limited to 20 dB so that a near-silent clip cannot be amplified into noise.

**End-keeping truncation.** Utterances longer than 10 s are truncated to their final 10 s; the start is discarded. The choice is deliberate. Affective information is not uniformly distributed across an utterance: the final pitch movement (the boundary tone), phrase-final lengthening and the final stressed syllable are prosodically prominent, and in conversational speech the emotionally loaded part of a turn (the punchline, the complaint, the exclamation) often comes at its end. Keeping the start would preserve the speaker's preamble and discard the part of the contour most likely to signal affect. Truncation is applied when the waveform is loaded, not at extraction, so the full extracted audio remains available.

### 3.3 Dialogue context modelling

The text input is a single string that contains the k preceding turns of the dialogue followed by the current utterance. Each turn is prefixed with its speaker's name in square brackets, turns are separated by RoBERTa's separator token `</s>`, and the current utterance is wrapped in two new special tokens, `<cur>` and `</cur>`. The format for k = 2 is:

```
[spk(t-2)]: u(t-2) </s> [spk(t-1)]: u(t-1) </s> <cur> [spk(t)]: u(t) </cur>
```

The current utterance is always last and always delimited, so that any token-level analysis can separate target tokens from context tokens. At the start of a dialogue, when fewer than k turns precede the target, only the available turns are used and nothing is padded. The context builder is a pure function of the dialogue, the turn index and k, and its output is fixed by five golden strings in the test suite; for k = 2 at the third turn of a toy dialogue it produces `[Chandler]: Hey. </s> [Monica]: Hi there! </s> <cur> [Ross]: I'm fine. </cur>`. The default is k = 2, and the tokenised string is limited to 160 tokens (`max_text_len`).

<!-- source: docs/ARCHITECTURE.md (Text input contract); results/context_ablation/xattn-seed42-20261010T061528Z/config.yaml (context_k 2, max_text_len 160, cur_token, end_cur_token, context_sep) -->

**A real example.** For dev utterance `dev/dia10_utt8`, spoken by Chandler, the current-utterance span of the context string is

```
<cur> [Chandler]: How do you figure that? </cur>
```

preceded by the two previous turns of dialogue 10 in the k = 2 format above; the full string, as the model receives it, is shown in Figure 4. The marker tokens are what allow the attribution report to tag each token as belonging to the current utterance or to the context: for this utterance, the five tokens with the largest Integrated Gradients scores are `figure` (+0.6147, current), `do` (+0.5397, current), `that` (-0.3507, current), `you` (+0.1923, current) and `two` (+0.1758, context). The attribution report also shows how speaker names are tokenised: "Phoebe" appears as the byte-level BPE pieces `Ph` and `oe` in several examples, and the bracket and colon of the speaker tag appear as tokens of their own.

<!-- source: results/reports/xai_examples.md (example 3, dev/dia10_utt8; examples 4, 12 and 19 for the Ph / oe pieces) -->

> **[SCREENSHOT 4]**
> **What to capture:** The terminal output of the demo predictor for `dev/dia10_utt8`, cropped to the `UTT_ID`, `SPEAKER`, `TEXT` and `CONTEXT STRING` lines, so that the full k = 2 context string with its `</s>` separators and `<cur> ... </cur>` markers is visible.
> **Where to find it:** On the machine that holds the checkpoints (Kaggle session or local copy), run `python scripts/demo_predict.py --ckpt artifacts/runs/xattn-seed42-20261006T070547Z/best.pt --config configs/xattn.yaml --utt-id dev/dia10_utt8` from the repository root; use a light terminal theme and convert to greyscale.
> **Caption to use:** Figure 4. A real k = 2 dialogue-context string from the MELD dev split, with speaker tags, turn separators and the current-utterance markers.

### 3.4 Encoders

**Text encoder.** RoBERTa-base [5] encodes the context string into a sequence of 768-dimensional token states. Its weights are frozen. The only trainable parameters inside the text encoder are the embedding rows of the two new tokens `<cur>` and `</cur>` (2 x 768 = 1,536 parameters); a gradient hook zeroes the gradient of every other embedding row, so the pretrained vocabulary is untouched.

**Speech encoder.** WavLM-base-plus [4] encodes the raw 16 kHz waveform. Its convolutional feature extractor downsamples the waveform to approximately 49 to 50 frames per second, so a 10 s clip yields at most about 500 frames of 768-dimensional features. Its weights are frozen and the encoder is held permanently in evaluation mode, so dropout is never active.

**Frame-level attention mask.** Utterances in a batch have different durations, so waveforms are zero-padded to the longest one and a sample-level mask marks the real samples. The fusion layers, however, operate on *frames*, not samples. The frame-level mask is derived by the speech model's own output-length computation (the convolution kernel and stride arithmetic of its feature extractor, via the installed `transformers` WavLM implementation), not by an approximate division, so that exactly the frames that correspond to real speech are marked valid. This mask is used in every cross-attention and pooling step, and a test enforces that the model's logits are invariant to how much padding surrounds an utterance (tolerance 1e-5).

**Why the encoders are frozen.** Three reasons, in order of importance.
1. *Controlled comparison.* The text-only baseline, the speech-only baseline and the multimodal model share identical, fixed encoders and differ only in the fusion stage. Any difference between them is therefore attributable to fusion, not to one model having fine-tuned its encoder further than another.
2. *Feature caching.* With a frozen speech encoder, the WavLM features of every utterance can be computed once and cached to disk (16-bit floating point, one file per utterance, keyed by the encoder checkpoint name). This matters for a second, speech-specific reason: WavLM's first convolutional layer uses group normalisation over the whole padded waveform, so features computed from a padded batch depend slightly on the other clips in that batch. Caching each utterance's features unpadded and re-padding at the frame level removes this dependence, and the cached path is the one used for training.
3. *Tractable compute.* Training touches only the fusion layers (Section 3.5), which made full-training-set runs feasible on a single Kaggle T4-class GPU. The cost is a cap on absolute performance (Section 5.2).

### 3.5 Fusion architecture

Figure 5 shows the multimodal model.

```
            text context string                     16 kHz waveform (<= 10 s, end kept)
                    |                                            |
     +------------------------------+          +------------------------------+
     | RoBERTa-base (frozen)        |          | WavLM-base-plus (frozen)     |
     | + trainable <cur>, </cur>    |          | cached frame features        |
     +------------------------------+          +------------------------------+
           (B, T_text, 768)                          (B, T_frames, 768)
                    |                                            |
           linear proj_t -> 256                        linear proj_a -> 256
                    |                                            |
                    +--------------------+-----------------------+
                                         |
                 Cross-modal block 1: text attends to speech frames,
                                      speech frames attend to text
                 (8 heads, pre-LayerNorm, residual, feed-forward,
                  padding masked in both directions)
                                         |
                 Cross-modal block 2: same structure
                                         |
                    +--------------------+-----------------------+
                    |                                            |
      masked attention pooling                      masked attention pooling
             z_t (256)                                     z_a (256)
                    |                                            |
                    +--------------------+-----------------------+
                                         |
                   gate  g = softmax(Linear([z_t ; z_a])) = (g_text, g_audio)
                                         |
                   fused = [ g_text * z_t ; g_audio * z_a ]   (512)
                                         |
                   MLP 512 -> 256 -> 7  ->  emotion logits
```

**Figure 5.** Gated bidirectional cross-attention fusion model (block diagram). The modality gate is applied *after* the cross-modal blocks; Section 4.4 shows why this ordering matters.

<!-- source: docs/ARCHITECTURE.md (System at a glance); results/context_ablation/xattn-seed42-20261010T061528Z/config.yaml (d_model 256, n_heads 8, n_fusion_blocks 2, num_classes 7, freeze_encoders true) -->

The components are as follows.

- **Projections.** Each modality's 768-dimensional states are linearly projected to a shared width d = 256.
- **Bidirectional cross-modal attention.** Two blocks follow. In each, the text stream attends to the speech frames and the speech stream attends to the text tokens, using multi-head attention [16] with eight heads, pre-layer normalisation, residual connections and a position-wise feed-forward layer. Padding is masked in both directions, so a padded frame can never be attended to by a token and vice versa. Text and speech are not time-aligned (subword tokens against 20 ms frames), and cross-attention lets the model learn soft alignments rather than requiring forced alignment.
- **Masked attention pooling.** Each stream is summarised into a single 256-dimensional vector (z_t for text, z_a for speech) by a learned query that attends over the valid positions only.
- **Scalar modality gate.** A linear layer over the concatenation [z_t ; z_a] followed by a softmax produces two weights (g_text, g_audio) that are positive and sum to one. The fused representation is [g_text * z_t ; g_audio * z_a]. The gate is returned with every prediction and was intended as an intrinsic, per-utterance indicator of which modality the model relied on.
- **Classifier.** An MLP maps the 512-dimensional fused vector through a 256-unit hidden layer to seven emotion logits.

The model has **3,688,969 trainable parameters**[^params]: 3,687,433 in the fusion side (two projections of 196,864 each, two cross-modal blocks of 3,159,040 together, two pooling queries of 256 each, a gate of 1,026 and an MLP of 133,127) plus 1,536 in the two new token embeddings. The two baselines reuse the same encoders and projections: the text-only baseline (B1) pools the projected text stream and classifies it with the same MLP form, and the speech-only baseline (B2) does the same for the speech stream. Neither baseline has cross-attention or a gate.

[^params]: `docs/ARCHITECTURE.md` states "approximately 2.3M trainable parameters". The implemented model has 3,688,969, obtained by counting the parameters of the instantiated architecture; the difference comes mainly from the feed-forward layers inside the cross-modal blocks. The documented figure is an outdated design estimate and does not affect any result.

> **[SCREENSHOT 5]**
> **What to capture:** Figure 5 (the block diagram above) as rendered, or a redrawn black-and-white version of it with the same boxes, labels and arrows.
> **Where to find it:** This file, `docs/FINAL_REPORT.md`, Section 3.5, rendered on GitHub (https://github.com/maulikrathor/emorec-xmm/blob/main/docs/FINAL_REPORT.md) in light mode; alternatively redraw it in any diagram tool using black lines and text only.
> **Caption to use:** Figure 5. Gated bidirectional cross-attention fusion over frozen RoBERTa-base and WavLM-base-plus encoders.

### 3.6 Training protocol

All three models (B1 text-only, B2 speech-only, M multimodal) are trained with the same protocol (Table 6).

<!-- source: results/context_ablation/*/config.yaml and run_meta.json (train.*, loss.name weighted_ce, run.seed 42); docs/ARCHITECTURE.md (loss weighting); results/reports/final_results.md (epochs run) -->

**Table 6.** Training configuration.

| item | setting |
|---|---|
| loss | class-weighted cross-entropy; weight of class c proportional to 1 / sqrt(train count of c), normalised to mean 1 |
| optimiser | AdamW, learning rate 2e-4, weight decay 0.01 |
| schedule | cosine decay with linear warmup over the first 10% of optimiser steps |
| precision | automatic mixed precision (AMP) |
| batch | 8 utterances, gradient accumulation over 4 steps (effective batch 32), gradient clipping at 1.0 |
| epochs | at most 10 |
| early stopping | patience 3 on dev weighted-F1; the best dev checkpoint is kept |
| seed | 42 (one seed per model) |
| training data | full train split, 9,988 utterances; cached speech features |

The inverse-square-root weighting is a compromise: inverse-frequency weights would multiply the loss of a fear example by about 17.6 relative to a neutral one (4,710 / 268 training utterances) and destabilise training, while unweighted cross-entropy lets the model ignore the rare classes. B1 and B2 ran all 10 epochs; M stopped early after 8 epochs with its best dev score at epoch 5.

<!-- source: results/reports/final_results.md (epochs run 10 / 10 / 8); results/context_ablation/README.md (xattn best at ep5); results/reports/meld_metadata.md (train neutral 4,710; fear 268) -->

### 3.7 Evaluation protocol

- **Metrics.** Weighted-F1 (per-class F1 averaged with weights proportional to class support, the convention for MELD) is the primary metric. Macro-F1, per-class F1, accuracy and confusion matrices are always reported alongside it, because weighted-F1 is dominated by neutral and can hide failure on rare classes. An all-neutral majority-class predictor is reported as a floor.
- **Uncertainty.** Each metric carries a bootstrap 95% confidence interval from 1,000 utterance-level resamples of the evaluation set (percentile method) [11].
- **Model comparison.** Two models evaluated on the same utterances are compared with an exact two-sided McNemar test [10] on their paired per-utterance correctness. The test uses only the discordant pairs: n01, the number of utterances the first model gets right and the second wrong, and n10, the reverse. A difference is claimed only when p < 0.05.
- **Split discipline.** Model selection (stopping epoch, checkpoint, every design choice) uses dev only. The test split was evaluated once, for the final B1, B2 and M checkpoints. Scripts refuse to run on a test manifest without an explicit flag. Variants trained later (the context ablation in Section 4.3) were evaluated on dev only.

### 3.8 Explainability methods

All explainability analyses use the final M checkpoint on the 1,108 dev utterances.

**Integrated Gradients.** IG [6] is computed with Captum [9] over the model's embedding-level forward pass (`forward_from_embeds`), so that attributions are taken with respect to continuous inputs: RoBERTa input embeddings for text and WavLM frame features for speech. One modality is attributed per pass while the other is held at its real value. The target is the logit of the *predicted* class, so an explanation describes the decision the model actually made. Baselines are the pad-token embedding at every text position and an all-zero frame for speech. Token scores are summed over the embedding dimension; frame scores are summed over the feature dimension and reported in 0.5 s bins. The integral is approximated with 32 steps and recomputed at 64 when the convergence delta exceeds 5% of the attribution sum; the faithfulness evaluation uses 64 steps throughout. Speech attributions are taken at the encoder's frame output before the projection, a documented deviation from the post-projection attachment point in `docs/ARCHITECTURE.md`; both points are frame-aligned, so the temporal reading is the same.

**Intrinsic gate analysis.** The gate values (g_text, g_audio) are recorded for every dev utterance exactly as produced, with no rescaling, and summarised overall, per true class and by prediction correctness.

**Modality ablation.** The trained model is run on modified inputs with its weights unchanged: `zero_audio` replaces the speech frames with zeros (mask unchanged); `zero_text` replaces every text embedding with the pad embedding; `shuffle_audio` gives each utterance the speech of a different utterance from the same batch (a seeded derangement). The unmodified run is the control, and each ablation is compared with it by McNemar on paired correctness. The shuffle condition is the key control: shuffled speech has the same shape, duration distribution and statistics as real speech, so if the model used speech only as a "speech is present" signal, shuffling would not hurt.

**Faithfulness.** On 40 class-stratified dev examples, positions (tokens or frames) are ranked by IG score and two deletion curves are computed at fractions 0.05, 0.1, 0.2 and 0.5 [8]. *Comprehensiveness* deletes the top-ranked fraction and records the drop in predicted-class probability (higher is more faithful). *Sufficiency* keeps only the top-ranked fraction and records the drop from the full input (lower is more faithful). Deletion replaces a token with the pad embedding and a frame with zeros. Each curve is summarised by its area under the curve (AUC) and compared with a control that ranks positions randomly (mean of five seeded random rankings per example), using a paired per-example bootstrap CI of the IG minus random difference.

**Parameter-randomisation sanity check.** Following Adebayo et al. [7], a copy of the model has its classifier head (the MLP) re-initialised, IG is recomputed for the same target class on 20 examples, and the Spearman rank correlation with the original attributions is measured. An attribution method that is sensitive to what the model learned should produce low correlation; a method is flagged as failing if the mean signed correlation stays at or above 0.5.

---

## 4. Results

### 4.1 Main comparison

Table 7 and Table 8 give the dev and test results of the three models and the all-neutral floor, with bootstrap 95% CIs.

<!-- source: results/reports/statistics.md (Bootstrap 95% CI); results/reports/final_results.md (Trivial baseline, dev) -->

**Table 7.** Dev results (n = 1,108), with bootstrap 95% CIs.

| model | weighted-F1 [95% CI] | macro-F1 [95% CI] | accuracy [95% CI] |
|---|---|---|---|
| all-neutral baseline | 0.2518 | 0.0850 | 0.4233 |
| B2 speech-only | 0.4012 [0.3687, 0.4342] | 0.2339 [0.2103, 0.2576] | 0.4558 [0.4242, 0.4865] |
| B1 text-only | 0.5548 [0.5230, 0.5855] | 0.3760 [0.3500, 0.4021] | 0.5776 [0.5478, 0.6065] |
| **M multimodal** | **0.6155** [0.5832, 0.6448] | **0.4670** [0.4251, 0.5057] | **0.6363** [0.6074, 0.6634] |

<!-- source: results/reports/statistics.md (Bootstrap 95% CI, test rows); all-neutral test accuracy = test neutral count 1,256 / 2,610 = 48.12% from results/reports/meld_metadata.md (R1 test). All-neutral test F1 values were not computed in results/. -->

**Table 8.** Test results (n = 2,610), with bootstrap 95% CIs.

| model | weighted-F1 [95% CI] | macro-F1 [95% CI] | accuracy [95% CI] |
|---|---|---|---|
| all-neutral baseline | not computed | not computed | 0.4812 |
| B2 speech-only | 0.4273 [0.4052, 0.4457] | 0.2222 [0.2063, 0.2357] | 0.4797 [0.4594, 0.4966] |
| B1 text-only | 0.6015 [0.5815, 0.6193] | 0.3948 [0.3734, 0.4151] | 0.6199 [0.6008, 0.6372] |
| **M multimodal** | **0.6173** [0.5973, 0.6367] | **0.4276** [0.4003, 0.4534] | **0.6387** [0.6195, 0.6559] |

Three readings follow.

1. **Text dominates speech on MELD.** B1 exceeds B2 by a wide margin on both splits (test weighted-F1 0.6015 against 0.4273). On test, B2's accuracy (0.4797) is slightly below that of always predicting neutral (0.4812): speech features from a frozen encoder, pooled and classified on their own, do not beat the majority class on accuracy. This is consistent with the noisy acoustic conditions of sitcom audio and with the observation in Section 4.4 that MELD transcripts already encode some prosody through punctuation.
2. **Fusion improves on text.** M improves on B1 by +0.0607 weighted-F1 on dev (macro-F1 +0.0910) and by +0.0158 on test (macro-F1 +0.0328, accuracy +0.0188). The larger macro-F1 gain indicates that the improvement is concentrated in smaller classes (Section 4.2).
3. **The held-out gain is small.** The dev gain is an upper estimate, because M's dev score is the best of eight evaluated epochs; the test figure is the one to quote.

The reports from which Tables 7 to 9 are taken are shown in Figure 6, and the weighted-F1 comparison is plotted in Figure 7. Table 9 gives the paired tests.

<!-- source: results/reports/statistics.md (McNemar, exact, two-sided) -->

**Table 9.** Exact two-sided McNemar tests on paired per-utterance correctness. n01: first model right, second wrong; n10: the reverse.

| comparison | split | n01 | n10 | p |
|---|---|---|---|---|
| M vs B1 | dev | 141 | 76 | 1.212e-05 |
| M vs B2 | dev | 287 | 87 | 4.600e-26 |
| B1 vs B2 | dev | 270 | 135 | 1.813e-11 |
| M vs B1 | test | 240 | 191 | 2.067e-02 |
| M vs B2 | test | 630 | 215 | 5.550e-48 |
| B1 vs B2 | test | 643 | 277 | 3.104e-34 |

**Overlapping intervals and a significant paired test.** On test, the weighted-F1 CIs of M [0.5973, 0.6367] and B1 [0.5815, 0.6193] overlap, while the McNemar test of M against B1 is significant (p = 0.021). These two facts are not in conflict, and the paired test is the correct instrument here. Each bootstrap interval is *marginal*: it describes the sampling uncertainty of one model's score as if the other model did not exist. M and B1 are evaluated on the same 2,610 utterances and share the same text encoder, so their errors are strongly correlated; most of the sampling variability in their scores is shared and cancels in a comparison. Overlap of marginal intervals is therefore a conservative criterion that can miss real differences between correlated systems. McNemar conditions on the 431 utterances where the two models disagree (240 + 191) and asks whether the split between "M right, B1 wrong" and the reverse is consistent with 50:50. It is not: M wins 49 more of these disagreements than it loses. The answer to RQ1 is therefore yes, with a qualification: adding speech produces a statistically significant improvement on held-out data, and the effect is small (+0.0158 weighted-F1). McNemar tests correctness (accuracy), not weighted-F1, and a paired bootstrap of the weighted-F1 difference was not computed.

> **[SCREENSHOT 6]**
> **What to capture:** The rendered statistics report showing both the bootstrap CI table and the McNemar table.
> **Where to find it:** `results/reports/statistics.md` on GitHub (https://github.com/maulikrathor/emorec-xmm/blob/main/results/reports/statistics.md), light mode, greyscale.
> **Caption to use:** Figure 6. Main results with bootstrap 95% confidence intervals and exact McNemar tests on paired predictions (seed 42, full training set).

> **[SCREENSHOT 7]**
> **What to capture:** The weighted-F1 comparison figure for B1, B2 and M on dev and test with CI error bars and the all-neutral dev line.
> **Where to find it:** `results/reports/figures/final_dev_vs_test_wf1.png`. The file uses colour: convert it to greyscale (for example in any image editor, or with Python `PIL.Image.open(path).convert("L")`) and check that the dev and test bars remain distinguishable before inserting.
> **Caption to use:** Figure 7. Weighted-F1 of the text-only, speech-only and multimodal models on dev and test, with bootstrap 95% CIs.

### 4.2 Per-class analysis

Table 10 gives per-class F1 for each model and split, taken from the per-run classification reports (two decimals, as written by those reports).

<!-- source: results/runs/{text_only,audio_only,xattn}-seed42-*/classification_report.txt (dev) and classification_report_test.txt (test) -->

**Table 10.** Per-class F1 of the best checkpoint of each model (support in parentheses).

| class | dev B1 | dev B2 | dev M | test B1 | test B2 | test M |
|---|---|---|---|---|---|---|
| neutral (469 / 1,256) | 0.73 | 0.64 | **0.78** | 0.76 | 0.66 | **0.78** |
| joy (163 / 402) | 0.53 | 0.18 | **0.60** | 0.58 | 0.21 | **0.60** |
| sadness (111 / 208) | **0.38** | 0.18 | 0.31 | **0.31** | 0.14 | 0.22 |
| anger (153 / 345) | 0.38 | 0.38 | **0.47** | 0.49 | 0.34 | **0.50** |
| surprise (150 / 281) | 0.57 | 0.26 | **0.66** | 0.53 | 0.21 | **0.55** |
| fear (40 / 50) | 0.04 | 0.00 | **0.25** | 0.10 | 0.00 | **0.21** |
| disgust (22 / 68) | 0.00 | 0.00 | **0.21** | 0.00 | 0.00 | **0.14** |

**Where fusion helps.** The largest gains are on the two rarest classes. Fear rises from 0.04 to 0.25 on dev and from 0.10 to 0.21 on test; disgust, which the text-only model never predicts correctly, rises from 0.00 to 0.21 on dev and to 0.14 on test. Surprise (+0.09 dev, +0.02 test) and anger (+0.09 dev, +0.01 test) also improve, clearly on dev and marginally on test, so the test evidence for these two classes is weak. Fear, surprise and anger are high-arousal emotions, the region of affect space where vocal intensity, pitch range and tempo are most informative [12], so these gains are in the direction an acoustic account predicts. The speech-only model alone scores 0.00 on fear and disgust on both splits, however, so the gain is not "the speech channel recognises fear"; it is a gain from combining the channels, in which speech refines a decision that the text has already narrowed down.

**Where fusion regresses: sadness.** M is worse than B1 on sadness on both splits, from 0.38 to 0.31 on dev and from 0.31 to 0.22 on test. The loss is in recall (Table 11): M predicts sadness less often and more precisely, and most of the sadness utterances it gives up are labelled neutral instead. Sadness is a low-arousal emotion; its typical vocal profile (low intensity, narrow pitch range, slow tempo) overlaps with flat, neutral delivery, and loudness normalisation removes absolute level as a cue. One plausible account is therefore that the speech channel pushes quiet, flat sadness utterances toward neutral. This explanation was not tested, and it is offered as a hypothesis only. Figure 8 plots the per-class F1 values of Table 10, and Figure 9 shows the full test confusion matrix of M, in which the sadness row and the neutral column make the regression visible.

<!-- source: results/runs/text_only-seed42-20261006T064248Z/classification_report*.txt and results/runs/xattn-seed42-20261006T070547Z/classification_report*.txt (precision and recall columns); results/runs/{text_only,xattn}-seed42-*/predictions_test.jsonl (row counts for true label sadness) -->

**Table 11.** The sadness regression: precision, recall and where test sadness utterances go.

| quantity | B1 text-only | M multimodal |
|---|---|---|
| dev precision / recall | 0.43 / 0.34 | 0.53 / 0.22 |
| test precision / recall | 0.37 / 0.27 | 0.43 / 0.14 |
| test sadness (n = 208) predicted as sadness | 56 | 30 |
| test sadness predicted as neutral | 97 | 114 |
| test sadness predicted as anger | 21 | 26 |
| test sadness predicted as fear | 2 | 12 |

> **[SCREENSHOT 8]**
> **What to capture:** The per-class F1 figure comparing B1, B2 and M on dev and test.
> **Where to find it:** `results/reports/figures/final_per_class_f1.png`. The file uses colour: convert to greyscale before inserting and confirm that the three model series remain distinguishable; if they do not, add the model names next to the bars by hand.
> **Caption to use:** Figure 8. Per-class F1 of the text-only, speech-only and multimodal models on dev and test; fusion gains are concentrated in fear and disgust, and sadness regresses.

> **[SCREENSHOT 9]**
> **What to capture:** The test-set confusion matrix of the multimodal model, with counts visible in every cell.
> **Where to find it:** `results/runs/xattn-seed42-20261006T070547Z/confusion_matrix_test.png`. It uses a blue colour map: convert to greyscale before inserting (the cell counts carry the information).
> **Caption to use:** Figure 9. Test confusion matrix of the multimodal model (n = 2,610); rows are true classes, columns predicted classes.

### 4.3 Dialogue context ablation

To measure what the k = 2 context contributes, M and B1 were retrained with k = 0 (current utterance only, still wrapped in its markers), with everything else unchanged. Table 12 gives the best dev weighted-F1 of each run.

<!-- source: results/context_ablation/README.md -->

**Table 12.** Dialogue context ablation on dev (one seed, no significance test).

| model | k = 2 best dev weighted-F1 | k = 0 best dev weighted-F1 | difference (k=2 minus k=0) |
|---|---|---|---|
| M multimodal | 0.6155 (epoch 5) | 0.6083 (epoch 5) | +0.0072 |
| B1 text-only | 0.5548 (epoch 10) | 0.5928 (epoch 7) | -0.0380 |

**Reading.** For the multimodal model, k = 2 context contributes +0.0072 weighted-F1. This is not a meaningful improvement: it is one seed, no significance test was run, and the difference is well inside the width of the dev confidence intervals in Table 7. This report does not claim that dialogue context helps. For the frozen text-only baseline, removing the context *improved* weighted-F1 by 0.0380. This is reported as a negative result.

**Explanation.** The likely cause is the token budget. The text input is limited to 160 tokens, and MELD utterances average about 8 whitespace tokens (7.96 in train). In a k = 2 context string, the two previous turns, their speaker tags and the separators occupy most of the sequence, and the current utterance is a small fraction of a mostly-context input. RoBERTa is frozen and only a pooling head is trained, so the model cannot learn to down-weight the context span; the `<cur>` and `</cur>` markers delimit the target, but the frozen encoder was never trained to use them. The multimodal model is less affected because its cross-attention provides a second route to the target utterance, through the speech of the current turn, which contains no context at all. In speech and language terms, concatenating context into a frozen encoder dilutes the target utterance rather than informing it.

<!-- source: results/context_ablation/README.md (Reading; Likely cause); results/reports/meld_metadata.md (R2 train tokens mean 7.96) -->

**Consequence for the headline comparison.** The strongest text-only baseline on dev is the k = 0 variant at 0.5928, not the k = 2 baseline at 0.5548. Against it, the dev gap of M narrows from +0.0607 to +0.0227. The held-out test comparison in Section 4.1 is unaffected: both test-evaluated models (M and B1) were trained with k = 2, and the k = 0 variants were deliberately not evaluated on test, because the test split is touched once, for the pre-registered final runs only. The fair summary is that M beats the B1 that was pre-registered and tested, and that a stronger text baseline exists on dev whose test performance is unknown. The committed ablation record is shown in Figure 10.

> **[SCREENSHOT 10]**
> **What to capture:** The rendered context-ablation README showing the k = 2 versus k = 0 table and the "Consequence for the headline comparison" paragraph.
> **Where to find it:** `results/context_ablation/README.md` on GitHub (https://github.com/maulikrathor/emorec-xmm/blob/main/results/context_ablation/README.md), light mode, greyscale.
> **Caption to use:** Figure 10. Dialogue-context ablation: k = 2 context adds +0.0072 dev weighted-F1 to the multimodal model and costs the frozen text-only baseline 0.0380.

### 4.4 Explainability results

#### (a) The gate is text-saturated

Across all 1,108 dev utterances the learned gate assigns almost all weight to text (Table 13). The mean g_audio is 0.0013 and the median 0.0003; 99.91 percent of utterances have g_audio below 0.05, and no utterance is audio-dominant (g_audio above 0.5). The largest g_audio on dev is 0.0692. The gate is slightly higher for fear (0.0045), anger (0.0030), disgust (0.0029) and sadness (0.0024) than for neutral and surprise (0.0005 each), and slightly higher on incorrect predictions (0.0016) than on correct ones (0.0011), but every value is two to three orders of magnitude below an even split. Figure 11 shows the full distribution on a logarithmic axis.

<!-- source: results/reports/gate_analysis.md -->

**Table 13.** Audio gate g_audio on dev (n = 1,108), reported exactly as produced.

| statistic | value |
|---|---|
| mean | 0.0013 |
| median | 0.0003 |
| standard deviation | 0.0039 |
| 5th / 95th percentile | 0.0001 / 0.0047 |
| fraction with g_audio < 0.05 (saturated toward text) | 0.9991 |
| fraction with g_audio > 0.5 (audio-dominant) | 0.0000 |
| mean by true class: neutral / joy / sadness / anger | 0.0005 / 0.0008 / 0.0024 / 0.0030 |
| mean by true class: surprise / fear / disgust | 0.0005 / 0.0045 / 0.0029 |
| mean on correct (n = 705) / incorrect (n = 403) predictions | 0.0011 / 0.0016 |

Read on its own, the gate says that the model ignores speech. That reading conflicts with Section 4.1, where adding speech significantly improves on text.

> **[SCREENSHOT 11]**
> **What to capture:** The gate distribution figure: the histogram of log10(g_audio) on dev and the bar chart of mean g_audio per true class.
> **Where to find it:** `results/reports/figures/final_gate_distribution.png`. The file uses colour: convert to greyscale before inserting.
> **Caption to use:** Figure 11. Distribution of the audio gate on dev (log scale) and its mean per true class; the gate is saturated toward text for every class.

#### (b) Modality ablation contradicts the gate

Table 14 shows what happens when each input channel is removed or corrupted at inference, with the trained weights unchanged.

<!-- source: results/reports/ablation.md (Results; Paired McNemar). n01 = control right and ablated wrong. -->

**Table 14.** Inference-time modality ablation on dev (n = 1,108), same weights in every row.

| condition | weighted-F1 [95% CI] | change in weighted-F1 | macro-F1 | McNemar vs control (n01 / n10) | p |
|---|---|---|---|---|---|
| none (control) | 0.6155 [0.5832, 0.6448] | 0 | 0.4670 | n/a | n/a |
| zero_audio | 0.5782 [0.5480, 0.6069] | -0.0373 | 0.4325 | 93 / 36 | 5.43e-07 |
| shuffle_audio | 0.5860 [0.5551, 0.6177] | -0.0296 | 0.4324 | 71 / 44 | 0.015 |
| zero_text | 0.2637 [0.2348, 0.2957] | -0.3519 | 0.0991 | 319 / 68 | 5.7e-40 |

Removing the speech costs 0.0373 weighted-F1 (p = 5.43e-07), even though the gate weights the speech summary at 0.0013. Giving each utterance someone else's speech costs 0.0296 (p = 0.015). Removing the text is catastrophic (-0.3519, p = 5.7e-40), confirming that text is the dominant channel (Figure 12). Per class, removing speech reduces anger F1 from 0.4739 to 0.3951 and surprise from 0.6598 to 0.5869, and shuffling speech reduces fear from 0.2466 to 0.0923; these are high-arousal classes, consistent with Section 4.2, although the supports are small and these per-class differences carry no confidence intervals.

<!-- source: results/reports/ablation.md (Per-class F1) -->

> **[SCREENSHOT 12]**
> **What to capture:** The modality ablation bar chart with the control bar, CI error bars and the annotated McNemar p values.
> **Where to find it:** `results/reports/figures/final_ablation.png`. The control bar is red and the ablation bars light blue: convert to greyscale before inserting; the control remains identifiable by its heavy black outline.
> **Caption to use:** Figure 12. Inference-time modality ablation on dev: removing or shuffling speech significantly lowers weighted-F1 despite a mean audio gate of 0.0013.

#### (c) Synthesis: why the gate and the ablation disagree

The gate is applied *downstream* of the cross-modal blocks (Figure 5). By the time the pooled representations are gated, the text stream has already attended to the speech frames twice, so speech-derived information is already mixed into the text summary z_t. The gate weights the pooled *speech summary* z_a, not *speech information*. A near-zero g_audio is therefore fully compatible with a model that relies on speech which has entered through the text stream. The modality ablation detects this because it removes the speech before the cross-attention, rather than reading a weight after it.

The shuffle control rules out a weaker explanation. If speech acted only as a presence or bias signal ("an audio stream exists"), replacing it with another utterance's speech, which has the same shape and statistics, would leave accuracy unchanged. It does not (p = 0.015), which is consistent with the model using utterance-specific acoustic content.

This is a measured case of an intrinsic interpretability signal failing to reflect information flow. What is *measured* is the disagreement: g_audio is about 0.001, yet zeroing or shuffling speech significantly lowers accuracy. The *mechanism* (speech entering the text stream through cross-attention) follows from the architecture and is the explanation consistent with the data, but it was not isolated by a dedicated experiment; a small but non-zero gated speech summary amplified by the MLP is an alternative route these experiments do not exclude. The practical consequence for RQ2 is that a gate value of this kind must not be reported as a measure of modality importance without an ablation or attribution check beside it.

#### (d) Integrated Gradients is faithful and passes the sanity check

<!-- source: results/reports/faithfulness.md (AUC table). Comprehensiveness: higher is more faithful; sufficiency: lower is more faithful. -->

**Table 15.** Faithfulness of IG attributions on 40 class-stratified dev examples (AUC of the deletion curve; bootstrap 95% CIs; the last column is a paired per-example CI).

| modality | curve | IG AUC [CI] | random AUC [CI] | IG minus random [paired CI] | IG better than random |
|---|---|---|---|---|---|
| text | comprehensiveness | 0.1490 [0.1184, 0.1779] | 0.0825 [0.0651, 0.0987] | +0.0665 [+0.0451, +0.0889] | yes |
| text | sufficiency | 0.0809 [0.0496, 0.1100] | 0.1394 [0.1104, 0.1678] | -0.0585 [-0.0815, -0.0359] | yes |
| speech | comprehensiveness | 0.0556 [0.0430, 0.0680] | 0.0023 [-0.0031, 0.0073] | +0.0534 [+0.0409, +0.0640] | yes |
| speech | sufficiency | -0.0383 [-0.0604, -0.0189] | 0.0089 [-0.0067, 0.0229] | -0.0472 [-0.0584, -0.0364] | yes |

IG beats the random control on all four curves (Table 15, Figure 13), with paired CIs that exclude zero in the faithful direction. Deleting the speech frames that IG ranks highest lowers the predicted-class probability far more than deleting random frames (comprehensiveness 0.0556 against 0.0023), so IG locates the acoustic regions the model uses. The negative speech sufficiency AUC means that keeping only the top-ranked frames *raises* the predicted-class probability above that of the full input on average: the remaining frames were, on balance, evidence against the prediction.

<!-- source: results/reports/faithfulness.md (Parameter-randomisation sanity check) -->

**Table 16.** Parameter-randomisation sanity check (classifier head re-initialised; 20 examples).

| modality | mean Spearman (signed) | mean Spearman (magnitudes) | verdict |
|---|---|---|---|
| text | -0.1432 | 0.2944 | attributions degrade |
| speech | -0.0798 | 0.3272 | attributions degrade |

Table 16 shows that after the classifier head is randomised, the attributions are essentially uncorrelated with the originals for both modalities, far below the 0.5 failure threshold. IG therefore depends on what the model learned and is not acting as a model-independent edge detector over the input.

**A qualitative example, and a note on punctuation as written prosody.** For `dev/dia56_utt9` (Joey: "Listen, I ah, I gotta say good bye, and-and I gotta say it quick cause this is killing me.", true and predicted sadness, probability 0.645), which has the largest audio gate on dev (0.0692), the five highest-scoring tokens all lie in the current utterance (Figure 14): `killing` (+0.3611), `is` (+0.2556), `quick` (-0.2347), `say` (+0.2329) and `this` (+0.2082). Several other examples assign large scores to punctuation: `!` scores +2.0637 for Ross's "Okay!" (predicted joy), and the two `?!` tokens score +0.6577 and +0.4548 for Rachel's "What?! Joey got a turkey stuck on his head?!" (predicted surprise). MELD's transcripts are scripted and punctuated, and punctuation is a partial written encoding of prosody (exclamatory intensity, questioning intonation). This is one reason the text channel is strong on this corpus, and it is a reason to expect the text channel to weaken under automatic speech recognition, whose output usually carries less of this punctuation (Section 6). Individual token scores should be read with caution, as the next paragraph explains.

<!-- source: results/reports/xai_examples.md (examples 1, 13 and 17) -->

**Convergence caveat.** The mean absolute IG convergence delta over the 80 faithfulness passes (40 text, 40 speech) is 0.284 at 64 steps. In the 20 example reports the large deltas come from the text pass (up to 1.2 in magnitude), while speech deltas are at most a few millionths. Completeness is therefore only approximately satisfied for text, and individual token scores should not be over-interpreted; the aggregate faithfulness result in Table 15 is the claim that is supported.

<!-- source: results/reports/faithfulness.md (IG convergence line); results/reports/xai_examples.md (per-example convergence deltas) -->

> **[SCREENSHOT 13]**
> **What to capture:** The two-panel faithfulness figure: comprehensiveness and sufficiency curves for IG (solid) and the random control (dashed), for text and speech.
> **Where to find it:** `results/reports/figures/faithfulness_curves.png`. Text curves are blue and speech curves orange: convert to greyscale before inserting and add the labels "text" and "speech" beside the corresponding curves if they become hard to tell apart (solid and dashed line styles already separate IG from the random control).
> **Caption to use:** Figure 13. Deletion-based faithfulness of Integrated Gradients against a random-attribution control on 40 dev examples.

> **[SCREENSHOT 14]**
> **What to capture:** The token-level IG attribution figure for `dev/dia56_utt9`, showing signed scores over the context and current-utterance tokens.
> **Where to find it:** `results/reports/figures/dev_dia56_utt9_tokens.png` (its companion speech figure is `dev_dia56_utt9_audio.png` in the same folder); convert to greyscale before inserting.
> **Caption to use:** Figure 14. Integrated Gradients token attributions for a correctly classified sadness utterance; the highest-scoring tokens lie inside the current-utterance span.

### 4.5 Reproducibility

- **Tests.** The repository's test suite reported 183 passing tests at the close of the final implementation ticket (Figure 15). The suite covers, among other things, the five golden context strings, end-keeping truncation, the frozen-encoder invariant, padding invariance of the logits, the gate simplex, and the explainability modules.
- **Deterministic seeding.** All randomness goes through a single seeding entry point. When the k = 2 multimodal model was retrained four days later from a later commit, for the context ablation, it reproduced the original run exactly: best dev weighted-F1 0.6155 at epoch 5, the same epoch-5 training loss to every printed digit, and identical predictions on all 1,108 dev utterances.
- **Traceability.** Every reported number traces to a run directory under `results/runs/` or `results/context_ablation/` (configuration, per-epoch metrics, run metadata with git SHA, per-utterance predictions, classification reports) or to a report under `results/reports/`.

<!-- source: docs/CLAUDE_CODE_HANDOFF.md section 0 (pytest -q: 183 passed); results/runs/xattn-seed42-20261006T070547Z/metrics.json and results/context_ablation/xattn-seed42-20261010T061528Z/metrics.json (epoch 5: dev_weighted_f1 0.6155359676338004, train_loss 1.0451721679076278 in both); predictions_dev.jsonl of the same two runs (identical, 1,108 lines); run_meta.json git_sha efb2451 and c217ad9 respectively -->

> **[SCREENSHOT 15]**
> **What to capture:** The final summary line of the test run, reading "183 passed", together with the command that produced it.
> **Where to find it:** From the repository root, in an environment where the package is installed (`pip install -e ".[dev,xai]"`), run `pytest -q`; capture the last lines of the terminal output in a light theme and convert to greyscale.
> **Caption to use:** Figure 15. The project test suite: 183 tests passing.

### 4.6 Worked examples from the demo predictor

The script `scripts/demo_predict.py` loads a checkpoint, builds the model input for one MELD utterance through the same dataset code used in training, and prints the context string, the true and predicted emotion, the class probabilities and the gate. Table 17 reports three outputs obtained from the checkpoint `artifacts/runs/xattn-seed42-20261010T062959Z` on dev utterances; the corresponding terminal outputs are Figures 16 to 18.

<!-- source: demo_predict.py console outputs supplied by the project author for checkpoint artifacts/runs/xattn-seed42-20261010T062959Z; these outputs are not stored under results/ (see the reconciliation note and Table 18) -->

**Table 17.** Three demo-predictor outputs (dev utterances).

| case | utterance | speaker and text | duration | true | predicted | p(predicted) | g_text / g_audio |
|---|---|---|---|---|---|---|---|
| CORRECT | `dev/dia104_utt7` | Chandler: "You are so great! I love you!" | 2.69 s | joy | joy | 0.82 | 0.999 / 0.001 |
| CORRECT | `dev/dia11_utt5` | Rachel: "Okay, okay, okay, look, just don't freak out, but I kinda lost it." | 3.01 s | fear | fear | 0.75 | 0.982 / 0.018 |
| FAILURE | `dev/dia15_utt6` | Rachel: "Its moving." | 0.15 s | fear | surprise | 0.84 | 0.999 / 0.001 |

**Correct case 1.** Chandler's "You are so great! I love you!" is lexically unambiguous, and the model predicts joy with high confidence.

**Correct case 2.** Rachel's "Okay, okay, okay, look, just don't freak out, but I kinda lost it." is a harder case: fear here is anticipatory (she expects a reaction), the words contain no fear term, and the repetition "okay, okay, okay" is a disfluency pattern of anxious speech. The model predicts fear correctly, a class for which the text-only baseline has an F1 of only 0.04 on dev (Table 10).

**Failure case.** Rachel's "Its moving." is labelled fear and predicted as surprise. Two properties make it hard. First, the clip is only 0.15 seconds long. Two words are unlikely to fit in 0.15 s at a normal speaking rate, so the segment boundary probably does not cover the whole utterance, and there is almost no prosodic evidence available; a clip this short is also below the 0.4 s threshold for loudness normalisation (Section 3.2). Second, the text is two words with no emotional vocabulary. Fear and surprise are both high-arousal emotions that share vocal features such as raised pitch and intensity, and they are often confused with each other; in M's committed test predictions, 6 of the 50 fear utterances are predicted as surprise, although neutral (18 of 50) is the most frequent confusion for fear.

<!-- source: results/runs/xattn-seed42-20261006T070547Z/predictions_test.jsonl (true fear row: neutral 18, surprise 6, fear 11; consistent with confusion_matrix_test.png) -->

In all three cases the gate reports near-total reliance on text (g_audio between 0.001 and 0.018), which the modality ablation in Section 4.4 shows to be an unreliable indicator of how much the model actually uses speech.

**Reconciliation note.** The committed prediction files do not reproduce the values in Table 17 (Table 18). Checkpoint `xattn-seed42-20261010T062959Z` is the **k = 0** model of the context ablation (Section 4.3); its committed dev predictions, and those of the headline k = 2 model, give the probabilities below, and both assign `dev/dia15_utt6` to **neutral**, not surprise. Because `scripts/demo_predict.py` builds the context string from the configuration file passed to it, and `configs/xattn.yaml` inherits k = 2 from `configs/base.yaml`, the most likely explanation is that the outputs in Table 17 were produced by running the k = 0 checkpoint with a k = 2 configuration, an input format it was not trained on. This could not be verified, because checkpoints are not stored in the repository. Screenshots 16 to 18 should therefore be captured with the checkpoint's own configuration file (command below), and Table 17 updated from those captures if they differ.

<!-- source: results/context_ablation/xattn-seed42-20261010T062959Z/config.yaml (context_k 0) and predictions_dev.jsonl; results/runs/xattn-seed42-20261006T070547Z/predictions_dev.jsonl; results/reports/gates.csv (g_audio of the headline k = 2 model). Label order: neutral, joy, sadness, anger, surprise, fear, disgust. -->

**Table 18.** Committed dev predictions for the same three utterances.

| utterance | true | headline M, k = 2 (`070547Z`): predicted (p) | headline M g_audio | k = 0 M (`062959Z`): predicted (p) |
|---|---|---|---|---|
| `dev/dia104_utt7` | joy | joy (0.9757) | 0.0017 | joy (0.9574) |
| `dev/dia11_utt5` | fear | fear (0.6646) | 0.0129 | fear (0.9248) |
| `dev/dia15_utt6` | fear | neutral (0.9489) | 0.0001 | neutral (0.9750) |

The qualitative conclusions of this section hold for both sources: the first two utterances are classified correctly, the third is a failure on a 0.15 s clip, and the gate is near zero for speech in every case.

> **[SCREENSHOT 16]**
> **What to capture:** The complete demo-predictor output for `dev/dia104_utt7` (CORRECT case), from `UTT_ID` to the `MODALITY GATE` lines.
> **Where to find it:** On the machine holding the checkpoints, from the repository root run `python scripts/demo_predict.py --ckpt artifacts/runs/xattn-seed42-20261010T062959Z/best.pt --config results/context_ablation/xattn-seed42-20261010T062959Z/config.yaml --utt-id dev/dia104_utt7`; light terminal theme, greyscale.
> **Caption to use:** Figure 16. Demo predictor, correct case: Chandler's "You are so great! I love you!" (true joy, predicted joy).

> **[SCREENSHOT 17]**
> **What to capture:** The complete demo-predictor output for `dev/dia11_utt5` (CORRECT case), from `UTT_ID` to the `MODALITY GATE` lines.
> **Where to find it:** Same command as Screenshot 16 with `--utt-id dev/dia11_utt5`.
> **Caption to use:** Figure 17. Demo predictor, correct case: Rachel's "Okay, okay, okay, look, just don't freak out, but I kinda lost it." (true fear, predicted fear).

> **[SCREENSHOT 18]**
> **What to capture:** The complete demo-predictor output for `dev/dia15_utt6` (FAILURE case), including the `AUDIO` line that shows the clip duration and the `CORRECT?: no` line.
> **Where to find it:** Same command as Screenshot 16 with `--utt-id dev/dia15_utt6`.
> **Caption to use:** Figure 18. Demo predictor, failure case: Rachel's "Its moving.", a 0.15 s clip labelled fear and misclassified.

---

## 5. Conclusion

### 5.1 Summary

This project built a complete speech and language pipeline for seven-class emotion recognition in conversation on MELD: validated corpus ingestion with exact media reconciliation and split-scoped lookup, 16 kHz loudness-normalised speech with end-preserving truncation, a dialogue-context text format with explicit current-utterance markers, frozen RoBERTa-base and WavLM-base-plus encoders, and a gated bidirectional cross-attention fusion model with 3,688,969 trainable parameters. It evaluated the model against text-only and speech-only baselines on MELD's official splits with paired statistics, and audited its explanations quantitatively.

The evidence supports the following conclusions.

1. **RQ1.** Adding the speech channel to a text model produces a statistically significant but modest improvement on held-out data: test weighted-F1 0.6173 against 0.6015, McNemar p = 0.021, with overlapping marginal confidence intervals. The gain is concentrated in rare and high-arousal classes (fear, disgust, and on dev also surprise and anger), and the model regresses on sadness, mainly by predicting neutral instead.
2. **Context.** With frozen encoders, concatenating two turns of dialogue context does not help the multimodal model (+0.0072 dev weighted-F1, one seed) and hurts the text-only baseline (-0.0380), most plausibly because the context dilutes the target utterance within a fixed token budget.
3. **RQ2.** The model's intrinsic modality gate cannot answer RQ2. It reports near-total text reliance (mean g_audio 0.0013), while modality ablation shows significant dependence on utterance-specific speech (zeroing speech -0.0373 weighted-F1, p = 5.43e-07; shuffling -0.0296, p = 0.015). Because the gate sits downstream of cross-attention, it measures reliance on the pooled speech summary, not on speech information. Integrated Gradients, by contrast, is measurably faithful for both modalities and passes the parameter-randomisation check, so per-utterance attribution is possible with IG, but not with the gate.

In speech and language terms, the result is that prosodic information is used by the model, in the classes where acoustic theory predicts it should matter, but it is used through a path that the model's own interpretability signal does not show.

### 5.2 Limitations

1. **One seed.** Every result is a single training run per model (seed 42). The bootstrap CIs and McNemar tests quantify evaluation-set sampling, not training randomness; seed-to-seed variance is unknown.
2. **Frozen encoders cap absolute performance.** Test weighted-F1 0.6173 is below the 67.4 reported by TelME [3] with trainable encoders and the visual channel. How much of the gap freezing explains was not measured.
3. **IG convergence.** The mean absolute convergence delta is 0.284 at 64 steps, driven by the text pass, so individual token scores should not be over-interpreted; only the aggregate faithfulness result is supported.
4. **Deletion operators match the IG baselines by construction** (pad embedding for text, zero frame for speech), which mildly favours IG in the faithfulness evaluation. No second deletion operator was run.
5. **Faithfulness used 40 examples** and the randomisation check 20.
6. **Ablation inputs are out of distribution.** The model never saw zeroed speech or all-pad text during training; the size of an ablation drop measures reliance under the intervention, not the information content of a channel.
7. **The gate mechanism is not isolated.** The explanation in Section 4.4(c) follows from the architecture and fits the data, but no experiment ablated the speech-to-text cross-attention path directly.
8. **Sadness regression.** M is worse than B1 on sadness on both splits (dev 0.38 to 0.31, test 0.31 to 0.22), a recall loss whose cause was not investigated.
9. **Disgust remains weak.** M's disgust F1 is 0.21 on dev and 0.14 on test, on supports of 22 and 68, and fear is 0.21 on test; no per-class confidence intervals were computed.
10. **Dev gap and baseline strength.** The dev gain of M is an upper estimate (selection on dev), and a k = 0 text baseline is stronger on dev than the tested one (Section 4.3).
11. **Mislabelled per-class heading in a results file.** The per-class table in `results/reports/final_results.md` is headed "best checkpoint, dev" but contains the test per-class F1 values (for example M sadness 0.2166, fear 0.2056, disgust 0.1412 match `classification_report_test.txt`, and M's dev values in `results/reports/ablation.md` differ). This report takes all per-class numbers from the per-run classification reports instead.
12. **Corpus scope.** MELD is scripted, acted US sitcom English with a small recurring cast, a laugh track and punctuated gold transcripts, and episodes recur across splits. No claim is made about spontaneous speech, other languages or dialects, or other recording conditions.
13. **Gold transcripts only.** No automatic speech recognition was used; a deployed system would see recognised text without the scripted punctuation that IG shows the model uses.
14. **Demo outputs.** The demo-predictor values in Table 17 are not reproduced by the committed prediction files (Section 4.6).

---

## 6. Future Work

1. **Multiple seeds with variance.** Train each model with several seeds (the design specifies 13, 42 and 2026) and report the mean and spread of every metric, so that training randomness and evaluation sampling can be separated.
2. **Partial encoder unfreezing or LoRA.** Fine-tune the top layers of RoBERTa and WavLM, or apply low-rank adaptation [17] to their attention projections, to measure how much of the gap to trainable systems such as TelME [3] the frozen design costs, and whether a trainable speech encoder changes the per-class picture.
3. **A context-length study with trainable encoders.** Repeat the k = 0 versus k = 2 comparison, and extend it to larger k, with encoders that can learn to use the `<cur>` markers, and compare concatenated context with a dedicated context model in the style of DialogueRNN [13].
4. **Cross-method attribution agreement.** Implement occlusion (3-token text spans, 1 s speech windows) and attention-based attribution, and measure their agreement with IG on the same utterances, to test whether the explanations converge.
5. **ASR in the loop.** Replace gold transcripts with automatic speech recognition output (for example Whisper [18]) at test time, and at training time, to measure the train-serve skew and to test the prediction from Section 4.4(d) that the text channel weakens when scripted punctuation disappears, which would shift reliance toward speech.
6. **Testing the gate mechanism directly.** Ablate the speech-to-text direction of the cross-modal attention while leaving the speech-to-speech path intact, and measure the change in accuracy and in the gate. If the hypothesis in Section 4.4(c) is right, this ablation should remove most of the speech benefit even though g_audio is unchanged. A complementary design moves the gate before the cross-modal blocks, where it would control information flow rather than only the pooled summary.

---

## 7. References

[1] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria and R. Mihalcea. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2019.

[2] A. Pandey, J. Singh and M. Kaur. Bridging Text and Speech for Emotion Understanding: An Explainable Multimodal Transformer Fusion Framework with Unified Audio-Text Attribution. *Journal of Intelligence*, 13(12), 159, 2025.

[3] T. Yun, H. Lim, J. Lee and M. Song. TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2024.

[4] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu, M. Zeng, X. Yu and F. Wei. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE Journal of Selected Topics in Signal Processing*, 16(6), 2022.

[5] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer and V. Stoyanov. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692, 2019.

[6] M. Sundararajan, A. Taly and Q. Yan. Axiomatic Attribution for Deep Networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 2017.

[7] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt and B. Kim. Sanity Checks for Saliency Maps. In *Advances in Neural Information Processing Systems 31 (NeurIPS)*, 2018.

[8] J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher and B. C. Wallace. ERASER: A Benchmark to Evaluate Rationalized NLP Models. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020.

[9] N. Kokhlikyan, V. Miglani, M. Martin, E. Wang, B. Alsallakh, J. Reynolds, A. Melnikov, N. Kliushkina, C. Araya, S. Yan and O. Reblitz-Richardson. Captum: A Unified and Generic Model Interpretability Library for PyTorch. arXiv:2009.07896, 2020.

[10] Q. McNemar. Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. *Psychometrika*, 12(2), 1947.

[11] B. Efron and R. J. Tibshirani. *An Introduction to the Bootstrap*. Chapman and Hall, 1993.

[12] K. R. Scherer. Vocal Communication of Emotion: A Review of Research Paradigms. *Speech Communication*, 40(1-2), 2003.

[13] N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, A. Gelbukh and E. Cambria. DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2019.

[14] S. Jain and B. C. Wallace. Attention is not Explanation. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2019.

[15] European Broadcasting Union. EBU R 128: Loudness Normalisation and Permitted Maximum Level of Audio Signals. EBU Recommendation.

[16] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin. Attention Is All You Need. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017.

[17] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang and W. Chen. LoRA: Low-Rank Adaptation of Large Language Models. In *International Conference on Learning Representations (ICLR)*, 2022.

[18] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey and I. Sutskever. Robust Speech Recognition via Large-Scale Weak Supervision. In *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023.

[19] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency and R. Salakhutdinov. Multimodal Transformer for Unaligned Multimodal Language Sequences. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2019.

---

## Working Demo Video Link

**Demo video:** `[INSERT URL HERE]`
