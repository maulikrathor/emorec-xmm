# IG examples

- checkpoint: `artifacts/runs/xattn-seed42-20261006T070547Z/best.pt`  |  model: `xattn`  |  manifest: `artifacts/manifests/dev.jsonl`  |  seed: 42
- generated (UTC): 2026-10-06 07:59:07Z  |  n_steps: 32 (re-run at 64 if |delta|/|sum(attr)| > 0.05)
- IG target is the PREDICTED class's logit. Text baseline: pad embeddings; audio baseline: zeros, attributed at the pre-`proj_a` frame output (documented deviation from ARCHITECTURE.md's post-`proj_a` point).
- Text scores are raw signed sums over the embedding dimension; audio scores are raw sums over the hidden dimension. Positive = pushes the predicted-class logit up relative to the baseline.

## 1. `dev/dia56_utt9`  (audio-dominant)

- speaker/text: **Joey**: Listen, I ah, I gotta say good bye, and-and I gotta say it quick cause this is killing me.
- true: **sadness**  |  predicted: **sadness** (correct)
- probs: sadness 0.645, fear 0.248, neutral 0.065, joy 0.020, anger 0.016, surprise 0.004, disgust 0.003
- gate: g_text=0.9308, g_audio=0.0692
- IG target: predicted class `sadness`; convergence delta text=-1.40e-02 (n_steps=64), audio=-1.18e-06 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `killing` +0.3611 [cur], `is` +0.2556 [cur], `quick` -0.2347 [cur], `say` +0.2329 [cur], `this` +0.2082 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [1.5-2.0 s] -0.00828, [3.5-4.0 s] -0.00797, [0.0-0.5 s] +0.00618
- figures: [tokens](figures/dev_dia56_utt9_tokens.png), [audio](figures/dev_dia56_utt9_audio.png)

## 2. `dev/dia104_utt9`  (audio-dominant)

- speaker/text: **Chandler**: Nothing! I said, I said "You're so great" and then I just, I just stopped talking!
- true: **fear**  |  predicted: **fear** (correct)
- probs: fear 0.256, joy 0.236, anger 0.226, sadness 0.176, neutral 0.054, surprise 0.044, disgust 0.009
- gate: g_text=0.9537, g_audio=0.0463
- IG target: predicted class `fear`; convergence delta text=3.81e-01 (n_steps=64), audio=5.07e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `stopped` +0.2380 [cur], `great` +0.2293 [cur], `I` +0.1280 [cur], `said` +0.1221 [cur], `great` +0.1080 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [1.5-2.0 s] -0.01259, [0.0-0.5 s] -0.01060, [2.5-3.0 s] +0.00902
- figures: [tokens](figures/dev_dia104_utt9_tokens.png), [audio](figures/dev_dia104_utt9_audio.png)

## 3. `dev/dia10_utt8`  (text-dominant)

- speaker/text: **Chandler**: How do you figure that?
- true: **neutral**  |  predicted: **neutral** (correct)
- probs: neutral 0.615, surprise 0.300, joy 0.064, anger 0.009, sadness 0.006, fear 0.004, disgust 0.001
- gate: g_text=1.0000, g_audio=0.0000
- IG target: predicted class `neutral`; convergence delta text=2.83e-01 (n_steps=64), audio=0.00e+00 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `figure` +0.6147 [cur], `do` +0.5397 [cur], `that` -0.3507 [cur], `you` +0.1923 [cur], `two` +0.1758 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [0.5-1.0 s] +0.03394, [0.0-0.5 s] +0.01078, [1.0-1.5 s] +0.00749
- figures: [tokens](figures/dev_dia10_utt8_tokens.png), [audio](figures/dev_dia10_utt8_audio.png)

## 4. `dev/dia110_utt2`  (text-dominant)

- speaker/text: **Phoebe**: Well, what period is it from?
- true: **neutral**  |  predicted: **neutral** (correct)
- probs: neutral 0.899, surprise 0.050, joy 0.031, anger 0.010, sadness 0.005, fear 0.003, disgust 0.001
- gate: g_text=1.0000, g_audio=0.0000
- IG target: predicted class `neutral`; convergence delta text=-6.73e-01 (n_steps=64), audio=1.43e-06 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `from` +0.4329 [cur], `period` +0.3096 [cur], `is` +0.2550 [cur], `oe` -0.1706 [cur], `` +0.1500 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [0.0-0.5 s] +0.01410, [1.5-2.0 s] +0.01239, [0.5-1.0 s] +0.00692
- figures: [tokens](figures/dev_dia110_utt2_tokens.png), [audio](figures/dev_dia110_utt2_audio.png)

## 5. `dev/dia7_utt8`  (misclassified)

- speaker/text: **Ross**: Shell drive us totally crazy.
- true: **disgust**  |  predicted: **fear** (incorrect)
- probs: fear 0.445, anger 0.338, sadness 0.129, neutral 0.048, joy 0.024, surprise 0.009, disgust 0.006
- gate: g_text=0.9951, g_audio=0.0049
- IG target: predicted class `fear`; convergence delta text=9.71e-02 (n_steps=64), audio=-1.37e-06 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `crazy` +0.4247 [cur], `us` +0.3321 [cur], `she` +0.2247 [ctx], `.` +0.2190 [ctx], `drive` +0.1874 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [1.0-1.5 s] -0.01220, [1.5-2.0 s] -0.00727, [0.0-0.5 s] +0.00538
- figures: [tokens](figures/dev_dia7_utt8_tokens.png), [audio](figures/dev_dia7_utt8_audio.png)

## 6. `dev/dia13_utt13`  (misclassified)

- speaker/text: **Frank**: They musta read the sonogram wrong.
- true: **surprise**  |  predicted: **neutral** (incorrect)
- probs: neutral 0.700, fear 0.128, sadness 0.067, joy 0.049, anger 0.026, surprise 0.025, disgust 0.005
- gate: g_text=0.9994, g_audio=0.0006
- IG target: predicted class `neutral`; convergence delta text=2.84e-01 (n_steps=64), audio=2.56e-06 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `wrong` -0.7525 [cur], `.` +0.5407 [cur], `` +0.2636 [cur], `son` -0.2612 [cur], `kindergarten` +0.2525 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [3.0-3.5 s] -0.00838, [2.0-2.5 s] +0.00728, [3.5-4.0 s] -0.00727
- figures: [tokens](figures/dev_dia13_utt13_tokens.png), [audio](figures/dev_dia13_utt13_audio.png)

## 7. `dev/dia63_utt4`  (stratified:joy)

- speaker/text: **Ross**: Yeah! And! And, it was the easiest 400 bucks Ive ever made.
- true: **joy**  |  predicted: **joy** (correct)
- probs: joy 0.957, neutral 0.025, surprise 0.009, anger 0.006, sadness 0.001, fear 0.001, disgust 0.000
- gate: g_text=0.9997, g_audio=0.0003
- IG target: predicted class `joy`; convergence delta text=6.49e-01 (n_steps=64), audio=-5.36e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `it` +0.4369 [cur], `Ross` +0.3627 [cur], `[` +0.3607 [ctx], `Yeah` +0.3129 [cur], `fun` +0.2919 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [3.0-3.5 s] -0.01742, [1.0-1.5 s] +0.01041, [2.5-3.0 s] +0.00866
- figures: [tokens](figures/dev_dia63_utt4_tokens.png), [audio](figures/dev_dia63_utt4_audio.png)

## 8. `dev/dia95_utt6`  (stratified:anger)

- speaker/text: **Monica**: I know! Although, you did tell an awful lot of jokes.
- true: **anger**  |  predicted: **joy** (incorrect)
- probs: joy 0.330, anger 0.179, neutral 0.160, sadness 0.135, surprise 0.082, disgust 0.062, fear 0.051
- gate: g_text=0.9996, g_audio=0.0004
- IG target: predicted class `joy`; convergence delta text=-6.06e-02 (n_steps=64), audio=0.00e+00 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `you` +0.2636 [cur], `of` +0.2049 [cur], `.` -0.1913 [cur], `jokes` +0.1909 [cur], `lot` +0.1486 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [0.0-0.5 s] +0.01696, [3.5-4.0 s] -0.01605, [0.5-1.0 s] +0.00827
- figures: [tokens](figures/dev_dia95_utt6_tokens.png), [audio](figures/dev_dia95_utt6_audio.png)

## 9. `dev/dia106_utt1`  (stratified:neutral)

- speaker/text: **Chandler**: Oh yeah, Dana Keystone. She was in my Movement class.
- true: **neutral**  |  predicted: **neutral** (correct)
- probs: neutral 0.736, joy 0.055, disgust 0.051, sadness 0.051, anger 0.048, surprise 0.041, fear 0.018
- gate: g_text=0.9993, g_audio=0.0007
- IG target: predicted class `neutral`; convergence delta text=-1.20e+00 (n_steps=64), audio=-4.77e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `.` -0.6441 [cur], `girl` -0.4828 [ctx], `[` +0.4173 [cur], `She` -0.4098 [cur], `class` +0.3075 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [1.5-2.0 s] -0.02101, [3.0-3.5 s] -0.00936, [0.5-1.0 s] +0.00577
- figures: [tokens](figures/dev_dia106_utt1_tokens.png), [audio](figures/dev_dia106_utt1_audio.png)

## 10. `dev/dia60_utt3`  (stratified:joy)

- speaker/text: **Monica**: I will! But not tonight. For dinner music, I thought we could listen to that tape you made me.
- true: **joy**  |  predicted: **joy** (correct)
- probs: joy 0.640, anger 0.154, neutral 0.144, sadness 0.022, fear 0.021, surprise 0.016, disgust 0.004
- gate: g_text=0.9991, g_audio=0.0009
- IG target: predicted class `joy`; convergence delta text=2.45e-01 (n_steps=64), audio=-2.38e-06 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `could` +0.4092 [cur], `tonight` +0.3464 [cur], `music` +0.2483 [cur], `will` +0.2243 [cur], `made` +0.2106 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [2.5-3.0 s] +0.01663, [3.5-4.0 s] +0.00671, [4.5-5.0 s] +0.00582
- figures: [tokens](figures/dev_dia60_utt3_tokens.png), [audio](figures/dev_dia60_utt3_audio.png)

## 11. `dev/dia26_utt16`  (stratified:sadness)

- speaker/text: **Rachel**: Ok, Monica. I just have to know one thing. Did you go with her to
- true: **sadness**  |  predicted: **neutral** (incorrect)
- probs: neutral 0.804, joy 0.053, fear 0.039, anger 0.037, sadness 0.031, disgust 0.020, surprise 0.017
- gate: g_text=0.9996, g_audio=0.0004
- IG target: predicted class `neutral`; convergence delta text=-1.17e+00 (n_steps=64), audio=8.94e-08 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `thing` +0.4576 [cur], `[` -0.3004 [ctx], `` -0.2363 [ctx], `just` -0.2120 [cur], `Ok` +0.2048 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [2.0-2.5 s] -0.00877, [3.5-4.0 s] -0.00319, [1.5-2.0 s] +0.00271
- figures: [tokens](figures/dev_dia26_utt16_tokens.png), [audio](figures/dev_dia26_utt16_audio.png)

## 12. `dev/dia108_utt3`  (stratified:anger)

- speaker/text: **Phoebe**: Ow!
- true: **anger**  |  predicted: **anger** (correct)
- probs: anger 0.271, surprise 0.267, fear 0.158, disgust 0.125, sadness 0.099, neutral 0.045, joy 0.035
- gate: g_text=0.9990, g_audio=0.0010
- IG target: predicted class `anger`; convergence delta text=-5.33e-01 (n_steps=64), audio=-4.77e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `O` -0.2288 [ctx], `it` +0.2083 [ctx], `Ph` +0.1645 [ctx], `hurt` +0.1475 [ctx], `oe` +0.1408 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [0.0-0.5 s] -0.03875, [0.5-1.0 s] -0.01195
- figures: [tokens](figures/dev_dia108_utt3_tokens.png), [audio](figures/dev_dia108_utt3_audio.png)

## 13. `dev/dia28_utt8`  (stratified:surprise)

- speaker/text: **Rachel**: What?! Joey got a turkey stuck on his head?!
- true: **surprise**  |  predicted: **surprise** (correct)
- probs: surprise 0.450, anger 0.292, disgust 0.092, joy 0.071, sadness 0.046, neutral 0.025, fear 0.023
- gate: g_text=0.9996, g_audio=0.0004
- IG target: predicted class `surprise`; convergence delta text=-1.67e-01 (n_steps=64), audio=-4.32e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `?!` +0.6577 [cur], `?!` +0.4548 [cur], `head` -0.4504 [cur], `stuck` -0.3871 [cur], `]:` +0.3526 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [0.0-0.5 s] +0.00605, [1.5-2.0 s] -0.00600, [2.0-2.5 s] +0.00445
- figures: [tokens](figures/dev_dia28_utt8_tokens.png), [audio](figures/dev_dia28_utt8_audio.png)

## 14. `dev/dia46_utt8`  (stratified:fear)

- speaker/text: **Joey**: Itll kill him. I mean itll, itll just kill him.
- true: **fear**  |  predicted: **neutral** (incorrect)
- probs: neutral 0.336, fear 0.327, sadness 0.222, anger 0.087, joy 0.019, disgust 0.005, surprise 0.003
- gate: g_text=0.9877, g_audio=0.0123
- IG target: predicted class `neutral`; convergence delta text=2.06e-01 (n_steps=64), audio=7.15e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `mean` +0.5166 [cur], `.` +0.2551 [cur], `It` +0.2130 [cur], `ll` -0.1739 [cur], `if` +0.1629 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [1.5-2.0 s] +0.00826, [1.0-1.5 s] -0.00662, [0.0-0.5 s] +0.00552
- figures: [tokens](figures/dev_dia46_utt8_tokens.png), [audio](figures/dev_dia46_utt8_audio.png)

## 15. `dev/dia58_utt1`  (stratified:disgust)

- speaker/text: **Ross**: Yknow what, I dont know how comfortable I am going to see how hot the sex is between some guy and your girlfriend.
- true: **disgust**  |  predicted: **joy** (incorrect)
- probs: joy 0.682, neutral 0.214, anger 0.045, disgust 0.029, sadness 0.016, surprise 0.007, fear 0.006
- gate: g_text=0.9993, g_audio=0.0007
- IG target: predicted class `joy`; convergence delta text=1.83e-01 (n_steps=64), audio=2.85e-06 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `how` +0.5679 [cur], `guy` +0.4800 [cur], `]:` -0.3580 [cur], `know` -0.3218 [cur], `I` -0.3059 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [3.5-4.0 s] +0.01641, [3.0-3.5 s] -0.00787, [2.5-3.0 s] +0.00709
- figures: [tokens](figures/dev_dia58_utt1_tokens.png), [audio](figures/dev_dia58_utt1_audio.png)

## 16. `dev/dia82_utt12`  (stratified:neutral)

- speaker/text: **Joey**: Joey Tribbiani.
- true: **neutral**  |  predicted: **neutral** (correct)
- probs: neutral 0.930, joy 0.023, fear 0.016, sadness 0.013, surprise 0.007, anger 0.007, disgust 0.005
- gate: g_text=0.9997, g_audio=0.0003
- IG target: predicted class `neutral`; convergence delta text=8.23e-03 (n_steps=32), audio=6.85e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `i` +0.2913 [cur], `]:` +0.2450 [ctx], `Joey` +0.2333 [cur], `` +0.2131 [ctx], `res` +0.2122 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [1.0-1.5 s] +0.01087, [2.0-2.5 s] -0.01044, [0.5-1.0 s] -0.01024
- figures: [tokens](figures/dev_dia82_utt12_tokens.png), [audio](figures/dev_dia82_utt12_audio.png)

## 17. `dev/dia44_utt10`  (stratified:joy)

- speaker/text: **Ross**: Okay!
- true: **joy**  |  predicted: **joy** (correct)
- probs: joy 0.661, neutral 0.212, anger 0.069, surprise 0.026, fear 0.020, sadness 0.011, disgust 0.002
- gate: g_text=0.9994, g_audio=0.0006
- IG target: predicted class `joy`; convergence delta text=2.72e-01 (n_steps=64), audio=6.85e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `!` +2.0637 [cur], `you` +0.3788 [ctx], `call` +0.2500 [ctx], `Okay` -0.2324 [cur], `]:` +0.2175 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [0.0-0.5 s] +0.02322, [0.5-1.0 s] -0.01853
- figures: [tokens](figures/dev_dia44_utt10_tokens.png), [audio](figures/dev_dia44_utt10_audio.png)

## 18. `dev/dia26_utt10`  (stratified:sadness)

- speaker/text: **Monica**: Look, I'm sorry, all right.
- true: **sadness**  |  predicted: **neutral** (incorrect)
- probs: neutral 0.435, sadness 0.431, fear 0.079, joy 0.028, anger 0.015, surprise 0.009, disgust 0.002
- gate: g_text=0.9992, g_audio=0.0008
- IG target: predicted class `neutral`; convergence delta text=4.78e-02 (n_steps=64), audio=-1.07e-06 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `'m` +0.4585 [cur], `Look` -0.3865 [cur], `right` +0.3588 [cur], `Yeah` -0.2525 [ctx], `,` +0.2385 [cur]
- top-3 audio windows by |mean score| (0.5 s bins): [2.5-3.0 s] +0.01500, [1.5-2.0 s] +0.00855, [0.5-1.0 s] -0.00410
- figures: [tokens](figures/dev_dia26_utt10_tokens.png), [audio](figures/dev_dia26_utt10_audio.png)

## 19. `dev/dia49_utt11`  (stratified:anger)

- speaker/text: **Phoebe**: So just stop all the yelling, just stop it!
- true: **anger**  |  predicted: **anger** (correct)
- probs: anger 0.820, fear 0.114, sadness 0.034, joy 0.013, neutral 0.011, disgust 0.006, surprise 0.002
- gate: g_text=0.9858, g_audio=0.0142
- IG target: predicted class `anger`; convergence delta text=1.23e-01 (n_steps=64), audio=-8.34e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `stop` +0.3099 [cur], `stop` +0.1994 [cur], `yelling` +0.1847 [cur], `Ph` +0.1348 [ctx], `believe` +0.1144 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [2.0-2.5 s] +0.02210, [0.5-1.0 s] +0.01148, [0.0-0.5 s] -0.00577
- figures: [tokens](figures/dev_dia49_utt11_tokens.png), [audio](figures/dev_dia49_utt11_audio.png)

## 20. `dev/dia13_utt10`  (stratified:surprise)

- speaker/text: **Frank**: Hey, where's his thing?
- true: **surprise**  |  predicted: **neutral** (incorrect)
- probs: neutral 0.749, surprise 0.124, joy 0.072, anger 0.032, sadness 0.011, disgust 0.008, fear 0.005
- gate: g_text=0.9999, g_audio=0.0001
- IG target: predicted class `neutral`; convergence delta text=4.23e-01 (n_steps=64), audio=8.34e-07 (n_steps=32)
- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): `thing` +0.3677 [cur], `coming` +0.2584 [ctx], `ad` -0.2508 [ctx], `[` +0.1628 [ctx], `` +0.1394 [ctx]
- top-3 audio windows by |mean score| (0.5 s bins): [1.0-1.5 s] +0.01797, [0.0-0.5 s] +0.01123, [0.5-1.0 s] +0.01049
- figures: [tokens](figures/dev_dia13_utt10_tokens.png), [audio](figures/dev_dia13_utt10_audio.png)
