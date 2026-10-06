# Statistics — full train, seed 42

## Bootstrap 95% CI (1000 resamples, percentile method)

| model | split | weighted-F1 [CI] | macro-F1 [CI] | accuracy [CI] |
|---|---|---|---|---|
| text_only | dev | 0.5548 [0.5230, 0.5855] | 0.3760 [0.3500, 0.4021] | 0.5776 [0.5478, 0.6065] |
| audio_only | dev | 0.4012 [0.3687, 0.4342] | 0.2339 [0.2103, 0.2576] | 0.4558 [0.4242, 0.4865] |
| xattn | dev | 0.6155 [0.5832, 0.6448] | 0.4670 [0.4251, 0.5057] | 0.6363 [0.6074, 0.6634] |
| text_only | test | 0.6015 [0.5815, 0.6193] | 0.3948 [0.3734, 0.4151] | 0.6199 [0.6008, 0.6372] |
| audio_only | test | 0.4273 [0.4052, 0.4457] | 0.2222 [0.2063, 0.2357] | 0.4797 [0.4594, 0.4966] |
| xattn | test | 0.6173 [0.5973, 0.6367] | 0.4276 [0.4003, 0.4534] | 0.6387 [0.6195, 0.6559] |

## McNemar (exact, two-sided) on paired predictions

| comparison | split | n01 | n10 | p |
|---|---|---|---|---|
| xattn vs text_only | dev | 141 | 76 | 1.212e-05 |
| xattn vs audio_only | dev | 287 | 87 | 4.600e-26 |
| text_only vs audio_only | dev | 270 | 135 | 1.813e-11 |
| xattn vs text_only | test | 240 | 191 | 2.067e-02 |
| xattn vs audio_only | test | 630 | 215 | 5.550e-48 |
| text_only vs audio_only | test | 643 | 277 | 3.104e-34 |

Claim rule (ED12): p < 0.05 permits a difference claim. xattn > text_only is
significant on both dev and test. Note the test bootstrap CIs overlap; the
paired McNemar test is the correct instrument for same-data model comparison,
and it is significant. Both are reported.
