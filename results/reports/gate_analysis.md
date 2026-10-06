# Gate analysis

- checkpoint: `artifacts/runs/xattn-seed42-20261006T070547Z/best.pt`  |  config: `configs/xattn.yaml`  |  model: `xattn`
- manifest: `artifacts/manifests/dev.jsonl`  |  n = 1108 utterances  |  seed: 42
- generated (UTC): 2026-10-06 07:59:07Z

The gate `g = (g_text, g_audio)` is reported exactly as the model produced it; nothing is rescaled or re-centred (docs/BACKLOG.md J-08).

## g_audio overall

| mean | median | std | p05 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|
| 0.0013 | 0.0003 | 0.0039 | 0.0001 | 0.0002 | 0.0003 | 0.0009 | 0.0047 |

## Dominance

- audio-dominant (g_audio > 0.5): 0.0000
- saturated toward audio (g_audio > 0.95): 0.0000
- saturated toward text (g_audio < 0.05): 0.9991

## g_audio per TRUE class

| class | n | mean g_audio | std |
|---|---|---|---|
| neutral | 469 | 0.0005 | 0.0016 |
| joy | 163 | 0.0008 | 0.0013 |
| sadness | 111 | 0.0024 | 0.0071 |
| anger | 153 | 0.0030 | 0.0050 |
| surprise | 150 | 0.0005 | 0.0009 |
| fear | 40 | 0.0045 | 0.0085 |
| disgust | 22 | 0.0029 | 0.0088 |

## g_audio by prediction correctness

| | n | mean g_audio |
|---|---|---|
| correct | 705 | 0.0011 |
| incorrect | 403 | 0.0016 |

## Most audio-dominant (top 10)

| utt_id | g_audio | true | pred |
|---|---|---|---|
| dev/dia56_utt9 | 0.0692 | sadness | sadness |
| dev/dia104_utt9 | 0.0463 | fear | fear |
| dev/dia36_utt11 | 0.0431 | disgust | fear |
| dev/dia19_utt3 | 0.0335 | anger | sadness |
| dev/dia26_utt4 | 0.0293 | neutral | sadness |
| dev/dia34_utt6 | 0.0233 | fear | fear |
| dev/dia14_utt8 | 0.0231 | anger | anger |
| dev/dia56_utt2 | 0.0217 | sadness | fear |
| dev/dia18_utt10 | 0.0213 | anger | anger |
| dev/dia82_utt8 | 0.0199 | anger | anger |

## Most text-dominant (top 10)

| utt_id | g_audio | true | pred |
|---|---|---|---|
| dev/dia10_utt8 | 0.0000 | neutral | neutral |
| dev/dia110_utt2 | 0.0000 | neutral | neutral |
| dev/dia60_utt5 | 0.0000 | joy | neutral |
| dev/dia47_utt0 | 0.0000 | neutral | neutral |
| dev/dia15_utt4 | 0.0000 | surprise | surprise |
| dev/dia1_utt1 | 0.0000 | joy | joy |
| dev/dia4_utt10 | 0.0000 | neutral | neutral |
| dev/dia95_utt4 | 0.0000 | neutral | neutral |
| dev/dia80_utt5 | 0.0000 | neutral | neutral |
| dev/dia15_utt5 | 0.0000 | neutral | neutral |
