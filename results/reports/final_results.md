# Pilot results

## Run header

- split names: `{'dev': 'artifacts/manifests/dev.jsonl', 'train': 'artifacts/manifests/train.jsonl'}`
- dev set size: 1108 (from `artifacts/manifests/dev.jsonl`)
- subset size(s): full
- seed(s): 42
- date(s) (UTC, from run directory names): 20261006T064248Z, 20261006T065407Z, 20261006T070547Z
- git SHA(s): efb2451c6932716fd64582b5f97f4e9310b4f15f

## Results

| model | train subset size | seed | epochs run | dev accuracy | dev macro-F1 | dev weighted-F1 |
|---|---|---|---|---|---|---|
| audio_only | full | 42 | 10 | 0.4558 | 0.2339 | 0.4012 |
| text_only | full | 42 | 10 | 0.5776 | 0.3760 | 0.5548 |
| xattn | full | 42 | 8 | 0.6363 | 0.4670 | 0.6155 |

### Trivial baseline (majority-class reference)

| model | train subset size | seed | epochs run | dev accuracy | dev macro-F1 | dev weighted-F1 |
|---|---|---|---|---|---|---|
| all-neutral (majority class) | - | - | - | 0.4233 | 0.0850 | 0.2518 |

(computed on `artifacts/manifests/dev.jsonl`, 1108 examples)

## Per-class F1 (best checkpoint, dev)

| model | neutral | joy | sadness | anger | surprise | fear | disgust |
|---|---|---|---|---|---|---|---|
| audio_only | 0.6581 | 0.2086 | 0.1417 | 0.3387 | 0.2081 | 0.0000 | 0.0000 |
| text_only | 0.7553 | 0.5817 | 0.3128 | 0.4902 | 0.5284 | 0.0952 | 0.0000 |
| xattn | 0.7786 | 0.5971 | 0.2166 | 0.5008 | 0.5535 | 0.2056 | 0.1412 |
| all-neutral (trivial) | 0.5948 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

