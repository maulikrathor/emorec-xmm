# MELD metadata report

Generated: 2026-08-19T06:06:54.212905+00:00

## Row counts

- train: 9989
- dev: 1109
- test: 2610

## R1 — per-split emotion distribution

### train
- neutral: 4710 (47.15%)
- joy: 1743 (17.45%)
- sadness: 683 (6.84%)
- anger: 1109 (11.1%)
- surprise: 1205 (12.06%)
- fear: 268 (2.68%)
- disgust: 271 (2.71%)

### dev
- neutral: 470 (42.38%)
- joy: 163 (14.7%)
- sadness: 111 (10.01%)
- anger: 153 (13.8%)
- surprise: 150 (13.53%)
- fear: 40 (3.61%)
- disgust: 22 (1.98%)

### test
- neutral: 1256 (48.12%)
- joy: 402 (15.4%)
- sadness: 208 (7.97%)
- anger: 345 (13.22%)
- surprise: 281 (10.77%)
- fear: 50 (1.92%)
- disgust: 68 (2.61%)

## R2 — utterance length statistics

- train: chars={'mean': 40.07, 'median': 32.0, 'min': 1.0, 'max': 327.0}, tokens={'mean': 7.96, 'median': 6.0, 'min': 1.0, 'max': 69.0}
- dev: chars={'mean': 39.77, 'median': 33.0, 'min': 1.0, 'max': 181.0}, tokens={'mean': 7.92, 'median': 6.0, 'min': 1.0, 'max': 37.0}
- test: chars={'mean': 41.43, 'median': 33.0, 'min': 2.0, 'max': 234.0}, tokens={'mean': 8.22, 'median': 7.0, 'min': 1.0, 'max': 45.0}

## R3 — turns per dialogue

- train: {'mean': 9.62, 'median': 9.0, 'min': 1.0, 'max': 24.0}
- dev: {'mean': 9.73, 'median': 10.0, 'min': 1.0, 'max': 23.0}
- test: {'mean': 9.32, 'median': 9.0, 'min': 1.0, 'max': 33.0}

## R4 — split integrity (Season, Episode, StartTime, EndTime) duplicates

Cross-split duplicate keys: 5
- [6, 4, '00:10:18,743', '00:10:21,536']: [{'split': 'train', 'utt_id': 'train/dia12_utt7'}, {'split': 'test', 'utt_id': 'test/dia85_utt0'}]
- [6, 4, '00:10:21,704', '00:10:24,706']: [{'split': 'train', 'utt_id': 'train/dia12_utt8'}, {'split': 'test', 'utt_id': 'test/dia85_utt1'}]
- [6, 24, '00:20:42,741', '00:20:45,159']: [{'split': 'train', 'utt_id': 'train/dia53_utt0'}, {'split': 'dev', 'utt_id': 'dev/dia52_utt2'}]
- [8, 24, '00:17:17,065', '00:17:18,066']: [{'split': 'train', 'utt_id': 'train/dia433_utt1'}, {'split': 'dev', 'utt_id': 'dev/dia77_utt0'}]
- [6, 17, '00:19:26,165', '00:19:29,083']: [{'split': 'train', 'utt_id': 'train/dia887_utt0'}, {'split': 'test', 'utt_id': 'test/dia117_utt8'}]

## R5 — episode overlap across splits

(Season, Episode) pairs appearing in more than one split: 156
MELD's official splits are drawn dialogue-wise from overlapping episodes, so this is expected dataset structure and NOT a bug introduced by this pipeline. Evidence for docs/KNOWN_LIMITATIONS.md #6 (speaker priors: the same episodes/speakers recur across splits).

## R6 — speakers

### train
distinct speakers: 260
- Joey: 1509
- Ross: 1459
- Rachel: 1435
- Phoebe: 1321
- Monica: 1299
- Chandler: 1283
- Janice: 58
- Carol: 46
- Emily: 43
- Tag: 41

### dev
distinct speakers: 47
- Ross: 217
- Phoebe: 185
- Rachel: 164
- Joey: 149
- Monica: 137
- Chandler: 101
- Frank: 17
- Carol: 13
- All: 10
- Susan: 10

### test
distinct speakers: 100
- Joey: 411
- Chandler: 379
- Ross: 373
- Rachel: 356
- Monica: 346
- Phoebe: 291
- Janice: 31
- Emily: 16
- Director: 16
- Gunther: 13

