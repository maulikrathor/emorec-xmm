# MELD media inventory report

Generated: 2026-08-19T06:07:19.602234+00:00

Total .mp4 files found under root (junk included): 15908

## Every .mp4-containing directory under root

- dev/dev_splits_complete: 1112
- test/output_repeated_splits_test: 4807
- train/train_splits: 9989

## Per split

### train
- mp4 files total (raw glob, junk included): 9989
- appledouble_skipped: 0
- real_media: 9989
- labeled_rows: 9989
- matched: 9989
- missing_media: 0
- orphan_media: 0

### dev
- mp4 files total (raw glob, junk included): 1112
- appledouble_skipped: 0
- real_media: 1112
- labeled_rows: 1109
- matched: 1108
- missing_media: 1
- orphan_media: 4
  orphan examples: ['dia49_utt4', 'dia49_utt5', 'dia66_utt10', 'dia66_utt9']

### test
- mp4 files total (raw glob, junk included): 4807
- appledouble_skipped: 2060
- real_media: 2747
- labeled_rows: 2610
- matched: 2610
- missing_media: 0
- orphan_media: 137
  orphan examples: ['dia108_utt1', 'dia108_utt2', 'dia93_utt5', 'dia93_utt6', 'dia93_utt7', 'final_videos_testdia100_utt1', 'final_videos_testdia100_utt10', 'final_videos_testdia100_utt11', 'final_videos_testdia100_utt16', 'final_videos_testdia100_utt17', 'final_videos_testdia100_utt5', 'final_videos_testdia100_utt6', 'final_videos_testdia100_utt7', 'final_videos_testdia101_utt0', 'final_videos_testdia101_utt2', 'final_videos_testdia102_utt7', 'final_videos_testdia103_utt0', 'final_videos_testdia104_utt1', 'final_videos_testdia108_utt0', 'final_videos_testdia11_utt1']

## Cross-directory duplicate stems

Stems appearing in more than one .mp4-containing directory: 2068
- dia0_utt0: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia0_utt1: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia0_utt2: ['test/output_repeated_splits_test', 'train/train_splits']
- dia100_utt0: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia100_utt1: ['test/output_repeated_splits_test', 'train/train_splits']
- dia100_utt2: ['test/output_repeated_splits_test', 'train/train_splits']
- dia100_utt3: ['test/output_repeated_splits_test', 'train/train_splits']
- dia100_utt4: ['test/output_repeated_splits_test', 'train/train_splits']
- dia100_utt5: ['test/output_repeated_splits_test', 'train/train_splits']
- dia100_utt6: ['test/output_repeated_splits_test', 'train/train_splits']
- dia100_utt7: ['test/output_repeated_splits_test', 'train/train_splits']
- dia101_utt0: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia101_utt1: ['test/output_repeated_splits_test', 'train/train_splits']
- dia102_utt0: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia102_utt1: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia102_utt2: ['test/output_repeated_splits_test', 'train/train_splits']
- dia103_utt0: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia103_utt1: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia103_utt2: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']
- dia103_utt3: ['test/output_repeated_splits_test', 'train/train_splits', 'dev/dev_splits_complete']

## Conclusion

Total .mp4 across mirror: 15908
  = sum(real_media per split) (train=9989, dev=1112, test=2747 -> 13848) + sum(appledouble per split) (train=0, dev=0, test=2060 -> 2060) = 15908
Labeled utterances: 13708
  = sum(labeled_rows per split) (train=9989, dev=1109, test=2610 -> 13708)
Of the 13848 real media files, 13707 matched a labeled row, 141 did not (orphan_media), and 1 labeled row(s) had no real media (missing_media).
gap = sum(orphan_media) (141) + sum(appledouble) (2060) = 2201
15908 (resolved split dirs) equals the mirror total (15908); no residual outside the three resolved split directories.
