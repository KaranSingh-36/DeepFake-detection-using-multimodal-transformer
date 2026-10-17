# LAV-DF sample run (2026-10-17)

First run of the FusionTrace pipeline on real LAV-DF clips. It is a small
sample, one seed per variant, so treat differences of a few points as noise.

- Data: random 3,000 train / 600 dev / 1,000 test clips (`make_sample.py`, seed 42),
  plus 421 untouched source clips cached for the modality swap.
- Settings: `configs/default.yaml` unchanged (frozen encoders, 8 epochs, 40,000 windows per epoch).
- Hardware: RTX 3050 Laptop GPU (4 GB). Feature cache 250 clips/min, training 40 s per epoch.
- Threshold for every test number is the one chosen on dev.

## Test results (1,000 clips)

| variant         | video AUC | balanced acc | real flagged | video-only flagged | audio-only flagged | both flagged | window AUC | segment AP@0.5 | AP@0.75 |
| --------------- | --------- | ------------ | ------------ | ------------------ | ------------------ | ------------ | ---------- | -------------- | ------- |
| cross_attention | 0.999     | 0.997        | 0.0 %        | 98.1 %             | 100 %              | 100 %        | 0.993      | 0.805          | 0.422   |
| concat          | 1.000     | 0.996        | 0.0 %        | 98.1 %             | 100 %              | 99.6 %       | 0.990      | 0.741          | 0.410   |
| audio_only      | 0.999     | 0.997        | 0.0 %        | 98.4 %             | 100 %              | 100 %        | 0.989      | 0.767          | 0.411   |
| video_only      | 0.893     | 0.809        | 23.3 %       | 91.8 %             | 69.2 %             | 92.7 %       | 0.755      | 0.177          | 0.028   |

## What the numbers mean

1. **Detection now follows the manipulated segment.** Window AUC is 0.99 and the
   timelines land on the annotated fake periods (`timelines/`). The legacy model
   could not do this.
2. **The audio stream carries almost all of it.** The audio-only variant matches
   cross-attention at clip level, and removing the audio track from 40 fake test
   clips dropped detection from 100 % to 0 %.
3. **"Video only" fakes are detected through their audio.** In the swap table,
   real source frames with the fake clip's audio are still flagged 97 % of the
   time. The cause is in the dataset: in 40 of 40 sampled "video only" fakes the
   audio is identical to the source before the fake period, then differs inside
   it (median waveform correlation 0.51) and the clip length changes (median
   0.13 s). LAV-DF time-edits the "unmodified" stream to fit the new segment.
4. **The visual pathway is weak.** Video head window AUC 0.85; the video-only
   variant flags 23 % of real clips. Frozen ImageNet ViT features at 5 fps do
   not see lip-sync edits well.
5. **Cross-attention is not shown to help yet.** It matches concat and audio-only
   at clip level; its segment AP@0.5 is higher (0.805 vs 0.741 / 0.767), from a
   single run each.

## Modality swap, share flagged at 0.5 (150 clips per fake type)

| fake type   | model input                | cross_attention | concat | video_only | audio_only |
| ----------- | -------------------------- | --------------- | ------ | ---------- | ---------- |
| video only  | fake clip as-is            | 0.97            | 0.99   | 0.97       | 0.98       |
| video only  | fake frames + source audio | 0.52            | 0.61   | 0.97       | 0.07       |
| video only  | source frames + fake audio | 0.97            | 0.99   | 0.39       | 0.98       |
| video only  | source clip (control)      | 0.05            | 0.26   | 0.39       | 0.03       |
| audio only  | fake clip as-is            | 1.00            | 1.00   | 0.88       | 1.00       |
| audio only  | fake frames + source audio | 0.25            | 0.48   | 0.88       | 0.17       |
| audio only  | source frames + fake audio | 1.00            | 1.00   | 0.35       | 1.00       |
| audio only  | source clip (control)      | 0.08            | 0.21   | 0.33       | 0.03       |
| video+audio | fake clip as-is            | 1.00            | 1.00   | 0.97       | 1.00       |
| video+audio | fake frames + source audio | 0.52            | 0.64   | 0.97       | 0.17       |
| video+audio | source frames + fake audio | 1.00            | 1.00   | 0.34       | 1.00       |
| video+audio | source clip (control)      | 0.06            | 0.20   | 0.32       | 0.04       |

The swap uses 0.5, not the dev threshold (0.998 for cross-attention). At 0.5 the
cross-attention model flags 7.2 % of real test clips; at the dev threshold, 0 %.

## Robustness probe (cross_attention, 40 real + 40 fake test clips)

| input                                 | real flagged | fake flagged |
| ------------------------------------- | ------------ | ------------ |
| original file                         | 0 %          | 100 %        |
| re-compressed (H.264 crf 32, AAC 48k) | 2 %          | 100 %        |
| upscaled to 640x640                   | 0 %          | 100 %        |
| padded to 1280x720 (small face)       | 0 %          | 98 %         |
| audio removed                         | 0 %          | 0 %          |

## Reproduce

```powershell
python results\lavdf_sample_run\make_sample.py
python scripts\build_cache.py --metadata results\lavdf_sample_run\sample.csv --splits train dev test
python scripts\build_cache.py --metadata results\lavdf_sample_run\cache_extra.csv --splits test
python scripts\train.py --metadata results\lavdf_sample_run\sample.csv --name real_xattn
python scripts\evaluate.py --checkpoint results\runs\real_xattn\best.pt --split test --swap --swap-per-type 150 --metadata results\lavdf_sample_run\sample.csv
```

Add `--set train.fusion=concat` (or `video_only`, `audio_only`) to `train.py` for the other rows.
