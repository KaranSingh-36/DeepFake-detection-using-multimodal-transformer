# LAV-DF sample run (2026-10-17)

Two versions of the FusionTrace pipeline on the same sample of real LAV-DF
clips. Small sample, one seed per variant: treat differences of a few points
as noise.

- Data: random 3,000 train / 600 dev / 1,000 test clips (`make_sample.py`, seed 42),
  plus 421 untouched source clips cached for the modality swap. One test clip
  failed to decode in the v2 run (transient read error), so v2 uses 999.
- Hardware: RTX 3050 Laptop GPU (4 GB), shared with another GPU job during the
  v2 run, so the v2 timings in the logs are inflated.
- The threshold for every test number is the one chosen on dev.
- `v1_face_only/`: ViT on the face crop only, no modality dropout, step
  threshold by balanced accuracy, `best.pt` by dev video AUC.
- `v2_face_mouth_dropout/`: ViT on the face crop **and** a mouth-region crop,
  modality dropout 0.3, step threshold by F1, `best.pt` by dev video AUC
  (saturated at 1.0, so epoch 2 was kept; see below). `config.yaml` is the
  exact configuration.

## Test results, v1 (face only, no dropout)

| variant | clip AUC | bal. acc | real flagged | video-only flagged | audio-only flagged | both flagged | window AUC fused | video head | audio head | AP@0.5 | AP@0.75 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_attention | 0.999 | 0.997 | 0.0 % | 98.1 % | 100.0 % | 100.0 % | 0.993 | 0.851 | 0.995 | 0.805 | 0.422 |
| concat | 1.000 | 0.996 | 0.0 % | 98.1 % | 100.0 % | 99.6 % | 0.990 | 0.813 | 0.994 | 0.741 | 0.410 |
| video_only | 0.893 | 0.809 | 23.3 % | 91.8 % | 69.2 % | 92.7 % | 0.755 | 0.855 | - | 0.177 | 0.028 |
| audio_only | 0.999 | 0.997 | 0.0 % | 98.4 % | 100.0 % | 100.0 % | 0.989 | - | 0.995 | 0.767 | 0.411 |

## Test results, v2 (face + mouth, modality dropout 0.3)

| variant | clip AUC | bal. acc | real flagged | video-only flagged | audio-only flagged | both flagged | window AUC fused | video head | audio head | AP@0.5 | AP@0.75 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_attention | 0.999 | 0.995 | 0.0 % | 97.3 % | 99.6 % | 100.0 % | 0.991 | 0.875 | 0.994 | 0.759 | 0.380 |
| concat | 0.999 | 0.995 | 0.4 % | 98.0 % | 100.0 % | 100.0 % | 0.991 | 0.863 | 0.993 | 0.757 | 0.455 |
| video_only | 0.918 | 0.824 | 20.9 % | 94.1 % | 65.8 % | 95.0 % | 0.792 | 0.898 | - | 0.257 | 0.058 |

The audio-only variant was not retrained for v2: its audio features are
unchanged and modality dropout is disabled for unimodal models, so its v1 row
applies.

## What changed between v1 and v2

1. **The fused head now uses the frames.** With the audio track removed from
   40 fake test clips, v1 detected 0 % and v2 detects 47.5 % (52.5 % of the
   video-only fakes) while flagging 5 % of real clips. The remaining gap is
   the visual pathway itself: see 3.
2. **The video head improved** (window AUC 0.851 to 0.875 for cross-attention;
   the video-only variant window AUC from 0.755 to 0.792, clip AUC 0.893 to
   0.918), from the mouth stream alone, since the video head is trained the
   same way in both versions.
3. **Audio still dominates on LAV-DF.** Audio head window AUC 0.994 against
   0.875 for video, and "video only" fakes are still flagged 97 % of the time
   with the real source frames swapped in. The reason is in the data: LAV-DF
   time-edits the audio of "video only" fakes to fit the new segment (in 40
   of 40 sampled clips the audio is identical to the source before the fake
   period and differs inside it), so audio is a valid cue for every fake
   type in this dataset. The visual pathway can only be made competitive by
   stronger visual features (fine-tuning the top ViT blocks,
   `train.source=raw`), not by re-weighting.
4. **Cross-attention vs concat:** equal at clip level, cross-attention has
   the better video head (0.875 vs 0.863) and fewer false alarms on the
   swapped inputs (control clips flagged 9 % vs 11-16 %); concat has the
   better AP@0.75. One seed each: not a significant difference.
5. **Localisation:** the F1-chosen step threshold produces about as many
   segments as there are true ones (881 vs 864; v1: 1,035) and cleaner
   timelines (`timelines/`), at a small cost in AP@0.5 (0.805 to 0.759).
6. **Checkpoint selection:** dev video AUC reached 1.0 in epoch 2 and never
   improved, so the v2 `best.pt` is epoch 2, although epochs 5-6 had a better
   video head on dev (0.90). The default is now `train.select_by:
   window_level.fused_auc`, which keeps separating epochs.

## Modality swap, share flagged at 0.5 (150 clips per fake type)

| fake type | model input | v1 cross_attention | v2 cross_attention | v2 concat |
|---|---|---|---|---|
| video only | fake clip as-is | 0.97 | 0.98 | 0.99 |
| video only | fake frames + source audio | 0.52 | 0.49 | 0.63 |
| video only | source frames + fake audio | 0.97 | 0.97 | 0.97 |
| video only | source clip (control) | 0.05 | 0.09 | 0.11 |
| audio only | fake clip as-is | 1.00 | 1.00 | 1.00 |
| audio only | fake frames + source audio | 0.25 | 0.23 | 0.45 |
| audio only | source frames + fake audio | 1.00 | 1.00 | 1.00 |
| audio only | source clip (control) | 0.08 | 0.08 | 0.16 |
| video+audio | fake clip as-is | 1.00 | 1.00 | 1.00 |
| video+audio | fake frames + source audio | 0.52 | 0.51 | 0.64 |
| video+audio | source frames + fake audio | 1.00 | 1.00 | 1.00 |
| video+audio | source clip (control) | 0.06 | 0.09 | 0.16 |

The swap uses 0.5, not the dev threshold (0.965 for v2 cross-attention). At
the dev threshold 0 % of real test clips are flagged.

## Robustness probe (v2 cross_attention, 40 real + 40 fake + 40 video-only fake test clips)

| input | real flagged | fake flagged | video-only fake flagged |
|---|---|---|---|
| original file | 0.0 % | 100.0 % | 100.0 % |
| re-compressed (H.264 crf 32, AAC 48k) | 2.5 % | 100.0 % | 95.0 % |
| upscaled to 640x640 | 0.0 % | 100.0 % | 100.0 % |
| padded to 1280x720 (small face) | 0.0 % | 97.5 % | 97.5 % |
| audio removed | 5.0 % | 47.5 % | 52.5 % |

## Reproduce

```powershell
python results\lavdf_sample_run\make_sample.py
python scripts\build_cache.py --metadata results\lavdf_sample_run\sample.csv --splits train dev test
python scripts\build_cache.py --metadata results\lavdf_sample_run\cache_extra.csv --splits test
python scripts\train.py --metadata results\lavdf_sample_run\sample.csv --name xattn
python scripts\evaluate.py --checkpoint results\runs\xattn\best.pt --split test --swap --swap-per-type 150 --metadata results\lavdf_sample_run\sample.csv
```

Add `--set train.fusion=concat` (or `video_only`, `audio_only`) to `train.py`
for the other rows. For the v1 numbers use `--set data.visual_streams=[face]`
(in a separate `data.cache_dir`) and `--set train.modality_dropout=0`.
