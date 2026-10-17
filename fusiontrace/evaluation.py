"""Score whole clips window by window and compute video, window and
localisation metrics. Shared by train.py (dev each epoch) and evaluate.py."""

import numpy as np
import pandas as pd
import torch

from fusiontrace.data import video_window_batch
from fusiontrace.metrics import (
    average_precision_at_iou,
    best_threshold,
    classification_report,
    frame_level_scores,
    safe_auc,
    safe_ap,
    segment_proposals,
    segments_from_curve,
)
from fusiontrace.windows import enumerate_window_starts, window_labels

FAKE_TYPES = ["real", "video only", "audio only", "video+audio"]


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _probs(outputs, key, n):
    logit = outputs.get(key)
    if logit is None:
        return np.full(n, np.nan, dtype=np.float32)
    return torch.sigmoid(logit.float()).cpu().numpy()


class StepAccumulator:
    """Average per-step probabilities of overlapping windows onto one clip grid."""

    def __init__(self, duration, rate):
        self.rate = float(rate)
        n = max(1, int(np.ceil(duration * self.rate)))
        self.sum = np.zeros(n)
        self.count = np.zeros(n)

    def add(self, start, probs):
        centres = start + (np.arange(len(probs)) + 0.5) / self.rate
        index = np.floor(centres * self.rate).astype(int)
        keep = (index >= 0) & (index < len(self.sum)) & ~np.isnan(probs)
        np.add.at(self.sum, index[keep], probs[keep])
        np.add.at(self.count, index[keep], 1.0)

    def curve(self):
        return np.where(self.count > 0, self.sum / np.maximum(self.count, 1), 0.0)

    def times(self):
        return (np.arange(len(self.sum)) + 0.5) / self.rate


def pool_scores(probs, method="max"):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[~np.isnan(probs)]
    if probs.size == 0:
        return float("nan")
    if method == "top3":
        return float(np.sort(probs)[-3:].mean())
    return float(probs.max())


@torch.inference_mode()
def score_videos(model, cache, metas, cfg, device, stride=None, batch_windows=256,
                 amp=True, feature_override=None, progress_every=500, log=print):
    """Run the fusion model over every window of every clip.

    feature_override(meta, features) may modify the cached features (used by the
    modality swap experiment). Returns (window_frame, video_records).
    """
    model.eval()
    window_seconds = float(cfg.window.seconds)
    stride = float(stride if stride is not None else cfg.window.eval_stride_seconds)
    pooling = cfg.eval.score_pooling
    use_amp = amp and device.type == "cuda"

    window_rows = []
    video_records = []
    pending = []          # (meta, starts, batch) waiting to be scored together
    pending_windows = 0

    def flush():
        nonlocal pending, pending_windows
        if not pending:
            return
        batch = {key: torch.cat([b[key] for _, _, b, _ in pending]) for key in pending[0][2]}
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch["visual"], batch["audio"], batch["visual_mask"], batch["audio_mask"])
        n = batch["visual"].shape[0]
        p_video, p_audio, p_fused = _probs(outputs, "video_logit", n), _probs(outputs, "audio_logit", n), _probs(outputs, "fused_logit", n)
        step_fused = torch.sigmoid(outputs["fused_step_logit"].float()).cpu().numpy() if outputs.get("fused_step_logit") is not None else None
        step_video = torch.sigmoid(outputs["video_step_logit"].float()).cpu().numpy() if outputs.get("video_step_logit") is not None else None
        step_audio = torch.sigmoid(outputs["audio_step_logit"].float()).cpu().numpy() if outputs.get("audio_step_logit") is not None else None
        visual_mask_np = batch["visual_mask"].cpu().numpy()
        audio_mask_np = batch["audio_mask"].cpu().numpy()
        offset = 0
        for meta, starts, _, rates in pending:
            k = len(starts)
            sl = slice(offset, offset + k)
            offset += k
            ends = starts + window_seconds
            visual_fps, audio_hz, clip_duration = rates
            acc_fused = StepAccumulator(clip_duration, visual_fps)
            acc_video = StepAccumulator(clip_duration, visual_fps) if step_video is not None else None
            acc_audio = StepAccumulator(clip_duration, audio_hz) if step_audio is not None else None
            for i, row in enumerate(range(sl.start, sl.stop)):
                vm = visual_mask_np[row]
                if step_fused is not None:
                    acc_fused.add(float(starts[i]), np.where(vm, step_fused[row], np.nan))
                if acc_video is not None:
                    acc_video.add(float(starts[i]), np.where(vm, step_video[row], np.nan))
                if acc_audio is not None:
                    am = audio_mask_np[row]
                    acc_audio.add(float(starts[i]), np.where(am, step_audio[row], np.nan))
            labels = [window_labels(meta, float(s), float(e), cfg.window.min_overlap_seconds,
                                    cfg.window.min_overlap_fraction) for s, e in zip(starts, ends)]
            for i in range(k):
                window_rows.append((meta.file, float(starts[i]), float(ends[i]),
                                    float(p_video[sl][i]), float(p_audio[sl][i]), float(p_fused[sl][i]),
                                    labels[i]["video"], labels[i]["audio"], labels[i]["any"], labels[i]["ambiguous"]))
            video_records.append({
                "file": meta.file,
                "fake_type": meta.fake_type,
                "label": meta.label,
                "modify_video": int(meta.modify_video),
                "modify_audio": int(meta.modify_audio),
                "duration": meta.duration,
                "periods": meta.fake_periods,
                "starts": starts.astype(np.float32),
                "p_video": p_video[sl].copy(),
                "p_audio": p_audio[sl].copy(),
                "p_fused": p_fused[sl].copy(),
                "score_fused": pool_scores(p_fused[sl], pooling),
                "score_video": pool_scores(p_video[sl], pooling),
                "score_audio": pool_scores(p_audio[sl], pooling),
                "step_times": acc_fused.times(),
                "step_fused": acc_fused.curve(),
                "step_video": acc_video.curve() if acc_video is not None else None,
                "audio_step_times": acc_audio.times() if acc_audio is not None else None,
                "step_audio": acc_audio.curve() if acc_audio is not None else None,
                "step_rate": visual_fps,
            })
        pending = []
        pending_windows = 0

    for index, meta in enumerate(metas):
        features = cache.read(meta.file)
        if feature_override is not None:
            features = feature_override(meta, features)
        starts = enumerate_window_starts(features["duration"], window_seconds, stride)
        batch = video_window_batch(features, starts, window_seconds)
        pending.append((meta, starts, batch, (features["visual_fps"], features["audio_hz"], features["duration"])))
        pending_windows += len(starts)
        if pending_windows >= batch_windows:
            flush()
        if progress_every and (index + 1) % progress_every == 0:
            log(f"  scored {index + 1:,}/{len(metas):,} clips")
    flush()

    window_frame = pd.DataFrame(window_rows, columns=[
        "file", "start", "end", "p_video", "p_audio", "p_fused",
        "y_video", "y_audio", "y_any", "ambiguous"])
    return window_frame, video_records


_ARRAY_KEYS = ("starts", "p_video", "p_audio", "p_fused", "periods", "step_times", "step_fused",
               "step_video", "audio_step_times", "step_audio")


def video_frame(video_records):
    return pd.DataFrame([{k: v for k, v in rec.items() if k not in _ARRAY_KEYS} for rec in video_records])


def summarize(window_frame, video_records, threshold=None, head="fused"):
    """Video-level and window-level metrics. threshold=None -> tuned on this data."""
    videos = video_frame(video_records)
    scores = videos[f"score_{head}"].to_numpy()
    labels = videos["label"].to_numpy()
    tuned_here = threshold is None
    if tuned_here:
        threshold = best_threshold(labels, scores)

    report = {"video_level": classification_report(labels, scores, threshold)}
    report["video_level"]["threshold_source"] = "tuned_here" if tuned_here else "given"

    per_type = {}
    for fake_type in FAKE_TYPES:
        subset = videos[videos["fake_type"] == fake_type]
        if len(subset) == 0:
            continue
        flagged = (subset[f"score_{head}"] >= threshold).mean()
        per_type[fake_type] = {
            "n": int(len(subset)),
            "flagged_rate": float(flagged),
            "mean_score": float(subset[f"score_{head}"].mean()),
        }
    report["per_fake_type"] = per_type

    # which modality head fires on which fake type (modality attribution)
    attribution = {}
    for fake_type in FAKE_TYPES:
        subset = videos[videos["fake_type"] == fake_type]
        if len(subset) == 0:
            continue
        attribution[fake_type] = {
            "video_head_mean_score": float(np.nanmean(subset["score_video"])) if subset["score_video"].notna().any() else None,
            "audio_head_mean_score": float(np.nanmean(subset["score_audio"])) if subset["score_audio"].notna().any() else None,
        }
    report["modality_attribution"] = attribution

    clean = window_frame[~window_frame["ambiguous"]]
    report["window_level"] = {
        "n_windows": int(len(clean)),
        "fused_auc": safe_auc(clean["y_any"], clean["p_fused"]),
        "fused_ap": safe_ap(clean["y_any"], clean["p_fused"]),
        "video_head_auc": safe_auc(clean["y_video"], clean["p_video"]) if clean["p_video"].notna().any() else None,
        "audio_head_auc": safe_auc(clean["y_audio"], clean["p_audio"]) if clean["p_audio"].notna().any() else None,
        "positive_rate": float(clean["y_any"].mean()) if len(clean) else None,
    }
    report["video_auc"] = report["video_level"]["auc"]
    report["threshold"] = float(threshold)
    return report


def localisation(video_records, cfg, step_threshold=None, curve_key="step_fused"):
    """Temporal localisation from the per-step curves.

    - frame-level AUC/AP of the dense curve against the fake periods
    - segment AP at several IoU thresholds, using multi-threshold proposals
      with NMS (threshold free)
    - `step_threshold`: tuned here for the best frame-level balanced accuracy
      when None (train on dev), otherwise the given value is used to report
      segments at a single operating point.
    """
    min_seg = float(cfg.eval.min_segment_seconds)
    predictions, ground_truth = {}, {}
    frame_truth, frame_scores = [], []
    for rec in video_records:
        times, curve = rec["step_times"], rec[curve_key]
        resolution = 1.0 / float(rec["step_rate"])
        ground_truth[rec["file"]] = list(rec["periods"])
        predictions[rec["file"]] = segment_proposals(times, curve, resolution, min_seg)
        frame_truth.append(frame_level_scores(times, curve, rec["periods"]))
        frame_scores.append(curve)
    frame_truth = np.concatenate(frame_truth) if frame_truth else np.zeros(0)
    frame_scores = np.concatenate(frame_scores) if frame_scores else np.zeros(0)
    if step_threshold is None:
        step_threshold = best_threshold(frame_truth, frame_scores) if len(frame_truth) else 0.5

    at_threshold = {rec["file"]: segments_from_curve(rec["step_times"], rec[curve_key], step_threshold, min_seg,
                                                     1.0 / float(rec["step_rate"])) for rec in video_records}
    result = {
        "frame_level_auc": safe_auc(frame_truth, frame_scores),
        "frame_level_ap": safe_ap(frame_truth, frame_scores),
        "frame_level_balanced_accuracy": float(((frame_scores >= step_threshold) == frame_truth.astype(bool))[frame_truth == 1].mean() / 2
                                               + ((frame_scores < step_threshold))[frame_truth == 0].mean() / 2)
        if len(frame_truth) and frame_truth.sum() and (frame_truth == 0).sum() else float("nan"),
        "step_threshold": float(step_threshold),
        "n_ground_truth_segments": int(sum(len(v) for v in ground_truth.values())),
        "n_proposals": int(sum(len(v) for v in predictions.values())),
        "n_segments_at_threshold": int(sum(len(v) for v in at_threshold.values())),
    }
    for iou in cfg.eval.iou_thresholds:
        result[f"ap@{iou}"] = average_precision_at_iou(predictions, ground_truth, float(iou))
    if ground_truth and any(ground_truth.values()):
        result["ap@0.5_at_threshold"] = average_precision_at_iou(at_threshold, ground_truth, 0.5)
    return result


def modality_swap(model, cache, metas, cfg, device, max_per_type=200, seed=42, log=print):
    """Replace one modality of a fake clip with its untouched source clip.

    Reports the mean video-level score under four conditions. A model that uses
    both streams should drop when the manipulated stream is swapped out and stay
    when the untouched stream is swapped out.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fake_type in ("video only", "audio only", "video+audio"):
        candidates = [m for m in metas if m.fake_type == fake_type and m.original and cache.has(m.original)]
        if not candidates:
            continue
        if len(candidates) > max_per_type:
            candidates = [candidates[i] for i in rng.choice(len(candidates), max_per_type, replace=False)]
        log(f"  swap experiment: {fake_type}, {len(candidates)} clips")

        def swap(which):
            def override(meta, features):
                source = cache.read(meta.original)
                out = dict(features)
                if which in ("audio", "both"):
                    out["audio"] = source["audio"]
                    out["audio_hz"] = source["audio_hz"]
                if which in ("visual", "both"):
                    out["visual"] = source["visual"]
                    out["visual_fps"] = source["visual_fps"]
                if which == "both":
                    out["duration"] = source["duration"]
                return out
            return override

        conditions = {
            "fake clip as-is": None,
            "fake frames + source audio": swap("audio"),
            "source frames + fake audio": swap("visual"),
            "source clip (control)": swap("both"),
        }
        for name, override in conditions.items():
            _, records = score_videos(model, cache, candidates, cfg, device,
                                      feature_override=override, progress_every=0, log=log)
            scores = np.array([r["score_fused"] for r in records])
            rows.append({
                "fake type": fake_type,
                "model input": name,
                "clips": len(records),
                "mean_score": float(np.nanmean(scores)),
                "flagged_at_0.5": float(np.nanmean(scores >= 0.5)),
            })
    return pd.DataFrame(rows)
