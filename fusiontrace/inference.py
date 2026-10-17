"""Analyse an arbitrary video file: per-window scores, timeline, fake segments."""

import time

import numpy as np
import torch

from fusiontrace.config import Cfg
from fusiontrace.data import video_window_batch
from fusiontrace.features import build_encoders, decode_clip, encode_clip
from fusiontrace.evaluation import StepAccumulator
from fusiontrace.metrics import segments_from_curve
from fusiontrace.model import WindowFusionModel
from fusiontrace.windows import enumerate_window_starts


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Cfg.wrap(checkpoint["config"])
    model = WindowFusionModel(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["fusion_state_dict"])
    model.to(device).eval()
    return checkpoint, cfg, model


class VideoAnalyzer:
    def __init__(self, checkpoint_path, device=None, log=print):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint, self.cfg, self.model = load_checkpoint(checkpoint_path, self.device)
        self.threshold = float(self.checkpoint.get("threshold", 0.5))
        self.step_threshold = float(self.checkpoint.get("step_threshold", self.threshold))
        self.manifest = self.checkpoint["manifest"]
        log(f"Loading encoders {self.manifest['video_model_name']} + {self.manifest['audio_model_name']}")
        self.encoders = build_encoders(self.cfg, self.device).eval()
        encoder_state = self.checkpoint.get("encoders_state_dict")
        if encoder_state:
            self.encoders.load_state_dict(encoder_state, strict=False)

    def extract_features(self, video_path, keep_frames=False):
        clip = decode_clip(video_path, self.manifest["visual_fps"], self.manifest["image_size"],
                           self.manifest["face_crop"], self.manifest["sample_rate"],
                           visual_streams=self.manifest.get("visual_streams", ["face"]))
        features = encode_clip(self.encoders, clip, self.manifest["visual_fps"], self.device,
                               audio_chunk_seconds=self.manifest["audio_chunk_seconds"],
                               audio_hop_seconds=self.manifest["audio_hop_seconds"])
        features["has_audio"] = clip["has_audio"]
        if not clip["has_audio"]:
            # Same representation of "no audio" as modality dropout used in training:
            # zero features, so the fused head falls back to the visual evidence.
            features["audio"] = np.zeros_like(features["audio"])
        if keep_frames:
            features["frames"] = clip["frames"]
        return features

    @torch.inference_mode()
    def analyze(self, video_path, stride=None, threshold=None, step_threshold=None, max_windows_per_batch=256,
                keep_frames=False):
        """Returns a JSON-serialisable dict. With keep_frames=True the decoded
        face-cropped frames (uint8 [T, S, S, 3], one per visual step) are added
        under the key "frames" (not JSON-serialisable)."""
        started = time.perf_counter()
        threshold = self.threshold if threshold is None else float(threshold)
        step_threshold = self.step_threshold if step_threshold is None else float(step_threshold)
        window_seconds = float(self.cfg.window.seconds)
        stride = float(stride or self.cfg.window.eval_stride_seconds)

        features = self.extract_features(video_path, keep_frames=keep_frames)
        starts = enumerate_window_starts(features["duration"], window_seconds, stride)
        probs = {"video": [], "audio": [], "fused": []}
        fps, audio_hz, duration = features["visual_fps"], features["audio_hz"], features["duration"]
        acc = {"fused": StepAccumulator(duration, fps), "video": StepAccumulator(duration, fps),
               "audio": StepAccumulator(duration, audio_hz)}
        seen = {"video": False, "audio": False}
        use_amp = self.device.type == "cuda"
        for i in range(0, len(starts), max_windows_per_batch):
            chunk_starts = starts[i:i + max_windows_per_batch]
            batch = video_window_batch(features, chunk_starts, window_seconds)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                out = self.model(batch["visual"], batch["audio"], batch["visual_mask"], batch["audio_mask"])
            n = batch["visual"].shape[0]
            for name in probs:
                logit = out[f"{name}_logit"]
                probs[name].append(torch.sigmoid(logit.float()).cpu().numpy() if logit is not None else np.full(n, np.nan))
            vmask = batch["visual_mask"].cpu().numpy()
            amask = batch["audio_mask"].cpu().numpy()
            for name, mask in (("fused", vmask), ("video", vmask), ("audio", amask)):
                logit = out.get(f"{name}_step_logit")
                if logit is None:
                    continue
                if name in seen:
                    seen[name] = True
                steps = torch.sigmoid(logit.float()).cpu().numpy()
                for j, s in enumerate(chunk_starts):
                    acc[name].add(float(s), np.where(mask[j], steps[j], np.nan))
        probs = {k: np.concatenate(v) for k, v in probs.items()}
        ends = starts + window_seconds

        times = acc["fused"].times()
        resolution = 1.0 / fps
        fused_curve = acc["fused"].curve()
        curves = {"fused": fused_curve}
        if seen["video"]:
            curves["video"] = acc["video"].curve()
        if seen["audio"]:
            audio_times = acc["audio"].times()
            curves["audio"] = np.interp(times, audio_times, acc["audio"].curve())

        segments = []
        for s, e, score in segments_from_curve(times, fused_curve, step_threshold,
                                               float(self.cfg.eval.min_segment_seconds), resolution):
            inside = (times >= s) & (times < e)
            v = float(curves["video"][inside].mean()) if "video" in curves else float("nan")
            a = float(curves["audio"][inside].mean()) if "audio" in curves else float("nan")
            if np.isnan(v) or np.isnan(a):
                modality = "video" if np.isnan(a) else "audio"
            elif v >= step_threshold and a >= step_threshold:
                modality = "video+audio"
            else:
                modality = "video" if v >= a else "audio"
            segments.append({"start": round(s, 2), "end": round(e, 2), "score": round(score, 4),
                             "modality": modality, "video_score": None if np.isnan(v) else round(v, 4),
                             "audio_score": None if np.isnan(a) else round(a, 4)})

        video_score = float(np.nanmax(probs["fused"]))
        step = 1
        result = {
            "file": str(video_path),
            "duration": round(float(features["duration"]), 3),
            "has_audio": bool(features["has_audio"]),
            "threshold": threshold,
            "step_threshold": step_threshold,
            "video_score": round(video_score, 4),
            "label": "DEEPFAKE" if video_score >= threshold else "REAL",
            "fake_seconds": round(float(sum(seg["end"] - seg["start"] for seg in segments)), 2),
            "segments": segments,
            "max_video_head": None if np.isnan(probs["video"]).all() else round(float(np.nanmax(probs["video"])), 4),
            "max_audio_head": None if np.isnan(probs["audio"]).all() else round(float(np.nanmax(probs["audio"])), 4),
            "windows": [{"start": round(float(s), 2), "end": round(float(e), 2),
                         "p_video": None if np.isnan(pv) else round(float(pv), 4),
                         "p_audio": None if np.isnan(pa) else round(float(pa), 4),
                         "p_fused": round(float(pf), 4)}
                        for s, e, pv, pa, pf in zip(starts, ends, probs["video"], probs["audio"], probs["fused"])],
            "curve": {"times": [round(float(t), 2) for t in times[::step]],
                      **{name: [round(float(x), 4) for x in curve[::step]] for name, curve in curves.items()}},
            "window_seconds": window_seconds,
            "stride_seconds": stride,
            "device": str(self.device),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        if keep_frames:
            result["frames"] = features["frames"]
            result["frame_times"] = [round(float(t), 3) for t in (np.arange(len(features["frames"])) + 0.5) / fps]
        return result


def plot_timeline(result, path, periods=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["curve"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = {"fused": "#d62728", "video": "#1f77b4", "audio": "#2ca02c"}
    for name in ("fused", "video", "audio"):
        if name in curve:
            ax.plot(curve["times"], curve[name], label=f"{name} head", color=colors[name],
                    linewidth=2.2 if name == "fused" else 1.2)
    for seg in result["segments"]:
        ax.axvspan(seg["start"], seg["end"], color="#d62728", alpha=0.15)
    for s, e in periods or []:
        ax.axvspan(s, e, ymin=0.95, ymax=1.0, color="black", alpha=0.6)
    ax.axhline(result.get("step_threshold", result["threshold"]), color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, result["duration"])
    ax.set_xlabel("seconds")
    ax.set_ylabel("P(fake)")
    ax.set_title(f"{result['label']}  score={result['video_score']:.3f}  "
                 f"({len(result['segments'])} flagged segment(s))")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
