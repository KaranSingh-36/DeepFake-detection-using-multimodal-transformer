"""Window enumeration and labelling from LAV-DF style metadata.

A window [start, end) is labelled from the manipulated periods of its video:

    overlap >= min(min_overlap_seconds, min_overlap_fraction * period)  -> positive
    overlap == 0                                                        -> negative
    otherwise                                                           -> ambiguous

The per-modality labels combine that with the file level flags:

    video_fake = modify_video and positive
    audio_fake = modify_audio and positive
    any_fake   = video_fake or audio_fake

So a clean window inside a fake video is REAL, and an audio-only fake has
video_fake = 0 but audio_fake = 1, which is what forces the model to listen.
"""

import ast
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class VideoMeta:
    file: str
    split: str
    duration: float
    modify_video: bool
    modify_audio: bool
    fake_periods: list = field(default_factory=list)
    original: str | None = None

    @property
    def label(self):
        return int(self.modify_video or self.modify_audio)

    @property
    def fake_type(self):
        if self.modify_video and self.modify_audio:
            return "video+audio"
        if self.modify_video:
            return "video only"
        if self.modify_audio:
            return "audio only"
        return "real"


def parse_periods(value):
    if isinstance(value, (list, tuple)):
        periods = value
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        periods = []
    else:
        text = str(value).strip()
        periods = ast.literal_eval(text) if text else []
    return [(float(s), float(e)) for s, e in periods if float(e) > float(s)]


def load_metadata(csv_path, split=None):
    frame = pd.read_csv(csv_path)
    if split is not None:
        frame = frame[frame["split"] == split]
    metas = []
    for row in frame.itertuples(index=False):
        original = getattr(row, "original", None)
        if isinstance(original, float) and math.isnan(original):
            original = None
        metas.append(
            VideoMeta(
                file=str(row.file),
                split=str(row.split),
                duration=float(row.duration),
                modify_video=bool(int(row.modify_video)),
                modify_audio=bool(int(row.modify_audio)),
                fake_periods=parse_periods(row.fake_periods),
                original=original,
            )
        )
    return metas


def window_state(start, end, periods, min_overlap_seconds=0.2, min_overlap_fraction=0.5):
    """'positive', 'negative' or 'ambiguous' for the window [start, end)."""
    state = "negative"
    for p_start, p_end in periods:
        overlap = min(end, p_end) - max(start, p_start)
        if overlap <= 0:
            continue
        needed = min(min_overlap_seconds, min_overlap_fraction * (p_end - p_start))
        if overlap >= needed:
            return "positive"
        state = "ambiguous"
    return state


def window_labels(meta, start, end, min_overlap_seconds=0.2, min_overlap_fraction=0.5):
    state = window_state(start, end, meta.fake_periods,
                         min_overlap_seconds, min_overlap_fraction)
    positive = state == "positive"
    video = int(meta.modify_video and positive)
    audio = int(meta.modify_audio and positive)
    return {
        "video": video,
        "audio": audio,
        "any": int(video or audio),
        "ambiguous": state == "ambiguous",
    }


def enumerate_window_starts(duration, window_seconds, stride_seconds):
    """Window start times covering [0, duration]; the last one is end-aligned."""
    if duration <= window_seconds:
        return np.zeros(1, dtype=np.float32)
    n = int(math.floor((duration - window_seconds) / stride_seconds)) + 1
    starts = np.arange(n, dtype=np.float64) * stride_seconds
    last = duration - window_seconds
    if last - starts[-1] > 1e-3:
        starts = np.append(starts, last)
    return starts.astype(np.float32)


def window_class(labels):
    """0 real, 1 video only, 2 audio only, 3 both. Used for balanced sampling."""
    return labels["video"] + 2 * labels["audio"]


CLASS_NAMES = ["real", "video only", "audio only", "video+audio"]
