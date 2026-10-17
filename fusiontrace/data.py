"""Window datasets over the feature cache (fast path) or raw videos."""

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from fusiontrace.media import read_audio, read_frames
from fusiontrace.windows import (
    CLASS_NAMES,
    enumerate_window_starts,
    window_class,
    window_labels,
)


# ----------------------------------------------------------------------------
# Slicing cached features
# ----------------------------------------------------------------------------
def slice_steps(sequence, rate, start, window_seconds):
    """Take round(window*rate) steps beginning at round(start*rate); zero-pad.

    sequence: [..., T, D] -> ([..., N, D] float32, mask [N] bool)
    """
    count = max(1, int(round(window_seconds * rate)))
    first = int(round(start * rate))
    total = sequence.shape[-2]
    first = min(max(first, 0), max(total - 1, 0))
    piece = sequence[..., first:first + count, :]
    valid = piece.shape[-2]
    if valid < count:
        pad_shape = list(piece.shape)
        pad_shape[-2] = count - valid
        piece = np.concatenate([piece, np.zeros(pad_shape, dtype=piece.dtype)], axis=-2)
    mask = np.zeros(count, dtype=bool)
    mask[:max(valid, 1)] = True
    return piece.astype(np.float32, copy=False), mask


def window_tensors(features, start, window_seconds):
    visual, visual_mask = slice_steps(features["visual"], features["visual_fps"], start, window_seconds)
    audio, audio_mask = slice_steps(features["audio"], features["audio_hz"], start, window_seconds)
    return {
        "visual": torch.from_numpy(visual),
        "visual_mask": torch.from_numpy(visual_mask),
        "audio": torch.from_numpy(audio),
        "audio_mask": torch.from_numpy(audio_mask),
    }


def video_window_batch(features, starts, window_seconds):
    """All windows of one clip stacked into a batch (for evaluation/inference)."""
    items = [window_tensors(features, float(s), window_seconds) for s in starts]
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


# ----------------------------------------------------------------------------
# Window table
# ----------------------------------------------------------------------------
class WindowTable:
    """Candidate (video, start) pairs for a list of VideoMeta with labels."""

    def __init__(self, metas, window_seconds, stride_seconds, min_overlap_seconds,
                 min_overlap_fraction, drop_ambiguous=True):
        self.metas = metas
        self.window_seconds = float(window_seconds)
        self.min_overlap_seconds = float(min_overlap_seconds)
        self.min_overlap_fraction = float(min_overlap_fraction)

        video_idx, starts, classes = [], [], []
        for index, meta in enumerate(metas):
            for start in enumerate_window_starts(meta.duration, window_seconds, stride_seconds):
                labels = window_labels(meta, float(start), float(start) + window_seconds,
                                       min_overlap_seconds, min_overlap_fraction)
                if labels["ambiguous"] and drop_ambiguous:
                    continue
                video_idx.append(index)
                starts.append(float(start))
                classes.append(window_class(labels))

        self.video_idx = np.asarray(video_idx, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.float32)
        self.classes = np.asarray(classes, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def class_counts(self):
        return {CLASS_NAMES[c]: int((self.classes == c).sum()) for c in range(4)}

    def balanced_weights(self):
        counts = np.bincount(self.classes, minlength=4).astype(np.float64)
        weights = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
        return weights[self.classes]


def make_balanced_sampler(table, num_samples, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(table.balanced_weights(), dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


def step_targets(meta, start, window_seconds, rate, count=None):
    """Per-step targets inside the window: (video [N], audio [N], any [N]) float32.

    A step is fake when its centre time lies inside a manipulated period.
    """
    count = count if count is not None else max(1, int(round(window_seconds * rate)))
    centres = start + (np.arange(count) + 0.5) / rate
    inside = np.zeros(count, dtype=bool)
    for p_start, p_end in meta.fake_periods:
        inside |= (centres >= p_start) & (centres < p_end)
    inside = inside.astype(np.float32)
    return inside * float(meta.modify_video), inside * float(meta.modify_audio), inside * float(meta.label)


def step_target_tensors(meta, start, window_seconds, visual_rate, audio_rate):
    video_steps, _, any_steps = step_targets(meta, start, window_seconds, visual_rate)
    _, audio_steps, _ = step_targets(meta, start, window_seconds, audio_rate)
    return {
        "y_video_steps": torch.from_numpy(video_steps),
        "y_audio_steps": torch.from_numpy(audio_steps),
        "y_any_steps": torch.from_numpy(any_steps),
    }


def labels_to_tensors(labels):
    return {
        "y_video": torch.tensor(float(labels["video"])),
        "y_audio": torch.tensor(float(labels["audio"])),
        "y_any": torch.tensor(float(labels["any"])),
        "ambiguous": torch.tensor(bool(labels["ambiguous"])),
    }


# ----------------------------------------------------------------------------
# Cached dataset
# ----------------------------------------------------------------------------
class CachedWindowDataset(Dataset):
    def __init__(self, table, cache, jitter_seconds=0.0, seed=0):
        self.table = table
        self.cache = cache
        self.jitter = float(jitter_seconds)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.table)

    def _pick_start(self, meta, nominal):
        if self.jitter <= 0:
            return nominal, window_labels(meta, nominal, nominal + self.table.window_seconds,
                                          self.table.min_overlap_seconds,
                                          self.table.min_overlap_fraction)
        upper = max(0.0, meta.duration - self.table.window_seconds)
        for _ in range(3):
            start = nominal + self.rng.uniform(-self.jitter, self.jitter)
            start = min(max(start, 0.0), upper)
            labels = window_labels(meta, start, start + self.table.window_seconds,
                                   self.table.min_overlap_seconds,
                                   self.table.min_overlap_fraction)
            if not labels["ambiguous"]:
                return start, labels
        return nominal, window_labels(meta, nominal, nominal + self.table.window_seconds,
                                      self.table.min_overlap_seconds,
                                      self.table.min_overlap_fraction)

    def __getitem__(self, index):
        meta = self.table.metas[int(self.table.video_idx[index])]
        start, labels = self._pick_start(meta, float(self.table.starts[index]))
        features = self.cache.read(meta.file)
        item = window_tensors(features, start, self.table.window_seconds)
        item.update(labels_to_tensors(labels))
        item.update(step_target_tensors(meta, start, self.table.window_seconds,
                                        features["visual_fps"], features["audio_hz"]))
        item["start"] = torch.tensor(start)
        item["file"] = meta.file
        return item


# ----------------------------------------------------------------------------
# Raw dataset (decodes video; for end-to-end fine-tuning)
# ----------------------------------------------------------------------------
class RawWindowDataset(Dataset):
    def __init__(self, table, dataset_root, visual_fps, image_size=224, face_crop=True,
                 sample_rate=16000, jitter_seconds=0.0, seed=0, audio_hz=25.0):
        self.audio_hz = float(audio_hz)
        self.table = table
        self.root = dataset_root
        self.visual_fps = float(visual_fps)
        self.image_size = int(image_size)
        self.face_crop = bool(face_crop)
        self.sample_rate = int(sample_rate)
        self.jitter = float(jitter_seconds)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        meta = self.table.metas[int(self.table.video_idx[index])]
        nominal = float(self.table.starts[index])
        upper = max(0.0, meta.duration - self.table.window_seconds)
        start = min(max(nominal + self.rng.uniform(-self.jitter, self.jitter), 0.0), upper) if self.jitter > 0 else nominal
        labels = window_labels(meta, start, start + self.table.window_seconds,
                               self.table.min_overlap_seconds, self.table.min_overlap_fraction)
        if labels["ambiguous"]:
            start = nominal
            labels = window_labels(meta, start, start + self.table.window_seconds,
                                   self.table.min_overlap_seconds, self.table.min_overlap_fraction)

        path = str(self.root / meta.file)
        frames, _ = read_frames(path, self.visual_fps, image_size=self.image_size,
                                face_crop=self.face_crop, start=start,
                                duration=self.table.window_seconds)
        count = max(1, int(round(self.table.window_seconds * self.visual_fps)))
        if frames.shape[0] < count:
            frames = np.concatenate([frames, np.repeat(frames[-1:], count - frames.shape[0], axis=0)])
        frames = frames[:count]

        samples = int(round(self.table.window_seconds * self.sample_rate))
        wave = read_audio(path, self.sample_rate, start=start, duration=self.table.window_seconds)
        if wave.shape[0] < samples:
            wave = np.pad(wave, (0, samples - wave.shape[0]))
        wave = wave[:samples]

        item = {
            "frames": torch.from_numpy(frames),            # uint8 [T, S, S, 3]
            "waveform": torch.from_numpy(wave.astype(np.float32)),
            "start": torch.tensor(start),
            "file": meta.file,
        }
        item.update(labels_to_tensors(labels))
        item.update(step_target_tensors(meta, start, self.table.window_seconds, self.visual_fps, self.audio_hz))
        return item
