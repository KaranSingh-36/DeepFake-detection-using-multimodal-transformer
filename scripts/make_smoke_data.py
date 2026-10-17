"""Generate a tiny synthetic LAV-DF-style dataset to exercise the whole pipeline.

Each "real" clip is a cartoon talking head: a face whose mouth opens with the
amplitude of a tone sequence. Fakes copy a real clip (the `original`) and inside
one or two short periods:
    video manipulation -> the mouth follows a different envelope and the face is blurred
    audio manipulation -> the tone is replaced by a different one with added noise

    python scripts/make_smoke_data.py --out data/smoke --train 24 --dev 8 --test 8
"""

import _bootstrap  # noqa: F401

import argparse
import json
import math
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fusiontrace.media import ffmpeg_exe

FPS = 25
SIZE = 96
SAMPLE_RATE = 16000


def tone_sequence(rng, duration, base_freq):
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    syllable = 0.5 + 0.5 * np.sin(2 * math.pi * rng.uniform(2.5, 4.5) * t + rng.uniform(0, 6))
    envelope = np.clip(syllable, 0, 1) ** 1.5
    notes = np.repeat(rng.uniform(0.8, 1.25, size=int(duration / 0.3) + 1), int(0.3 * SAMPLE_RATE))[:n]
    freq = base_freq * notes
    phase = np.cumsum(2 * math.pi * freq / SAMPLE_RATE)
    audio = 0.4 * envelope * (np.sin(phase) + 0.3 * np.sin(2 * phase))
    return audio.astype(np.float32), envelope.astype(np.float32)


def render_frames(envelope, duration, colors, rng, mouth_override=None, blur_mask=None):
    n_frames = int(round(duration * FPS))
    frames = np.empty((n_frames, SIZE, SIZE, 3), dtype=np.uint8)
    bg, skin, mouth = colors
    cx, cy = SIZE // 2 + int(rng.integers(-6, 6)), SIZE // 2 + int(rng.integers(-6, 6))
    for i in range(n_frames):
        t = i / FPS
        sample = min(len(envelope) - 1, int(t * SAMPLE_RATE))
        opening = envelope[sample]
        if mouth_override is not None and mouth_override[i] is not None:
            opening = mouth_override[i]
        frame = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
        cv2.ellipse(frame, (cx, cy), (28, 34), 0, 0, 360, skin, -1)
        cv2.circle(frame, (cx - 10, cy - 10), 3, (20, 20, 20), -1)
        cv2.circle(frame, (cx + 10, cy - 10), 3, (20, 20, 20), -1)
        height = max(1, int(2 + 12 * opening))
        cv2.ellipse(frame, (cx, cy + 14), (10, height), 0, 0, 360, mouth, -1)
        if blur_mask is not None and blur_mask[i]:
            frame = cv2.GaussianBlur(frame, (7, 7), 0)
            frame = np.clip(frame.astype(np.int16) + np.array([12, -8, 6]), 0, 255).astype(np.uint8)
        frames[i] = frame
    return frames


def write_clip(path, frames, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        command = [
            ffmpeg_exe(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "pipe:0",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-shortest", str(path),
        ]
        subprocess.run(command, input=frames.tobytes(), check=True)


def make_periods(rng, duration, count):
    periods = []
    for _ in range(count):
        length = float(rng.uniform(0.3, 1.0))
        for _ in range(20):
            start = float(rng.uniform(0.2, duration - length - 0.2))
            if all(start + length < s - 0.3 or start > e + 0.3 for s, e in periods):
                periods.append((round(start, 3), round(start + length, 3)))
                break
    return sorted(periods)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/smoke")
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--dev", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    rows = []
    counter = 0
    for split, count in (("train", args.train), ("dev", args.dev), ("test", args.test)):
        n_real = max(1, count // 4)
        n_fake = count - n_real
        kinds = (["video", "audio", "both"] * (n_fake // 3 + 1))[:n_fake]
        originals = []
        for _ in range(n_real):
            duration = float(rng.uniform(3.0, 5.5))
            audio, envelope = tone_sequence(rng, duration, float(rng.uniform(150, 260)))
            colors = (tuple(int(c) for c in rng.integers(40, 200, 3)),
                      tuple(int(c) for c in rng.integers(150, 240, 3)),
                      tuple(int(c) for c in rng.integers(0, 90, 3)))
            frames = render_frames(envelope, duration, colors, rng)
            name = f"{split}/{counter:06d}.mp4"
            counter += 1
            write_clip(out / name, frames, audio)
            originals.append((name, duration, audio, envelope, colors))
            rows.append(dict(file=name, split=split, label=0, modify_video=0, modify_audio=0, n_fakes=0,
                             duration=round(duration, 3), video_frames=len(frames), audio_channels=1,
                             audio_frames=len(audio), original=None, fake_periods=[]))

        for kind in kinds:
            src_name, duration, audio, envelope, colors = originals[int(rng.integers(len(originals)))]
            periods = make_periods(rng, duration, int(rng.integers(1, 3)))
            fake_audio = audio.copy()
            mouth_override = [None] * int(round(duration * FPS))
            blur_mask = [False] * len(mouth_override)
            for start, end in periods:
                s_sample, e_sample = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
                s_frame, e_frame = int(start * FPS), int(end * FPS)
                if kind in ("audio", "both"):
                    replacement, _ = tone_sequence(rng, end - start + 0.1, float(rng.uniform(380, 600)))
                    replacement = replacement[: e_sample - s_sample]
                    noise = rng.normal(0, 0.03, size=len(replacement)).astype(np.float32)
                    fake_audio[s_sample:s_sample + len(replacement)] = replacement + noise
                if kind in ("video", "both"):
                    other = 0.5 + 0.5 * np.sin(np.arange(e_frame - s_frame) * rng.uniform(0.4, 1.2) + rng.uniform(0, 6))
                    for i, value in zip(range(s_frame, e_frame), other):
                        mouth_override[i] = float(value)
                        blur_mask[i] = True
            frames = render_frames(envelope, duration, colors, rng, mouth_override, blur_mask)
            name = f"{split}/{counter:06d}.mp4"
            counter += 1
            write_clip(out / name, frames, fake_audio)
            rows.append(dict(file=name, split=split, label=1, modify_video=int(kind in ("video", "both")),
                             modify_audio=int(kind in ("audio", "both")), n_fakes=len(periods),
                             duration=round(duration, 3), video_frames=len(frames), audio_channels=1,
                             audio_frames=len(fake_audio), original=src_name,
                             fake_periods=[list(p) for p in periods]))

    frame = pd.DataFrame(rows)
    frame["fake_periods"] = frame["fake_periods"].apply(json.dumps)
    frame.to_csv(out / "metadata.csv", index=False)
    print(f"Wrote {len(frame)} clips to {out} ({frame.groupby(['split', 'label']).size().to_dict()})")
    print(f"Metadata: {out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
