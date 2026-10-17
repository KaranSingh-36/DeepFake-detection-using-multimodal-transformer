"""Video and audio decoding shared by caching, training and inference."""

import shutil
import subprocess

import cv2
import numpy as np

from fusiontrace.faces import crop_and_resize, stable_face_box, stable_mouth_box

VISUAL_STREAMS = ("face", "mouth")

_FFMPEG = None


def ffmpeg_exe():
    """ffmpeg on PATH, else the binary bundled with imageio-ffmpeg."""
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = shutil.which("ffmpeg")
        if _FFMPEG is None:
            try:
                import imageio_ffmpeg

                _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "ffmpeg was not found on PATH. Install ffmpeg or "
                    "`pip install imageio-ffmpeg`."
                ) from exc
    return _FFMPEG


def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or count <= 0:
        raise RuntimeError(f"Video has no readable frames: {path}")
    return {"fps": fps, "frames": count, "width": width, "height": height,
            "duration": count / fps}


def read_audio(path, sample_rate=16000, start=None, duration=None):
    """Mono float32 waveform in [-1, 1]. Returns an empty array if no audio."""
    command = [ffmpeg_exe(), "-v", "error", "-nostdin"]
    if start is not None and start > 0:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    command += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="ignore").strip()
        if "does not contain any stream" in message or "Output file is empty" in message:
            return np.zeros(0, dtype=np.float32)
        raise RuntimeError(f"ffmpeg failed on {path}: {message}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _target_frame_indices(src_fps, src_count, target_fps, start=0.0, duration=None):
    """Frame indices whose timestamps are nearest to (k + 0.5) / target_fps."""
    total = src_count / src_fps
    if duration is None:
        duration = total - start
    n = max(1, int(round(duration * target_fps)))
    times = start + (np.arange(n) + 0.5) / target_fps
    indices = np.clip(np.round(times * src_fps).astype(np.int64), 0, src_count - 1)
    return indices, times


def _decode_frames(path, target_fps, start=0.0, duration=None, max_side=720):
    """Decode frames at `target_fps` -> (list of HxWx3 uint8 RGB, times [N]).

    Frames are decoded sequentially once; only the wanted indices are
    converted. Large frames are downscaled to `max_side`.
    """
    info = probe_video(path)
    indices, times = _target_frame_indices(
        info["fps"], info["frames"], target_fps, start=start, duration=duration
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    first = int(indices[0])
    position = 0
    if first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if position < 0 or position > first:
            cap.release()
            cap = cv2.VideoCapture(str(path))
            position = 0

    wanted = {}
    for k, idx in enumerate(indices):
        wanted.setdefault(int(idx), []).append(k)

    frames = [None] * len(indices)
    last_good = None
    remaining = len(indices)

    while remaining > 0:
        if not cap.grab():
            break
        if position in wanted:
            ok, bgr = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                scale = max_side / float(max(h, w))
                if scale < 1.0:
                    rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                last_good = rgb
            for k in wanted[position]:
                frames[k] = last_good
                remaining -= 1
        position += 1
        if position > int(indices[-1]):
            break
    cap.release()

    # Fill gaps (decoder stopped early or a frame failed) with the nearest decoded frame.
    for k in range(len(frames)):
        if frames[k] is None and last_good is not None:
            frames[k] = last_good
    if frames[0] is None:
        raise RuntimeError(f"No frames could be decoded from {path}")
    for k in range(1, len(frames)):
        if frames[k] is None:
            frames[k] = frames[k - 1]
    return frames, times.astype(np.float32)


def read_frames(path, target_fps, image_size=224, face_crop=True, start=0.0,
                duration=None, max_side=720):
    """Face-cropped frames at `target_fps` -> (uint8 [N, S, S, 3], times [N])."""
    frames, times = _decode_frames(path, target_fps, start, duration, max_side)
    box = stable_face_box(frames) if face_crop else None
    return crop_and_resize(frames, box, image_size), times


def read_frame_streams(path, target_fps, image_size=224, face_crop=True, streams=VISUAL_STREAMS,
                       start=0.0, duration=None, max_side=720):
    """Decode once and return every requested visual stream.

    -> ({"face": uint8 [N, S, S, 3], "mouth": uint8 [N, S, S, 3]}, times [N])

    "face"  : the frame, cropped to the face when one is found (or as-is).
    "mouth" : the lower-face region of the face stream (see faces.stable_mouth_box),
              resized to the same size, so the encoder sees lip-sync artefacts at
              full resolution.
    """
    frames, times = _decode_frames(path, target_fps, start, duration, max_side)
    box = stable_face_box(frames) if face_crop else None
    face = crop_and_resize(frames, box, image_size)
    out = {}
    for name in streams:
        if name == "face":
            out[name] = face
        elif name == "mouth":
            mouth_box = stable_mouth_box(list(face))
            out[name] = crop_and_resize(face, mouth_box, image_size)
        else:
            raise ValueError(f"unknown visual stream {name!r}; choose from {VISUAL_STREAMS}")
    return out, times
