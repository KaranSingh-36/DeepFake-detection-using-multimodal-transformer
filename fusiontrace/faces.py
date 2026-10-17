"""Lightweight face localisation for cropping before the visual encoder.

Uses OpenCV's YuNet detector (bundled with OpenCV >= 4.5.4 / 5.x; the 230 KB
ONNX file is downloaded once into ~/.cache/fusiontrace). Faces in a talking
head clip barely move, so one median box over a few frames is applied to every
frame of the clip. Clips that are already tight face crops (LAV-DF is 224x224
face-centred) are detected and left untouched. If no detector can be set up
the frames are used as-is and a warning is printed once.
"""

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

_DETECTOR = None
_DETECTOR_FAILED = False


def model_cache_dir():
    override = os.environ.get("FUSIONTRACE_MODEL_DIR")
    return Path(override) if override else Path.home() / ".cache" / "fusiontrace"


def yunet_path(download=True):
    path = model_cache_dir() / YUNET_FILE
    if path.is_file() or not download:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(YUNET_URL, tmp)
    tmp.replace(path)
    return path


def _detector():
    global _DETECTOR, _DETECTOR_FAILED
    if _DETECTOR is not None or _DETECTOR_FAILED:
        return _DETECTOR
    try:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        _DETECTOR = cv2.FaceDetectorYN.create(str(yunet_path()), "", (320, 320), 0.6, 0.3, 50)
    except Exception as exc:
        _DETECTOR_FAILED = True
        print(f"[fusiontrace] face detector unavailable ({exc}); frames are used without cropping.")
    return _DETECTOR


def detect_largest_face(frame_rgb):
    """-> (x, y, w, h) of the highest-scoring face or None."""
    detector = _detector()
    if detector is None:
        return None
    height, width = frame_rgb.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = faces[np.argmax(faces[:, -1])][:4]
    return float(x), float(y), float(w), float(h)


def stable_face_box(frames, max_probe=5, margin=1.4, already_cropped_fraction=0.45):
    """Median face box over up to `max_probe` frames, expanded to a square.

    Returns (x0, y0, x1, y1) in pixel coordinates or None when the clip should
    be used as-is (no face found, or the face already fills the frame).
    """
    if len(frames) == 0 or _detector() is None:
        return None

    height, width = frames[0].shape[:2]
    probe = np.linspace(0, len(frames) - 1, min(max_probe, len(frames))).astype(int)
    boxes = [detect_largest_face(frames[i]) for i in probe]
    boxes = [b for b in boxes if b is not None]
    if len(boxes) < max(1, len(probe) // 2):
        return None

    x, y, w, h = np.median(np.array(boxes, dtype=np.float32), axis=0)
    if (w * h) / float(width * height) >= already_cropped_fraction:
        return None

    size = max(w, h) * margin
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(round(max(0, cx - size / 2.0)))
    y0 = int(round(max(0, cy - size / 2.0)))
    x1 = int(round(min(width, cx + size / 2.0)))
    y1 = int(round(min(height, cy + size / 2.0)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def crop_and_resize(frames, box, image_size):
    """frames: list/array of HxWx3 uint8 -> uint8 array [N, image_size, image_size, 3]."""
    out = np.empty((len(frames), image_size, image_size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        if box is not None:
            x0, y0, x1, y1 = box
            frame = frame[y0:y1, x0:x1]
        out[i] = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return out
