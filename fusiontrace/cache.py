"""Per-video feature cache: one .npz per clip holding frozen encoder outputs.

    <cache_dir>/manifest.json
    <cache_dir>/<split>/<clip>.npz   visual float16 [Tv, Dv]
                                     audio  float16 [L, Ta, Da]
                                     visual_fps, audio_hz, duration
"""

import json
from pathlib import Path

import numpy as np


class FeatureCache:
    def __init__(self, cache_dir):
        self.root = Path(cache_dir)

    @property
    def manifest_path(self):
        return self.root / "manifest.json"

    def path_for(self, file):
        return self.root / Path(file).with_suffix(".npz")

    def has(self, file):
        return self.path_for(file).is_file()

    def write(self, file, visual, audio, visual_fps, audio_hz, duration):
        path = self.path_for(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            visual=np.ascontiguousarray(visual, dtype=np.float16),
            audio=np.ascontiguousarray(audio, dtype=np.float16),
            visual_fps=np.float32(visual_fps),
            audio_hz=np.float32(audio_hz),
            duration=np.float32(duration),
        )
        tmp.replace(path)

    def read(self, file):
        with np.load(self.path_for(file)) as data:
            return {
                "visual": data["visual"],
                "audio": data["audio"],
                "visual_fps": float(data["visual_fps"]),
                "audio_hz": float(data["audio_hz"]),
                "duration": float(data["duration"]),
            }

    def write_manifest(self, manifest):
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self.read_manifest()
        if existing is not None:
            mismatched = {
                key: (existing.get(key), value)
                for key, value in manifest.items()
                if existing.get(key) != value
            }
            if mismatched:
                raise RuntimeError(
                    f"Cache at {self.root} was built with different settings: "
                    f"{mismatched}. Use another cache_dir or delete it."
                )
            return
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def read_manifest(self):
        if not self.manifest_path.is_file():
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def require_manifest(self):
        manifest = self.read_manifest()
        if manifest is None:
            raise FileNotFoundError(
                f"No feature cache at {self.root}. Run scripts/build_cache.py first."
            )
        return manifest

    def available(self, files):
        return [file for file in files if self.has(file)]
