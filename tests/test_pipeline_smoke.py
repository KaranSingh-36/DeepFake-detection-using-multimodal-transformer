"""End-to-end check on synthetic clips: data -> cache -> train -> evaluate -> analyze.

Needs ffmpeg and downloads the two pretrained encoders the first time, so it is
skipped unless FUSIONTRACE_SMOKE=1.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

pytestmark = pytest.mark.skipif(os.environ.get("FUSIONTRACE_SMOKE") != "1",
                                reason="set FUSIONTRACE_SMOKE=1 to run the slow pipeline test")


def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"command failed: {args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result.stdout


def test_pipeline(tmp_path):
    data = tmp_path / "smoke"
    cache = tmp_path / "features"
    run(SCRIPTS / "make_smoke_data.py", "--out", data, "--train", "12", "--dev", "4", "--test", "4")
    assert (data / "metadata.csv").is_file()

    common = ["--set", f"data.cache_dir={cache}", "--metadata", data / "metadata.csv"]
    out = run(SCRIPTS / "build_cache.py", *common, "--dataset-root", data, "--workers", "0",
              "--set", "data.visual_fps=5")
    assert "failed 0" in out or "0 failed" in out
    assert (cache / "manifest.json").is_file()

    out = run(SCRIPTS / "train.py", *common, "--name", "pytest_smoke", "--set", "train.epochs=2",
              "--set", "train.windows_per_epoch=64", "--set", "train.batch_size=8",
              "--set", "train.num_workers=0", "--set", "train.max_dev_videos=4")
    assert "DEV | video AUC" in out
    best = ROOT / "results" / "runs" / "pytest_smoke" / "best.pt"
    assert best.is_file()

    out = run(SCRIPTS / "evaluate.py", "--checkpoint", best, "--split", "test", "--swap",
              "--metadata", data / "metadata.csv", "--cache-dir", cache, "--out", tmp_path / "eval")
    metrics = json.loads((tmp_path / "eval" / "metrics.json").read_text())
    assert "video_level" in metrics and "localisation" in metrics and "modality_swap" in metrics

    clip = next((data / "test").glob("*.mp4"))
    out = run(SCRIPTS / "analyze_video.py", clip, "--checkpoint", best, "--json", tmp_path / "r.json",
              "--plot", tmp_path / "r.png")
    result = json.loads((tmp_path / "r.json").read_text())
    assert result["label"] in ("REAL", "DEEPFAKE") and len(result["windows"]) >= 1
    assert (tmp_path / "r.png").is_file()
