"""Run the frozen encoders once over every clip and store the features.

    python scripts/build_cache.py --splits train dev test
    python scripts/build_cache.py --splits train --max-videos 25000 --seed 42
    python scripts/build_cache.py --config configs/smoke.yaml --metadata data/smoke/metadata.csv --dataset-root data/smoke

Re-running skips clips that are already cached, so it can be interrupted.
"""

import _bootstrap  # noqa: F401

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from fusiontrace.cache import FeatureCache
from fusiontrace.config import load_config
from fusiontrace.features import build_encoders, cache_manifest, decode_clip, encode_clip
from fusiontrace.paths import project_path, resolve_dataset_root
from fusiontrace.windows import load_metadata


class ClipDecoder(Dataset):
    def __init__(self, metas, root, cfg):
        self.metas, self.root, self.cfg = metas, root, cfg

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        try:
            clip = decode_clip(str(self.root / meta.file), self.cfg.data.visual_fps,
                               self.cfg.data.image_size, self.cfg.data.face_crop,
                               self.cfg.data.sample_rate)
            return meta.file, clip, None
        except Exception as exc:  # decoding problems must not kill the run
            return meta.file, None, f"{type(exc).__name__}: {exc}"


def select_subset(metas, max_videos, seed):
    if max_videos is None or max_videos >= len(metas):
        return metas
    rng = random.Random(seed)
    return rng.sample(metas, max_videos)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max-videos", type=int, default=None, help="random subset per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metadata", default=None, help="override data.metadata_csv")
    parser.add_argument("--dataset-root", default=None, help="override data.dataset_root")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.metadata:
        cfg.data.metadata_csv = args.metadata
    if args.dataset_root:
        cfg.data.dataset_root = args.dataset_root

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = resolve_dataset_root(cfg.data.dataset_root)
    cache = FeatureCache(project_path(cfg.data.cache_dir))
    metadata_csv = project_path(cfg.data.metadata_csv)

    print("=" * 72)
    print("FUSIONTRACE FEATURE CACHE")
    print("=" * 72)
    print(f"Dataset root : {root}")
    print(f"Metadata     : {metadata_csv}")
    print(f"Cache dir    : {cache.root}")
    print(f"Device       : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Visual fps   : {cfg.data.visual_fps} | audio layers {list(cfg.data.audio_layers)} @ {50 / cfg.data.audio_pool:.0f} Hz")

    encoders = build_encoders(cfg, device).eval()
    cache.write_manifest(cache_manifest(cfg, encoders))

    error_log = cache.root / "errors.log"
    overall_start = time.perf_counter()
    total_done = 0

    for split in args.splits:
        metas = load_metadata(metadata_csv, split=split)
        metas = select_subset(metas, args.max_videos, args.seed)
        todo = [m for m in metas if not cache.has(m.file)]
        print()
        print(f"[{split}] {len(metas):,} clips selected, {len(metas) - len(todo):,} already cached, {len(todo):,} to do")
        if not todo:
            continue

        loader = DataLoader(ClipDecoder(todo, root, cfg), batch_size=None, shuffle=False,
                            num_workers=args.workers, prefetch_factor=4 if args.workers > 0 else None,
                            persistent_workers=False)
        split_start = time.perf_counter()
        done = failed = 0
        seconds_cached = 0.0
        bytes_written = 0
        for file, clip, error in loader:
            if error is not None:
                failed += 1
                with open(error_log, "a", encoding="utf-8") as handle:
                    handle.write(f"{file}\t{error}\n")
                continue
            try:
                features = encode_clip(encoders, clip, cfg.data.visual_fps, device,
                                       audio_chunk_seconds=cfg.data.audio_chunk_seconds,
                                       audio_hop_seconds=cfg.data.audio_hop_seconds)
                cache.write(file, features["visual"], features["audio"], features["visual_fps"],
                            features["audio_hz"], features["duration"])
            except Exception:
                failed += 1
                with open(error_log, "a", encoding="utf-8") as handle:
                    handle.write(f"{file}\t{traceback.format_exc()}\n")
                continue
            done += 1
            seconds_cached += features["duration"]
            bytes_written += features["visual"].nbytes + features["audio"].nbytes
            if done % 50 == 0 or done == len(todo):
                elapsed = time.perf_counter() - split_start
                rate = done / elapsed
                remaining = (len(todo) - done - failed) / max(rate, 1e-9)
                print(f"  {done:,}/{len(todo):,} clips | {rate * 60:.1f} clips/min | "
                      f"{bytes_written / done / 1e6:.2f} MB/clip | ETA {remaining / 3600:.2f} h | failed {failed}",
                      flush=True)
        total_done += done
        print(f"[{split}] done: {done:,} cached, {failed:,} failed, {seconds_cached / 3600:.2f} h of video, "
              f"{bytes_written / 1e9:.2f} GB written in {(time.perf_counter() - split_start) / 60:.1f} min")

    print()
    print(f"Finished: {total_done:,} clips in {(time.perf_counter() - overall_start) / 60:.1f} min")
    if error_log.is_file():
        print(f"Some clips failed; see {error_log}")


if __name__ == "__main__":
    main()
