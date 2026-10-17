"""Evaluate a checkpoint on a split of the feature cache.

    python scripts/evaluate.py --checkpoint results/runs/latest/best.pt --split test
    python scripts/evaluate.py --checkpoint ... --split test --swap        # modality swap check
    python scripts/evaluate.py --checkpoint ... --split dev --max-videos 2000 --tune-threshold

Writes metrics.json, video_predictions.csv, window_predictions.csv and
modality_swap.csv to results/eval/<run>_<split>/.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import random
from pathlib import Path

import torch

from fusiontrace import PROJECT_ROOT
from fusiontrace.cache import FeatureCache
from fusiontrace.evaluation import localisation, modality_swap, score_videos, summarize, video_frame
from fusiontrace.inference import load_checkpoint
from fusiontrace.paths import project_path
from fusiontrace.windows import load_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "results" / "runs" / "latest" / "best.pt"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tune-threshold", action="store_true", help="pick the threshold on this split instead of the dev one")
    parser.add_argument("--swap", action="store_true", help="run the modality swap experiment")
    parser.add_argument("--swap-per-type", type=int, default=200)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint, cfg, model = load_checkpoint(args.checkpoint, device)
    if args.metadata:
        cfg.data.metadata_csv = args.metadata
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir
    split = args.split or cfg.eval.split
    cache = FeatureCache(project_path(cfg.data.cache_dir))
    cache.require_manifest()

    run_name = Path(args.checkpoint).resolve().parent.name
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "results" / "eval" / f"{run_name}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = load_metadata(project_path(cfg.data.metadata_csv), split=split)
    cached = [m for m in metas if cache.has(m.file)]
    if args.max_videos and args.max_videos < len(cached):
        cached = random.Random(args.seed).sample(cached, args.max_videos)
    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')}, fusion={cfg.train.fusion})")
    print(f"Split {split}: {len(cached):,} cached clips of {len(metas):,}")
    print(f"Device: {device}")

    window_frame, records = score_videos(model, cache, cached, cfg, device,
                                         batch_windows=int(cfg.eval.batch_windows), amp=device.type == "cuda")
    threshold = None if args.tune_threshold else float(checkpoint.get("threshold", 0.5))
    report = summarize(window_frame, records, threshold=threshold)
    report["localisation"] = localisation(records, cfg, step_threshold=None if args.tune_threshold
                                          else checkpoint.get("step_threshold"))
    report["checkpoint"] = str(args.checkpoint)
    report["split"] = split
    report["n_clips"] = len(cached)

    videos = video_frame(records)
    videos.to_csv(out_dir / "video_predictions.csv", index=False)
    window_frame.to_csv(out_dir / "window_predictions.csv", index=False)

    if args.swap:
        swap = modality_swap(model, cache, cached, cfg, device, max_per_type=args.swap_per_type, seed=args.seed)
        swap.to_csv(out_dir / "modality_swap.csv", index=False)
        report["modality_swap"] = swap.to_dict(orient="records")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)

    vl = report["video_level"]
    print()
    print("VIDEO LEVEL")
    print(f"  AUC {vl['auc']:.4f} | AP {vl['average_precision']:.4f} | accuracy {vl['accuracy']:.4f} | "
          f"balanced accuracy {vl['balanced_accuracy']:.4f} | threshold {vl['threshold']:.3f} ({vl['threshold_source']})")
    print(f"  precision {vl['precision']:.4f} | recall {vl['recall']:.4f} | F1 {vl['f1']:.4f} | FPR {vl['false_positive_rate']:.4f} "
          f"| TN {vl['tn']} FP {vl['fp']} FN {vl['fn']} TP {vl['tp']}")
    print("PER FAKE TYPE (share flagged as fake)")
    for name, row in report["per_fake_type"].items():
        attribution = report["modality_attribution"][name]
        print(f"  {name:12s} n={row['n']:6d} flagged {row['flagged_rate']:.3f} | mean fused {row['mean_score']:.3f} "
              f"| video head {attribution['video_head_mean_score']} | audio head {attribution['audio_head_mean_score']}")
    wl = report["window_level"]
    print("WINDOW LEVEL")
    print(f"  fused AUC {wl['fused_auc']:.4f} | video head AUC {wl['video_head_auc']} | audio head AUC {wl['audio_head_auc']} "
          f"| windows {wl['n_windows']:,} (positive rate {wl['positive_rate']:.3f})")
    loc = report["localisation"]
    print("LOCALISATION")
    print("  " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in loc.items()))
    if args.swap:
        print("MODALITY SWAP")
        print(swap.to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
