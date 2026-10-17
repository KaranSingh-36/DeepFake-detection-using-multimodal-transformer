"""Train the window-level fusion model.

    python scripts/train.py                                   # cross-attention on the cache
    python scripts/train.py --set train.fusion=concat          # ablation
    python scripts/train.py --set train.fusion=audio_only
    python scripts/train.py --set train.source=raw --set train.trainable_video_blocks=2 \
                            --set train.trainable_audio_blocks=2 --set train.raw_batch_size=2

Outputs go to results/runs/<name>/ (config.yaml, history.json, last.pt, best.pt)
and the best checkpoint is also copied to results/runs/latest/best.pt.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fusiontrace import PROJECT_ROOT
from fusiontrace.cache import FeatureCache
from fusiontrace.config import load_config, save_config
from fusiontrace.data import CachedWindowDataset, RawWindowDataset, WindowTable, make_balanced_sampler
from fusiontrace.evaluation import localisation, score_videos, summarize
from fusiontrace.features import build_encoders, cache_manifest
from fusiontrace.features import visual_streams_of
from fusiontrace.model import EndToEndModel, WindowFusionModel, apply_modality_dropout, multitask_loss
from fusiontrace.paths import project_path, resolve_dataset_root
from fusiontrace.windows import load_metadata


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(metas, limit, seed):
    if limit is None or limit >= len(metas):
        return metas
    rng = random.Random(seed)
    return rng.sample(metas, int(limit))


def amp_enabled(cfg, device):
    setting = cfg.train.amp
    if setting == "auto" or setting is None:
        return device.type == "cuda"
    return bool(setting) and device.type == "cuda"


def lr_lambda(total_steps, warmup_fraction):
    warmup = max(1, int(total_steps * warmup_fraction))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def build_table(metas, cfg, stride):
    return WindowTable(metas, cfg.window.seconds, stride, cfg.window.min_overlap_seconds,
                       cfg.window.min_overlap_fraction, drop_ambiguous=True)


def checkpoint_payload(cfg, manifest, model_kwargs, fusion_model, encoders, epoch, metrics, threshold, step_threshold):
    encoder_state = None
    if encoders is not None and encoders.is_trainable:
        encoder_state = {k: v.detach().cpu() for k, v in encoders.state_dict().items()
                         if any(k.startswith(n) for n, p in encoders.named_parameters() if p.requires_grad)}
    return {
        "config": cfg.to_plain(),
        "manifest": manifest,
        "model_kwargs": model_kwargs,
        "fusion_state_dict": {k: v.detach().cpu() for k, v in fusion_model.state_dict().items()},
        "encoders_state_dict": encoder_state,
        "epoch": epoch,
        "metrics": metrics,
        "threshold": float(threshold),
        "step_threshold": float(step_threshold),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def select_metric(report, key):
    """Value of `key` in the dev report; dotted keys reach nested entries,
    e.g. "video_auc", "window_level.fused_auc", "localisation.ap@0.5"."""
    parts = str(key).split(".")
    node, i = report, 0
    while i < len(parts):
        for j in range(len(parts), i, -1):            # longest key first ("ap@0.5" contains a dot)
            candidate = ".".join(parts[i:j])
            if isinstance(node, dict) and candidate in node:
                node, i = node[candidate], j
                break
        else:
            return float(report["video_level"]["auc"])
    return float(node) if node is not None else -1.0


def update_latest(run_dir, name, metric_name, metric_value, fusion, log):
    """results/runs/latest/best.pt = the best checkpoint across runs.

    Only cross-attention / concat runs (the models that use both modalities)
    replace it, and only when their selection metric is at least as good as the
    one recorded in latest/source.json. Unimodal ablations never become the
    dashboard default.
    """
    latest = PROJECT_ROOT / "results" / "runs" / "latest"
    info_path = latest / "source.json"
    if fusion in ("video_only", "audio_only"):
        return
    if info_path.is_file():
        try:
            previous = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("run") != name and previous.get("metric_name") == metric_name \
                and float(previous.get("metric_value", -1.0)) > float(metric_value):
            log(f"  results/runs/latest keeps run '{previous.get('run')}' ({metric_name}={previous.get('metric_value'):.4f})")
            return
    latest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "best.pt", latest / "best.pt")
    info_path.write_text(json.dumps({"run": name, "run_dir": str(run_dir), "fusion": fusion,
                                     "metric_name": metric_name, "metric_value": float(metric_value),
                                     "saved_at": datetime.now().isoformat(timespec="seconds")}, indent=2),
                         encoding="utf-8")
    log(f"  results/runs/latest/best.pt now points at run '{name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--name", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="path to last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.metadata:
        cfg.data.metadata_csv = args.metadata
    if args.dataset_root:
        cfg.data.dataset_root = args.dataset_root
    if args.name:
        cfg.train.name = args.name

    seed_everything(int(cfg.train.seed))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = amp_enabled(cfg, device)
    source = cfg.train.source
    if source not in ("cache", "raw"):
        raise ValueError("train.source must be 'cache' or 'raw'")

    name = cfg.train.name or f"{cfg.train.fusion}_{source}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = PROJECT_ROOT / "results" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(*parts):
        text = " ".join(str(p) for p in parts)
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    log("=" * 72)
    log(f"FUSIONTRACE TRAINING  run={name}  source={source}  fusion={cfg.train.fusion}")
    log("=" * 72)
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "") + f" | AMP: {use_amp}")

    metadata_csv = project_path(cfg.data.metadata_csv)
    cache = FeatureCache(project_path(cfg.data.cache_dir))
    train_metas = load_metadata(metadata_csv, split="train")
    dev_metas = load_metadata(metadata_csv, split="dev")

    encoders = None
    if source == "cache":
        manifest = cache.require_manifest()
        cached_train = [m for m in train_metas if cache.has(m.file)]
        cached_dev = [m for m in dev_metas if cache.has(m.file)]
        log(f"Cache: {cache.root}")
        log(f"Train clips cached: {len(cached_train):,}/{len(train_metas):,} | dev cached: {len(cached_dev):,}/{len(dev_metas):,}")
        if not cached_train or not cached_dev:
            raise RuntimeError("Need cached train and dev clips. Run scripts/build_cache.py first.")
        train_metas, dev_metas = cached_train, cached_dev
    else:
        root = resolve_dataset_root(cfg.data.dataset_root)
        encoders = build_encoders(cfg, device, cfg.train.trainable_video_blocks,
                                  cfg.train.trainable_audio_blocks, cfg.train.gradient_checkpointing)
        manifest = cache_manifest(cfg, encoders)
        dev_cached = [m for m in dev_metas if cache.has(m.file)]
        if cache.read_manifest() is None or not dev_cached:
            raise RuntimeError("Raw training still needs cached dev clips for validation "
                               "(scripts/build_cache.py --splits dev --max-videos 2000).")
        dev_metas = dev_cached
        log(f"Dataset root: {root} | trainable blocks: video={cfg.train.trainable_video_blocks} audio={cfg.train.trainable_audio_blocks}")

    train_metas = subset(train_metas, cfg.train.max_train_videos, cfg.train.seed)
    dev_metas = subset(dev_metas, cfg.train.max_dev_videos, cfg.train.seed)

    table = build_table(train_metas, cfg, cfg.window.train_stride_seconds)
    log(f"Training clips: {len(train_metas):,} -> {len(table):,} candidate windows {table.class_counts()}")
    log(f"Dev clips scored each epoch: {len(dev_metas):,}")

    if source == "cache":
        dataset = CachedWindowDataset(table, cache, cfg.window.jitter_seconds, cfg.train.seed)
        batch_size = int(cfg.train.batch_size)
    else:
        dataset = RawWindowDataset(table, root, cfg.data.visual_fps, cfg.data.image_size,
                                   cfg.data.face_crop, cfg.data.sample_rate, cfg.window.jitter_seconds,
                                   cfg.train.seed, audio_hz=encoders.audio_hz,
                                   visual_streams=visual_streams_of(cfg))
        batch_size = int(cfg.train.raw_batch_size)

    windows_per_epoch = min(int(cfg.train.windows_per_epoch), max(len(table), 1) * 4)
    sampler = make_balanced_sampler(table, windows_per_epoch, cfg.train.seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=int(cfg.train.num_workers),
                        pin_memory=device.type == "cuda", drop_last=False,
                        persistent_workers=int(cfg.train.num_workers) > 0)

    model_kwargs = {
        "visual_dim": int(manifest["visual_dim"]),
        "audio_dim": int(manifest["audio_dim"]),
        "n_audio_layers": len(manifest["audio_layers"]),
        "dim": int(cfg.model.dim),
        "heads": int(cfg.model.heads),
        "temporal_layers": int(cfg.model.temporal_layers),
        "dropout": float(cfg.model.dropout),
        "fusion": cfg.train.fusion,
    }
    fusion_model = WindowFusionModel(**model_kwargs).to(device)
    model = fusion_model if source == "cache" else EndToEndModel(encoders, fusion_model).to(device)
    n_params = sum(p.numel() for p in fusion_model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Fusion model params: {n_params:,} | trainable (incl. encoders): {n_trainable:,}")

    param_groups = [{"params": [p for p in fusion_model.parameters() if p.requires_grad], "lr": float(cfg.train.lr)}]
    if encoders is not None and encoders.is_trainable:
        param_groups.append({"params": [p for p in encoders.parameters() if p.requires_grad],
                             "lr": float(cfg.train.raw_lr_encoders)})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.train.weight_decay))
    steps_per_epoch = math.ceil(windows_per_epoch / batch_size)
    total_steps = steps_per_epoch * int(cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(total_steps, float(cfg.train.warmup_fraction)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_metric = -float("inf")
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        fusion_model.load_state_dict(state["fusion_state_dict"])
        if state.get("encoders_state_dict") and encoders is not None:
            encoders.load_state_dict(state["encoders_state_dict"], strict=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = state.get("history", [])
        best_metric = state.get("best_metric", best_metric)
        start_epoch = int(state["epoch"])
        log(f"Resumed from {args.resume} at epoch {start_epoch}")

    loss_weights = dict(cfg.train.loss_weights)
    modality_dropout = float(cfg.train.get("modality_dropout", 0.0))
    if cfg.train.fusion in ("video_only", "audio_only"):
        modality_dropout = 0.0          # a unimodal model has nothing to fall back on
    dropout_generator = torch.Generator().manual_seed(int(cfg.train.seed) + 1)
    log(f"Modality dropout: {modality_dropout:.2f} (share of windows that lose audio or video during training)")
    log_every = max(1, steps_per_epoch // 10)

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        model.train()
        epoch_start = time.perf_counter()
        running = {"total": 0.0, "video": 0.0, "audio": 0.0, "fused": 0.0, "steps": 0.0}
        counts = {"total": 0, "video": 0, "audio": 0, "fused": 0, "steps": 0}
        for step, batch in enumerate(loader, start=1):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if source == "cache":
                    if modality_dropout > 0.0:
                        batch = apply_modality_dropout(batch, modality_dropout, dropout_generator)
                    outputs = model(batch["visual"], batch["audio"], batch["visual_mask"], batch["audio_mask"])
                else:
                    outputs = model(batch["frames"], batch["waveform"])
                loss, parts = multitask_loss(outputs, batch, loss_weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], float(cfg.train.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["total"] += float(loss.detach()); counts["total"] += 1
            for key, value in parts.items():
                running[key] += float(value); counts[key] += 1
            if step % log_every == 0 or step == steps_per_epoch:
                parts_text = " ".join(f"{k}={running[k] / max(counts[k], 1):.4f}" for k in ("video", "audio", "fused", "steps") if counts[k])
                log(f"epoch {epoch + 1} step {step}/{steps_per_epoch} loss={running['total'] / counts['total']:.4f} {parts_text} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} {(time.perf_counter() - epoch_start) / 60:.1f} min")

        train_seconds = time.perf_counter() - epoch_start
        log(f"epoch {epoch + 1}: scoring {len(dev_metas):,} dev clips ...")
        eval_start = time.perf_counter()
        window_frame, records = score_videos(fusion_model, cache, dev_metas, cfg, device,
                                             batch_windows=int(cfg.eval.batch_windows), amp=use_amp,
                                             progress_every=0, log=log)
        report = summarize(window_frame, records, threshold=None)
        report["localisation"] = localisation(records, cfg, step_threshold=None)
        vl = report["video_level"]
        per_type = {k: round(v["flagged_rate"], 3) for k, v in report["per_fake_type"].items()}
        log(f"epoch {epoch + 1} DEV | video AUC {vl['auc']:.4f} | AP {vl['average_precision']:.4f} | "
            f"bal.acc {vl['balanced_accuracy']:.4f} @thr {vl['threshold']:.3f} | FPR {vl['false_positive_rate']:.4f} | "
            f"flagged by type {per_type}")
        wl = report["window_level"]
        loc = report["localisation"]
        log(f"            window AUC fused {wl['fused_auc']:.4f} | video head {wl['video_head_auc']} | audio head {wl['audio_head_auc']}")
        log(f"            localisation: frame AUC {loc['frame_level_auc']:.4f} | "
            + " | ".join(f"AP@{iou} {loc[f'ap@{iou}']:.4f}" for iou in cfg.eval.iou_thresholds)
            + f" | step thr {loc['step_threshold']:.3f} | eval {(time.perf_counter() - eval_start) / 60:.1f} min")

        record = {
            "epoch": epoch + 1,
            "train_loss": running["total"] / max(counts["total"], 1),
            "train_loss_parts": {k: running[k] / max(counts[k], 1) for k in ("video", "audio", "fused", "steps") if counts[k]},
            "train_seconds": train_seconds,
            "dev": report,
        }
        history.append(record)
        with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, default=float)

        metric_value = select_metric(report, cfg.train.select_by)
        if isinstance(metric_value, float) and math.isnan(metric_value):
            metric_value = -1.0
        payload = checkpoint_payload(cfg, manifest, model_kwargs, fusion_model, encoders, epoch + 1, report,
                                     report["threshold"], report["localisation"]["step_threshold"])
        payload.update({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "history": history, "best_metric": max(best_metric, metric_value)})
        torch.save(payload, run_dir / "last.pt")
        if metric_value > best_metric:
            best_metric = metric_value
            best = {k: v for k, v in payload.items() if k not in ("optimizer_state_dict", "scheduler_state_dict")}
            torch.save(best, run_dir / "best.pt")
            log(f"  new best {cfg.train.select_by}={metric_value:.4f} -> {run_dir / 'best.pt'}")
            if bool(cfg.train.get("update_latest", True)):
                update_latest(run_dir, name, cfg.train.select_by, metric_value, cfg.train.fusion, log)

    log(f"Done. Best {cfg.train.select_by}: {best_metric:.4f}. Run directory: {run_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
