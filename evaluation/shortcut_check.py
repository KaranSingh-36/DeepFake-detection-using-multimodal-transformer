"""
Check whether the model detects the manipulation itself or a dataset shortcut.

LAV-DF fakes are only fake for a short segment (~0.7 s on average), while the
model sees 8 uniformly sampled frames and the first few seconds of audio. For
many fakes the manipulated segment is therefore never shown to the model.

Part 1 (CSV only, fast): compare the miss rate of fakes whose manipulated
segment is visible in the model input against fakes where it is not.

Part 2 (--swap, needs GPU + dataset): swap one modality of a fake with the
untouched source video ("original" column) and see which stream still makes
the model say DEEPFAKE.

Usage (from the project root):

    python evaluation/shortcut_check.py
    python evaluation/shortcut_check.py --swap --samples 200
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

NUM_FRAMES = 8
AUDIO_DURATION = 4.0


def load_table(metadata_csv, predictions_csv):
    metadata = pd.read_csv(metadata_csv)
    predictions = pd.read_csv(predictions_csv)

    table = predictions.merge(metadata, on="file")
    table["periods"] = table["fake_periods"].apply(ast.literal_eval)

    def frames_in_fake(row):
        indices = np.linspace(0, row.video_frames - 1, NUM_FRAMES).astype(int)
        times = indices * row.duration / row.video_frames
        return sum(
            any(start <= t <= end for start, end in row.periods)
            for t in times
        )

    table["frames_in_fake"] = table.apply(frames_in_fake, axis=1)
    table["audio_in_window"] = table["periods"].apply(
        lambda periods: any(start < AUDIO_DURATION for start, _ in periods)
    )

    video = table["modify_video"] == 1
    audio = table["modify_audio"] == 1

    table["modality"] = np.select(
        [video & audio, video, audio],
        ["video+audio", "video only", "audio only"],
        "real",
    )

    table["visible"] = (
        (video & (table["frames_in_fake"] > 0))
        | (audio & table["audio_in_window"])
    )

    return table


def visibility_report(table):
    fakes = table[table["true_label"] == 1].copy()
    fakes["missed"] = fakes["predicted_label"] == 0

    print("=" * 80)
    print("PART 1: IS THE MANIPULATED SEGMENT VISIBLE IN THE MODEL INPUT?")
    print("=" * 80)

    summary = (
        fakes.groupby(["modality", "visible"])
        .agg(
            samples=("missed", "size"),
            missed=("missed", "sum"),
            miss_rate=("missed", "mean"),
            mean_fake_probability=("deepfake_probability", "mean"),
        )
        .round(4)
    )

    print(summary.to_string())

    overall = fakes.groupby("visible")["missed"].agg(["size", "sum", "mean"])
    hidden = overall.loc[False]
    shown = overall.loc[True]

    print()
    print(
        f"Manipulation visible:     {int(shown['size'])} fakes, "
        f"{100 * (1 - shown['mean']):.2f}% detected"
    )
    print(
        f"Manipulation NOT visible: {int(hidden['size'])} fakes, "
        f"{100 * (1 - hidden['mean']):.2f}% detected"
    )
    print()
    print(
        "If the second number is close to the first, the model is not\n"
        "detecting the manipulated segment - it is using a cue that is\n"
        "present in the whole file."
    )


def swap_experiment(table, args):
    import torch
    from torch.utils.data import DataLoader, Dataset

    from dataset.lavdf_dataset import LAVDFDataset
    from models.multimodal_model import MultimodalCrossAttentionModel

    print()
    print("=" * 80)
    print("PART 2: MODALITY SWAP WITH THE UNTOUCHED SOURCE VIDEO")
    print("=" * 80)

    loader_dataset = LAVDFDataset(
        metadata_csv=args.metadata,
        dataset_root=args.dataset_root,
        split="test",
        num_frames=NUM_FRAMES,
        audio_duration=AUDIO_DURATION,
    )

    candidates = table[
        (table["true_label"] == 1)
        & table["modality"].isin(["audio only", "video only"])
    ]

    pairs = (
        candidates.groupby(["modality", "visible"])
        .sample(args.samples, random_state=args.seed)
        [["file", "original", "modality", "visible"]]
        .reset_index(drop=True)
    )

    class PairDataset(Dataset):

        def __len__(self):
            return len(pairs)

        def __getitem__(self, index):
            row = pairs.iloc[index]
            fake = loader_dataset._get_video_path(row["file"])
            real = loader_dataset._get_video_path(row["original"])

            return {
                "fake_frames": loader_dataset._load_frames(fake),
                "fake_audio": loader_dataset._load_audio(fake),
                "real_frames": loader_dataset._load_frames(real),
                "real_audio": loader_dataset._load_audio(real),
                "index": index,
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalCrossAttentionModel()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.to(device).eval()

    conditions = {
        "source real video (control)": ("real_frames", "real_audio"),
        "fake video as-is": ("fake_frames", "fake_audio"),
        "fake frames + real audio": ("fake_frames", "real_audio"),
        "real frames + fake audio": ("real_frames", "fake_audio"),
    }

    probabilities = {
        name: np.zeros(len(pairs), dtype=np.float32) for name in conditions
    }

    loader = DataLoader(
        PairDataset(),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            for name, (frames_key, audio_key) in conditions.items():
                with torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                ):
                    logits = model(
                        batch[frames_key].to(device),
                        batch[audio_key].to(device),
                    )

                probability = torch.softmax(logits.float(), dim=1)[:, 1]
                probabilities[name][batch["index"].numpy()] = (
                    probability.cpu().numpy()
                )

            if step % 20 == 0:
                print(f"  {step * args.batch_size}/{len(pairs)} pairs")

    print()
    print("Share predicted DEEPFAKE (probability >= 0.5) per condition:")
    print()

    rows = []

    for (modality, visible), group in pairs.groupby(["modality", "visible"]):
        mask = np.zeros(len(pairs), dtype=bool)
        mask[group.index] = True

        for name in conditions:
            values = probabilities[name][mask]
            rows.append({
                "fake type": modality,
                "manipulation in input": "yes" if visible else "no",
                "model input": name,
                "samples": int(mask.sum()),
                "predicted_fake": round(float((values >= 0.5).mean()), 4),
                "mean_probability": round(float(values.mean()), 4),
            })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--metadata",
        default=str(PROJECT_ROOT / "metadata" / "lavdf_metadata.csv"),
    )
    parser.add_argument(
        "--predictions",
        default=str(
            PROJECT_ROOT / "results" / "test"
            / "multimodal_25000_epoch2_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT / "results" / "checkpoints"
            / "multimodal_25000_2epoch_epoch_2.pth"
        ),
    )
    parser.add_argument("--dataset-root", default=r"C:\AI_DATA\LAV-DF\LAV-DF")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    table = load_table(args.metadata, args.predictions)
    visibility_report(table)

    if args.swap:
        swap_experiment(table, args)


if __name__ == "__main__":
    main()
