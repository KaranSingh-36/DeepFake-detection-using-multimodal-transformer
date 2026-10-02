import os
import sys
import time
import multiprocessing

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORTS
# ============================================================

import torch
from torch.utils.data import DataLoader

from dataset.lavdf_dataset import LAVDFDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"C:\AI_DATA\LAV-DF\LAV-DF"

METADATA_CSV = os.path.join(
    PROJECT_ROOT,
    "metadata",
    "lavdf_metadata.csv"
)

BATCH_SIZE = 1

NUM_FRAMES = 8
IMAGE_SIZE = 224

SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0

# Only benchmark a small number of samples.
MAX_BATCHES = 200

# Change this between tests:
NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LAV-DF DATALOADER SPEED BENCHMARK")
    print("=" * 80)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Workers:", NUM_WORKERS)
    print("Batch size:", BATCH_SIZE)
    print("Max batches:", MAX_BATCHES)
    print("Pin memory:", PIN_MEMORY)

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LAVDFDataset(
        metadata_csv=METADATA_CSV,
        dataset_root=DATASET_ROOT,
        split="train",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        sample_rate=SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
    )

    print("Dataset samples:", len(dataset))

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    print()
    print("DataLoader created successfully.")
    print()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    start_time = time.time()

    total_samples = 0

    for batch_index, batch in enumerate(loader):

        total_samples += batch["label"].shape[0]

        if batch_index == 0:
            print("First batch received.")
            print("Frames shape:", batch["frames"].shape)
            print("Audio shape:", batch["audio"].shape)
            print("Labels shape:", batch["label"].shape)
            print()

        if (batch_index + 1) % 25 == 0:

            elapsed = time.time() - start_time

            samples_per_second = (
                total_samples / elapsed
            )

            print(
                f"Batch {batch_index + 1}/{MAX_BATCHES} | "
                f"Samples: {total_samples} | "
                f"Time: {elapsed:.2f}s | "
                f"Speed: {samples_per_second:.3f} samples/sec"
            )

        if batch_index + 1 >= MAX_BATCHES:
            break

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    samples_per_second = total_samples / elapsed
    seconds_per_sample = elapsed / total_samples

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print()
    print(f"Workers:             {NUM_WORKERS}")
    print(f"Samples processed:   {total_samples}")
    print(f"Total time:          {elapsed:.2f} seconds")
    print(f"Samples/sec:         {samples_per_second:.4f}")
    print(f"Seconds/sample:      {seconds_per_sample:.4f}")

    print()
    print("=" * 80)


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()