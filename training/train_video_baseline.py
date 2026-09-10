import os
import sys
import time

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset.lavdf_dataset import LAVDFDataset
from models.video_model import VideoDeepfakeDetector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_FRAMES = 8
IMAGE_SIZE = 224
BATCH_SIZE = 2

# Small smoke test
TRAIN_SAMPLES = 100
VAL_SAMPLES = 20

EPOCHS = 1

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("VIDEO-ONLY BASELINE TRAINING")
    print("=" * 70)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    # -----------------------------------------------------
    # Training dataset
    # -----------------------------------------------------

    print()
    print("Loading training dataset...")

    train_dataset = LAVDFDataset(
        split="train",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        sample_rate=16000,
        audio_duration=4.0,
    )

    print(
        "Full training dataset:",
        len(train_dataset)
    )

    # -----------------------------------------------------
    # Validation dataset
    # -----------------------------------------------------

    print()
    print("Loading validation dataset...")

    val_dataset = LAVDFDataset(
        split="dev",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        sample_rate=16000,
        audio_duration=4.0,
    )

    print(
        "Full validation dataset:",
        len(val_dataset)
    )

    # -----------------------------------------------------
    # Small smoke-test subsets
    # -----------------------------------------------------

    train_size = min(
        TRAIN_SAMPLES,
        len(train_dataset)
    )

    val_size = min(
        VAL_SAMPLES,
        len(val_dataset)
    )

    train_subset = Subset(
        train_dataset,
        range(train_size)
    )

    val_subset = Subset(
        val_dataset,
        range(val_size)
    )

    print()
    print("Training samples:", len(train_subset))
    print("Validation samples:", len(val_subset))

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print()
    print("Creating video model...")

    model = VideoDeepfakeDetector(
        freeze_vit=True
    )

    model = model.to(device)

    print("Model created.")

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # -----------------------------------------------------
    # Mixed precision
    # -----------------------------------------------------

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        model.train()

        # Keep ViT frozen
        if hasattr(model, "vit"):
            model.vit.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):

            frames = batch["frames"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                logits = model(frames)

                loss = criterion(
                    logits,
                    labels
                )

            # ---------------------------------------------
            # Backward
            # ---------------------------------------------

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            running_loss += loss.item()

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if (batch_idx + 1) % 10 == 0:

                current_accuracy = (
                    correct / total
                ) * 100

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Accuracy: {current_accuracy:.2f}%"
                )

        epoch_loss = (
            running_loss /
            len(train_loader)
        )

        epoch_accuracy = (
            correct / total
        ) * 100

        elapsed = time.time() - start_time

        print()
        print(
            f"Epoch {epoch + 1} complete"
        )

        print(
            f"Training loss: {epoch_loss:.4f}"
        )

        print(
            f"Training accuracy: {epoch_accuracy:.2f}%"
        )

        print(
            f"Time: {elapsed:.1f} seconds"
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                frames = batch["frames"].to(
                    device,
                    non_blocking=True
                )

                labels = batch["label"].to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp
                ):

                    logits = model(frames)

                    loss = criterion(
                        logits,
                        labels
                    )

                val_loss += loss.item()

                predictions = logits.argmax(
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        validation_loss = (
            val_loss /
            len(val_loader)
        )

        validation_accuracy = (
            val_correct /
            val_total
        ) * 100

        print()
        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.2f}%"
        )

    # -----------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "video_baseline_smoke_test.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": EPOCHS,
            "num_frames": NUM_FRAMES,
            "image_size": IMAGE_SIZE,
        },
        checkpoint_path
    )

    print()
    print("=" * 70)
    print("TRAINING SMOKE TEST SUCCESSFUL")
    print("=" * 70)

    print()
    print(
        "Checkpoint saved to:"
    )

    print(
        checkpoint_path
    )


if __name__ == "__main__":
    main()