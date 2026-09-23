import os
import sys
import time
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset.lavdf_dataset import LAVDFDataset
from models.multimodel_model import MultimodalCrossAttentionModel


# ================================================================
# CONFIGURATION
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
EPOCHS = 1

TRAIN_SAMPLES = 25000
VAL_SAMPLES = 2000

NUM_FRAMES = 8
IMAGE_SIZE = 224
SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0

CHECKPOINT_DIR = "results/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ================================================================
# DEVICE INFORMATION
# ================================================================

print("=" * 70)
print("MULTIMODAL 25000-SAMPLE TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()


# ================================================================
# DATASET
# ================================================================

print("Loading LAV-DF datasets...")

train_dataset = LAVDFDataset(
    metadata_csv="metadata/lavdf_metadata.csv",
    dataset_root="datasets/LAV-DF/LAV-DF",
    split="train",
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    sample_rate=SAMPLE_RATE,
    audio_duration=AUDIO_DURATION
)

dev_dataset = LAVDFDataset(
    metadata_csv="metadata/lavdf_metadata.csv",
    dataset_root="datasets/LAV-DF/LAV-DF",
    split="dev",
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    sample_rate=SAMPLE_RATE,
    audio_duration=AUDIO_DURATION
)

print(f"Full training dataset: {len(train_dataset)}")
print(f"Full validation dataset: {len(dev_dataset)}")

# ================================================================
# CONTROLLED STRATIFIED SUBSETS
# ================================================================

print()
print("Creating controlled stratified subsets...")

SEED = 42

# Read metadata
metadata = pd.read_csv("metadata/lavdf_metadata.csv")

# Keep only the corresponding splits
train_meta = metadata[
    metadata["split"] == "train"
].reset_index(drop=True)

dev_meta = metadata[
    metadata["split"] == "dev"
].reset_index(drop=True)

# ------------------------------------------------
# Training subset
# ------------------------------------------------

train_real = train_meta[
    train_meta["label"] == 0
]

train_fake = train_meta[
    train_meta["label"] == 1
]

# Preserve original class distribution
train_real_count = round(
    TRAIN_SAMPLES * len(train_real) / len(train_meta)
)

train_fake_count = TRAIN_SAMPLES - train_real_count

train_real_sample = train_real.sample(
    n=train_real_count,
    random_state=SEED
)

train_fake_sample = train_fake.sample(
    n=train_fake_count,
    random_state=SEED
)

train_selected = pd.concat(
    [train_real_sample, train_fake_sample]
).sample(
    frac=1,
    random_state=SEED
)

# Dataset indices
train_indices = train_selected.index.tolist()

# ------------------------------------------------
# Validation subset
# ------------------------------------------------

dev_real = dev_meta[
    dev_meta["label"] == 0
]

dev_fake = dev_meta[
    dev_meta["label"] == 1
]

dev_real_count = round(
    VAL_SAMPLES * len(dev_real) / len(dev_meta)
)

dev_fake_count = VAL_SAMPLES - dev_real_count

dev_real_sample = dev_real.sample(
    n=dev_real_count,
    random_state=SEED
)

dev_fake_sample = dev_fake.sample(
    n=dev_fake_count,
    random_state=SEED
)

dev_selected = pd.concat(
    [dev_real_sample, dev_fake_sample]
).sample(
    frac=1,
    random_state=SEED
)

# Dataset indices
dev_indices = dev_selected.index.tolist()

# ------------------------------------------------
# Create subsets
# ------------------------------------------------

train_dataset = Subset(
    train_dataset,
    train_indices
)

dev_dataset = Subset(
    dev_dataset,
    dev_indices
)

print()
print("CONTROLLED SUBSET CREATED")
print("-" * 70)
print(f"Training samples:   {len(train_dataset)}")
print(f"  REAL:             {train_real_count}")
print(f"  DEEPFAKE:         {train_fake_count}")

print(f"Validation samples: {len(dev_dataset)}")
print(f"  REAL:             {dev_real_count}")
print(f"  DEEPFAKE:         {dev_fake_count}")

print(f"Random seed:        {SEED}")
print("-" * 70)
# ================================================================
# DATALOADERS
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ================================================================
# MODEL
# ================================================================

print()
print("Loading multimodal model...")

model = MultimodalCrossAttentionModel()
model = model.to(DEVICE)

print("Multimodal model loaded.")


# ================================================================
# LOSS + OPTIMIZER
# ================================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)


# ================================================================
# TRAINING
# ================================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):

        video = batch["frames"].to(DEVICE, non_blocking=True)
        audio = batch["audio"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits= model(
            video,
            audio
        )

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 25 == 0:
            current_accuracy = 100.0 * correct / total

            if torch.cuda.is_available():
                memory = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Accuracy: {current_accuracy:.2f}% | "
                    f"GPU Memory: {memory:.2f} GB"
                )
            else:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Accuracy: {current_accuracy:.2f}%"
                )

    epoch_loss /= len(train_loader)

    train_accuracy = 100.0 * correct / total

    elapsed = time.time() - start_time

    print()
    print("-" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {epoch_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Training Time: {elapsed:.2f} seconds")
    print("-" * 70)


# ================================================================
# VALIDATION
# ================================================================

print()
print("=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

model.eval()

val_loss = 0.0
val_correct = 0
val_total = 0

with torch.no_grad():

    for batch in val_loader:

        video = batch["frames"].to(DEVICE, non_blocking=True)
        audio = batch["audio"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        logits= model(
            video,
            audio
        )

        loss = criterion(logits, labels)

        val_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        val_correct += (
            predictions == labels
        ).sum().item()

        val_total += labels.size(0)

val_loss /= len(val_loader)

val_accuracy = (
    100.0 * val_correct / val_total
)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Validation Samples: {val_total}")

# ============================================================
# SAVE CHECKPOINT AFTER EVERY EPOCH
# ============================================================

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    f"multimodal_25000_epoch_{epoch + 1}.pth"
)

torch.save({
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),

    "train_loss": epoch_loss,
    "train_accuracy": 100.0 * correct / total,

    "val_loss": val_loss,
    "val_accuracy": val_accuracy,

    "train_samples": TRAIN_SAMPLES,
    "val_samples": VAL_SAMPLES,

}, checkpoint_path)

print("\nCheckpoint saved:")
print(checkpoint_path)

# ================================================================
# GPU MEMORY
# ================================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print()
    print("GPU MEMORY")
    print(f"Current allocated: {allocated:.2f} GB")
    print(f"Current reserved:  {reserved:.2f} GB")
    print(f"Peak allocated:    {peak:.2f} GB")


print()
print("=" * 70)
print("25000-SAMPLE MULTIMODAL TRAINING COMPLETED")
print("=" * 70)