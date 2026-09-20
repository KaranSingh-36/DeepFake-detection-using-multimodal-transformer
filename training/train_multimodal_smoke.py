import os
import sys
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset.lavdf_dataset import LAVDFDataset
from models.multimodel_model import MultimodalCrossAttentionModel


# ================================================================
# CONFIGURATION
# ================================================================

BATCH_SIZE = 1
NUM_FRAMES = 8
AUDIO_SAMPLES = 64000

TRAIN_SAMPLES = 8
VAL_SAMPLES = 4

LEARNING_RATE = 1e-5


# ================================================================
# DEVICE
# ================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("REAL LAV-DF MULTIMODAL TRAINING SMOKE TEST")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    total_vram = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(
        f"VRAM: {total_vram:.2f} GB"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ================================================================
# DATASET
# ================================================================

print("\nLoading LAV-DF dataset...")

train_dataset = LAVDFDataset(
    split="train",
    num_frames=NUM_FRAMES
)

dev_dataset = LAVDFDataset(
    split="dev",
    num_frames=NUM_FRAMES
)

print(
    f"Full train dataset: {len(train_dataset)} samples"
)

print(
    f"Full dev dataset:   {len(dev_dataset)} samples"
)


# ================================================================
# SMALL SUBSETS
# ================================================================

train_dataset = Subset(
    train_dataset,
    range(
        min(TRAIN_SAMPLES, len(train_dataset))
    )
)

dev_dataset = Subset(
    dev_dataset,
    range(
        min(VAL_SAMPLES, len(dev_dataset))
    )
)

print(
    f"Smoke-test train samples: {len(train_dataset)}"
)

print(
    f"Smoke-test validation samples: {len(dev_dataset)}"
)


# ================================================================
# DATALOADERS
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ================================================================
# MODEL
# ================================================================

print("\nLoading multimodal model...")

model = MultimodalCrossAttentionModel()

model = model.to(device)

model.train()

print("Multimodal model loaded.")


# ================================================================
# LOSS + OPTIMIZER
# ================================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ================================================================
# HELPER FUNCTION
# ================================================================

def prepare_batch(batch):
    """
    Converts the dataset output into:

        video  -> [B, T, 3, 224, 224]
        audio  -> [B, samples]
        labels -> [B]

    Supports the dictionary format used by the project dataset.
    """

    if isinstance(batch, dict):

        video = batch.get("frames")

        if video is None:
            video = batch.get("video")

        audio = batch.get("audio")

        labels = batch.get("label")

        if labels is None:
            labels = batch.get("labels")

    elif isinstance(batch, (tuple, list)):

        if len(batch) < 3:
            raise ValueError(
                "Dataset batch must contain "
                "video, audio and label."
            )

        video = batch[0]
        audio = batch[1]
        labels = batch[2]

    else:

        raise TypeError(
            f"Unsupported batch type: {type(batch)}"
        )

    if video is None:
        raise ValueError(
            "Could not find video/frames in dataset batch."
        )

    if audio is None:
        raise ValueError(
            "Could not find audio in dataset batch."
        )

    if labels is None:
        raise ValueError(
            "Could not find labels in dataset batch."
        )

    # ------------------------------------------------------------
    # Make sure video has batch dimension
    # ------------------------------------------------------------

    if video.ndim == 4:
        video = video.unsqueeze(0)

    # ------------------------------------------------------------
    # Make sure audio has batch dimension
    # ------------------------------------------------------------

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    # ------------------------------------------------------------
    # Make sure labels are [B]
    # ------------------------------------------------------------

    if labels.ndim == 0:
        labels = labels.unsqueeze(0)

    # ------------------------------------------------------------
    # Correct audio length for this smoke test
    # ------------------------------------------------------------

    if audio.shape[-1] > AUDIO_SAMPLES:

        audio = audio[..., :AUDIO_SAMPLES]

    elif audio.shape[-1] < AUDIO_SAMPLES:

        padding = AUDIO_SAMPLES - audio.shape[-1]

        audio = torch.nn.functional.pad(
            audio,
            (0, padding)
        )

    return video, audio, labels


# ================================================================
# INSPECT FIRST REAL BATCH
# ================================================================

print("\nLoading first real batch...")

first_batch = next(iter(train_loader))

video, audio, labels = prepare_batch(
    first_batch
)

print("\nReal batch shapes:")

print(
    f"Video:  {video.shape}"
)

print(
    f"Audio:  {audio.shape}"
)

print(
    f"Labels: {labels.shape}"
)

print(
    f"Label values: {labels.tolist()}"
)


# ================================================================
# MOVE TO GPU
# ================================================================

video = video.to(
    device,
    non_blocking=True
)

audio = audio.to(
    device,
    non_blocking=True
)

labels = labels.to(
    device,
    non_blocking=True
)


# ================================================================
# SINGLE REAL TRAINING STEP
# ================================================================

print("\nRunning REAL-DATA forward pass...")

start_time = time.time()

optimizer.zero_grad(
    set_to_none=True
)

logits = model(
    video,
    audio
)

loss = criterion(
    logits,
    labels
)

print(
    f"Logits shape: {logits.shape}"
)

print(
    f"Loss: {loss.item():.6f}"
)

print("\nRunning REAL-DATA backward pass...")

loss.backward()

print(
    "Backward pass completed."
)

optimizer.step()

elapsed = time.time() - start_time

print(
    f"Training step time: {elapsed:.2f} seconds"
)


# ================================================================
# PREDICTION
# ================================================================

probabilities = torch.softmax(
    logits.detach(),
    dim=-1
)

predictions = torch.argmax(
    probabilities,
    dim=-1
)

print("\nPrediction:")

print(
    f"Actual:     {labels.tolist()}"
)

print(
    f"Predicted:  {predictions.tolist()}"
)

print(
    f"Probability: {probabilities.tolist()}"
)


# ================================================================
# GPU MEMORY
# ================================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print("\nGPU memory:")

    print(
        f"Current allocated: {allocated:.2f} MB"
    )

    print(
        f"Current reserved:  {reserved:.2f} MB"
    )

    print(
        f"Peak allocated:    {peak:.2f} MB"
    )


# ================================================================
# VALIDATION
# ================================================================

print("\nRunning validation smoke test...")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in dev_loader:

        video, audio, labels = prepare_batch(
            batch
        )

        video = video.to(
            device,
            non_blocking=True
        )

        audio = audio.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(
            video,
            audio
        )

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        # Only a smoke test.
        break


if total > 0:

    validation_accuracy = (
        correct / total
    )

else:

    validation_accuracy = 0.0


print(
    f"Validation samples tested: {total}"
)

print(
    f"Validation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


# ================================================================
# SAVE CHECKPOINT
# ================================================================

checkpoint_dir = os.path.join(
    PROJECT_ROOT,
    "results",
    "checkpoints"
)

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)

checkpoint_path = os.path.join(
    checkpoint_dir,
    "multimodal_real_data_smoke_test.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
        "validation_accuracy": validation_accuracy
    },
    checkpoint_path
)

print(
    f"\nCheckpoint saved to:"
)

print(
    checkpoint_path
)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 70)
print(
    "REAL-DATA MULTIMODAL TRAINING SMOKE TEST COMPLETED"
)
print("=" * 70)