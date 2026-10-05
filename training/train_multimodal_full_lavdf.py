import os
import sys
import time
import json
import random
import multiprocessing as mp

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataset.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalCrossAttentionModel


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT:
# Keep "benchmark" for the first run.
# Change to "full" only after the optimized 500/100 benchmark succeeds.
RUN_MODE = "full"

EPOCHS = 1
FULL_EPOCHS = 2

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2

NUM_FRAMES = 8
IMAGE_SIZE = 224
SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = NUM_WORKERS > 0

USE_AMP = torch.cuda.is_available()

# Same benchmark size as the run you just completed.
BENCHMARK_TRAIN_BATCHES = 500
BENCHMARK_VAL_BATCHES = 100

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "checkpoints",
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "full_lavdf_training",
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")


def make_loader(dataset, shuffle):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }

    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(dataset, **kwargs)


def move_batch(batch):
    video = batch["frames"].to(
        DEVICE,
        non_blocking=True,
    )

    audio = batch["audio"].to(
        DEVICE,
        non_blocking=True,
    )

    labels = batch["label"].to(
        DEVICE,
        non_blocking=True,
    )

    return video, audio, labels


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    max_batches=None,
):
    model.train()

    running_loss = torch.zeros(
        (),
        device=DEVICE,
    )

    correct = torch.zeros(
        (),
        dtype=torch.long,
        device=DEVICE,
    )

    sample_count = 0
    processed_batches = 0

    data_wait_seconds = 0.0
    compute_seconds = 0.0

    optimizer.zero_grad(
        set_to_none=True,
    )

    iterator = iter(loader)
    previous_step_end = time.perf_counter()

    while True:

        if (
            max_batches is not None
            and processed_batches >= max_batches
        ):
            break

        try:
            batch = next(iterator)

        except StopIteration:
            break

        batch_received = time.perf_counter()

        data_wait_seconds += (
            batch_received
            - previous_step_end
        )

        video, audio, labels = move_batch(batch)

        compute_start = time.perf_counter()

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            logits = model(
                video,
                audio,
            )

            loss = criterion(
                logits,
                labels,
            )

        loss_for_backward = (
            loss
            / GRADIENT_ACCUMULATION_STEPS
        )

        if USE_AMP:

            scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()

        processed_batches += 1

        should_step = (
            processed_batches
            % GRADIENT_ACCUMULATION_STEPS
            == 0
        )

        if should_step:

            if USE_AMP:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True,
            )

        predictions = torch.argmax(
            logits.detach(),
            dim=1,
        )

        running_loss += loss.detach()

        correct += (
            predictions == labels
        ).sum()

        sample_count += labels.numel()

        compute_seconds += (
            time.perf_counter()
            - compute_start
        )

        previous_step_end = (
            time.perf_counter()
        )

        if processed_batches % 100 == 0:

            current_loss = (
                running_loss.item()
                / processed_batches
            )

            current_accuracy = (
                100.0
                * correct.item()
                / max(sample_count, 1)
            )

            if torch.cuda.is_available():

                memory = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                print(
                    f"Batch "
                    f"{processed_batches:,}/"
                    f"{max_batches if max_batches is not None else len(loader):,} | "
                    f"Loss: "
                    f"{current_loss:.4f} | "
                    f"Accuracy: "
                    f"{current_accuracy:.2f}% | "
                    f"GPU: "
                    f"{memory:.2f} GB"
                )

            else:

                print(
                    f"Batch "
                    f"{processed_batches:,} | "
                    f"Loss: "
                    f"{current_loss:.4f} | "
                    f"Accuracy: "
                    f"{current_accuracy:.2f}%"
                )

    # Apply remaining accumulated gradients if the processed
    # batch count is not divisible by accumulation steps.
    if (
        processed_batches > 0
        and processed_batches
        % GRADIENT_ACCUMULATION_STEPS
        != 0
    ):

        if USE_AMP:

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()

        optimizer.zero_grad(
            set_to_none=True,
        )

    epoch_loss = (
        running_loss.item()
        / max(processed_batches, 1)
    )

    accuracy = (
        100.0
        * correct.item()
        / max(sample_count, 1)
    )

    return {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "batches": processed_batches,
        "samples": sample_count,
        "data_wait_seconds": data_wait_seconds,
        "compute_seconds": compute_seconds,
    }


@torch.inference_mode()
def validate(
    model,
    loader,
    criterion,
    max_batches=None,
):
    model.eval()

    running_loss = torch.zeros(
        (),
        device=DEVICE,
    )

    processed_batches = 0
    sample_count = 0

    # These are tiny [N,2] and [N] tensors, so keeping them on GPU
    # avoids a CPU synchronization on every validation batch.
    all_logits = []
    all_labels = []

    data_wait_seconds = 0.0
    compute_seconds = 0.0

    iterator = iter(loader)
    previous_step_end = time.perf_counter()

    while True:

        if (
            max_batches is not None
            and processed_batches >= max_batches
        ):
            break

        try:
            batch = next(iterator)

        except StopIteration:
            break

        batch_received = time.perf_counter()

        data_wait_seconds += (
            batch_received
            - previous_step_end
        )

        video, audio, labels = move_batch(batch)

        compute_start = time.perf_counter()

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            logits = model(
                video,
                audio,
            )

            loss = criterion(
                logits,
                labels,
            )

        running_loss += loss.detach()

        all_logits.append(
            logits.detach()
        )

        all_labels.append(
            labels.detach()
        )

        processed_batches += 1
        sample_count += labels.numel()

        compute_seconds += (
            time.perf_counter()
            - compute_start
        )

        previous_step_end = (
            time.perf_counter()
        )

        if processed_batches % 100 == 0:

            print(
                f"Validation batch "
                f"{processed_batches:,}/"
                f"{max_batches if max_batches is not None else len(loader):,}"
            )

    if processed_batches == 0:
        raise RuntimeError(
            "Validation processed zero batches."
        )

    logits_tensor = torch.cat(
        all_logits,
        dim=0,
    )

    labels_tensor = torch.cat(
        all_labels,
        dim=0,
    )

    predictions_tensor = torch.argmax(
        logits_tensor,
        dim=1,
    )

    probabilities_tensor = torch.softmax(
        logits_tensor,
        dim=1,
    )[:, 1]

    # One CPU transfer for the complete metric calculation.
    y_true = (
        labels_tensor
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions_tensor
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities_tensor
        .float()
        .cpu()
        .numpy()
    )

    val_loss = (
        running_loss.item()
        / processed_batches
    )

    val_accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
        * 100.0
    )

    val_precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    val_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:

        val_auc = roc_auc_score(
            y_true,
            y_prob,
        )

    except ValueError:

        val_auc = float("nan")

    return {
        "loss": val_loss,
        "accuracy": val_accuracy,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
        "auc": val_auc,
        "batches": processed_batches,
        "samples": sample_count,
        "data_wait_seconds": data_wait_seconds,
        "compute_seconds": compute_seconds,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_stats,
    val_stats,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "train_loss":
                train_stats["loss"],

            "train_accuracy":
                train_stats["accuracy"],

            "val_loss":
                val_stats["loss"],

            "val_accuracy":
                val_stats["accuracy"],

            "val_precision":
                val_stats["precision"],

            "val_recall":
                val_stats["recall"],

            "val_f1":
                val_stats["f1"],

            "val_auc":
                val_stats["auc"],

            "train_samples":
                train_stats["samples"],

            "validation_samples":
                val_stats["samples"],

            "num_frames":
                NUM_FRAMES,

            "image_size":
                IMAGE_SIZE,

            "sample_rate":
                SAMPLE_RATE,

            "audio_duration":
                AUDIO_DURATION,

            "learning_rate":
                LEARNING_RATE,

            "gradient_accumulation":
                GRADIENT_ACCUMULATION_STEPS,

            "seed":
                SEED,
        },
        path,
    )


def main():

    seed_everything()

    print("=" * 80)
    print(
        "LAV-DF MULTIMODAL TRAINING - "
        "OPTIMIZED PIPELINE"
    )
    print("=" * 80)

    print()
    print(
        f"Run mode: "
        f"{RUN_MODE}"
    )

    print(
        f"Device: "
        f"{DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    print(
        f"Epochs: "
        f"{FULL_EPOCHS if RUN_MODE == 'full' else EPOCHS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"AMP enabled: "
        f"{USE_AMP}"
    )

    print(
        f"DataLoader workers: "
        f"{NUM_WORKERS}"
    )

    print(
        f"Prefetch factor: "
        f"{PREFETCH_FACTOR}"
    )

    print()

    # ============================================================
    # DATASET
    # ============================================================

    print("=" * 80)
    print("LOADING LAV-DF DATASET")
    print("=" * 80)

    train_dataset = LAVDFDataset(
        metadata_csv="metadata/lavdf_metadata.csv",
        dataset_root=r"C:\AI_DATA\LAV-DF\LAV-DF",
        split="train",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        sample_rate=SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
    )

    dev_dataset = LAVDFDataset(
        metadata_csv="metadata/lavdf_metadata.csv",
        dataset_root=r"C:\AI_DATA\LAV-DF\LAV-DF",
        split="dev",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        sample_rate=SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
    )

    print()
    print(
        f"FULL TRAINING SAMPLES: "
        f"{len(train_dataset):,}"
    )

    print(
        f"FULL VALIDATION SAMPLES: "
        f"{len(dev_dataset):,}"
    )

    print()

    EXPECTED_TRAIN = 78703
    EXPECTED_DEV = 31501

    if len(train_dataset) != EXPECTED_TRAIN:

        print(
            "WARNING: Training size differs "
            f"from expected {EXPECTED_TRAIN}."
        )

    if len(dev_dataset) != EXPECTED_DEV:

        print(
            "WARNING: Dev size differs "
            f"from expected {EXPECTED_DEV}."
        )

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    metadata = pd.read_csv(
        "metadata/lavdf_metadata.csv"
    )

    train_meta = metadata[
        metadata["split"] == "train"
    ]

    dev_meta = metadata[
        metadata["split"] == "dev"
    ]

    print("=" * 80)
    print("CLASS DISTRIBUTION")
    print("=" * 80)

    print()
    print("TRAIN")
    print(
        train_meta["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("DEV")
    print(
        dev_meta["label"]
        .value_counts()
        .sort_index()
    )

    print()

    # ============================================================
    # DATALOADERS
    # ============================================================

    print("=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

    train_loader = make_loader(
        train_dataset,
        shuffle=True,
    )

    val_loader = make_loader(
        dev_dataset,
        shuffle=False,
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader):,}"
    )

    print()

    # ============================================================
    # MODEL
    # ============================================================

    print("=" * 80)
    print("LOADING MULTIMODAL MODEL")
    print("=" * 80)

    model = MultimodalCrossAttentionModel().to(
        DEVICE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Model loaded successfully."
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if USE_AMP:

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    else:

        scaler = None

    if RUN_MODE == "full":

        epochs_to_run = FULL_EPOCHS
        max_train = None
        max_val = None

    else:

        epochs_to_run = EPOCHS
        max_train = BENCHMARK_TRAIN_BATCHES
        max_val = BENCHMARK_VAL_BATCHES

    history = []

    best_val_auc = -1.0
    best_val_f1 = -1.0

    total_start = time.perf_counter()

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(
        epochs_to_run
    ):

        print()
        print("=" * 80)
        print(
            f"EPOCH "
            f"{epoch + 1}/"
            f"{epochs_to_run}"
        )
        print("=" * 80)

        epoch_start = time.perf_counter()

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=max_train,
        )

        epoch_training_time = (
            time.perf_counter()
            - epoch_start
        )

        print()
        print("-" * 80)
        print(
            f"EPOCH "
            f"{epoch + 1} "
            f"TRAINING COMPLETE"
        )
        print("-" * 80)

        print(
            f"Training Loss: "
            f"{train_stats['loss']:.4f}"
        )

        print(
            f"Training Accuracy: "
            f"{train_stats['accuracy']:.2f}%"
        )

        print(
            f"Training Batches: "
            f"{train_stats['batches']:,}"
        )

        print(
            f"Training Samples: "
            f"{train_stats['samples']:,}"
        )

        print(
            f"Training Time: "
            f"{epoch_training_time:.2f} seconds"
        )

        print(
            f"DataLoader wait: "
            f"{train_stats['data_wait_seconds']:.2f} seconds"
        )

        print(
            f"Model/compute time: "
            f"{train_stats['compute_seconds']:.2f} seconds"
        )

        print()

        # ========================================================
        # VALIDATION
        # ========================================================

        print("=" * 80)
        print(
            f"VALIDATION - "
            f"EPOCH {epoch + 1}"
        )
        print("=" * 80)

        validation_start = time.perf_counter()

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            max_batches=max_val,
        )

        validation_time = (
            time.perf_counter()
            - validation_start
        )

        print()
        print("-" * 80)
        print(
            f"EPOCH "
            f"{epoch + 1} "
            f"VALIDATION RESULTS"
        )
        print("-" * 80)

        print(
            f"Validation Loss:      "
            f"{val_stats['loss']:.4f}"
        )

        print(
            f"Validation Accuracy:  "
            f"{val_stats['accuracy']:.2f}%"
        )

        print(
            f"Validation Precision: "
            f"{val_stats['precision']:.4f}"
        )

        print(
            f"Validation Recall:    "
            f"{val_stats['recall']:.4f}"
        )

        print(
            f"Validation F1:        "
            f"{val_stats['f1']:.4f}"
        )

        print(
            f"Validation ROC-AUC:   "
            f"{val_stats['auc']:.4f}"
        )

        print(
            f"Validation Batches:   "
            f"{val_stats['batches']:,}"
        )

        print(
            f"Validation Samples:   "
            f"{val_stats['samples']:,}"
        )

        print(
            f"Validation Time:      "
            f"{validation_time:.2f} seconds"
        )

        print(
            f"Validation Data wait: "
            f"{val_stats['data_wait_seconds']:.2f} seconds"
        )

        print(
            f"Validation Compute:   "
            f"{val_stats['compute_seconds']:.2f} seconds"
        )

        # ========================================================
        # CHECKPOINT
        # ========================================================

        epoch_checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            (
                "multimodal_full_lavdf_"
                f"epoch_{epoch + 1}.pth"
            ),
        )

        save_checkpoint(
            epoch_checkpoint_path,
            model,
            optimizer,
            epoch + 1,
            train_stats,
            val_stats,
        )

        print()
        print(
            "Epoch checkpoint saved:"
        )
        print(
            epoch_checkpoint_path
        )

        if not np.isnan(
            val_stats["auc"]
        ):

            is_better = (
                val_stats["auc"]
                > best_val_auc
            )

        else:

            is_better = (
                val_stats["f1"]
                > best_val_f1
            )

        if is_better:

            best_val_auc = (
                val_stats["auc"]
            )

            best_val_f1 = (
                val_stats["f1"]
            )

            best_checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "multimodal_full_lavdf_best.pth",
            )

            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                train_stats,
                val_stats,
            )

            print()
            print(
                "NEW BEST CHECKPOINT SAVED:"
            )
            print(
                best_checkpoint_path
            )

        # ========================================================
        # HISTORY
        # ========================================================

        epoch_record = {
            "epoch":
                epoch + 1,

            "train_loss":
                float(
                    train_stats["loss"]
                ),

            "train_accuracy":
                float(
                    train_stats["accuracy"]
                ),

            "val_loss":
                float(
                    val_stats["loss"]
                ),

            "val_accuracy":
                float(
                    val_stats["accuracy"]
                ),

            "val_precision":
                float(
                    val_stats["precision"]
                ),

            "val_recall":
                float(
                    val_stats["recall"]
                ),

            "val_f1":
                float(
                    val_stats["f1"]
                ),

            "val_auc":
                float(
                    val_stats["auc"]
                ),

            "training_time_seconds":
                float(
                    epoch_training_time
                ),

            "validation_time_seconds":
                float(
                    validation_time
                ),

            "train_samples":
                int(
                    train_stats["samples"]
                ),

            "validation_samples":
                int(
                    val_stats["samples"]
                ),

            "train_data_wait_seconds":
                float(
                    train_stats[
                        "data_wait_seconds"
                    ]
                ),

            "train_compute_seconds":
                float(
                    train_stats[
                        "compute_seconds"
                    ]
                ),

            "val_data_wait_seconds":
                float(
                    val_stats[
                        "data_wait_seconds"
                    ]
                ),

            "val_compute_seconds":
                float(
                    val_stats[
                        "compute_seconds"
                    ]
                ),
        }

        history.append(
            epoch_record
        )

        history_path = os.path.join(
            RESULTS_DIR,
            "training_history.json",
        )

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=4,
            )

        # ========================================================
        # GPU MEMORY
        # ========================================================

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
            print("=" * 80)
            print("GPU MEMORY")
            print("=" * 80)

            print(
                f"Current allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"Current reserved:  "
                f"{reserved:.2f} GB"
            )

            print(
                f"Peak allocated:    "
                f"{peak:.2f} GB"
            )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 80)

    if RUN_MODE == "benchmark":

        print(
            "OPTIMIZED PIPELINE BENCHMARK COMPLETED"
        )

    else:

        print(
            "FULL LAV-DF TRAINING COMPLETED"
        )

    print("=" * 80)

    print()
    print(
        f"Run mode: "
        f"{RUN_MODE}"
    )

    print(
        f"Total time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Total time: "
        f"{total_time / 3600:.2f} hours"
    )

    print(
        f"Full training dataset: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Full validation dataset: "
        f"{len(dev_dataset):,}"
    )

    print(
        f"Epochs completed: "
        f"{epochs_to_run}"
    )

    print(
        f"Best Validation ROC-AUC: "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Best Validation F1: "
        f"{best_val_f1:.4f}"
    )

    print()
    print(
        "Training history:"
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "training_history.json",
        )
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    main()
