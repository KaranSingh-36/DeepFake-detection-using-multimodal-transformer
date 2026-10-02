import os
import sys
import time
import json
import random
import multiprocessing

import numpy as np
import pandas as pd


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ================================================================
# PYTORCH
# ================================================================

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


# ================================================================
# PROJECT IMPORTS
# ================================================================

from dataset.lavdf_dataset import LAVDFDataset

from models.multimodal_model import (
    MultimodalCrossAttentionModel
)


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# TRAINING CONFIGURATION
# ================================================================

# IMPORTANT:
# Start with 1 epoch.
# We will first benchmark the corrected pipeline.
EPOCHS = 1

BATCH_SIZE = 1

# Keeps effective batch size = 2
# without increasing VRAM usage substantially.
GRADIENT_ACCUMULATION_STEPS = 2


# ================================================================
# LAV-DF CONFIGURATION
# ================================================================

NUM_FRAMES = 8

IMAGE_SIZE = 224

SAMPLE_RATE = 16000

AUDIO_DURATION = 4.0


# ================================================================
# OPTIMIZER
# ================================================================

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4


# ================================================================
# DATA LOADING
# ================================================================

# You already confirmed that 2 workers runs.
NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()

# Prefetch a small number of batches.
PREFETCH_FACTOR = 2


# ================================================================
# MIXED PRECISION
# ================================================================

USE_AMP = torch.cuda.is_available()


# ================================================================
# TEST MODE
# ================================================================

# ------------------------------------------------
# IMPORTANT
# ------------------------------------------------
#
# For the FIRST run after changing the dataset loader,
# keep this at 500.
#
# After the 500-sample benchmark succeeds,
# change it to None for the complete dataset.
#
# ------------------------------------------------

MAX_TRAIN_BATCHES = 500

# Validation benchmark.
#
# We do NOT want to process all 31,501 validation
# samples during the pipeline test.
#
MAX_VAL_BATCHES = 100


# ================================================================
# DATASET PATH
# ================================================================

DATASET_ROOT = r"C:\AI_DATA\LAV-DF\LAV-DF"


# ================================================================
# OUTPUT
# ================================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "checkpoints"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "full_lavdf_training"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )

    torch.backends.cudnn.benchmark = True


# ================================================================
# MAIN
# ================================================================

def main():

    # ============================================================
    # HEADER
    # ============================================================

    print("=" * 80)
    print("LAV-DF MULTIMODAL TRAINING")
    print("=" * 80)

    print()

    print(
        f"Device: {DEVICE}"
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

    print()

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"AMP enabled: {USE_AMP}"
    )

    print(
        f"DataLoader workers: "
        f"{NUM_WORKERS}"
    )

    print(
        f"Prefetch factor: "
        f"{PREFETCH_FACTOR}"
    )

    print(
        f"Max train batches: "
        f"{MAX_TRAIN_BATCHES}"
    )

    print(
        f"Max validation batches: "
        f"{MAX_VAL_BATCHES}"
    )

    print()


    # ============================================================
    # DATASET
    # ============================================================

    print("=" * 80)
    print("LOADING LAV-DF DATASET")
    print("=" * 80)

    print()

    train_dataset = LAVDFDataset(

        metadata_csv=(
            os.path.join(
                PROJECT_ROOT,
                "metadata",
                "lavdf_metadata.csv"
            )
        ),

        dataset_root=DATASET_ROOT,

        split="train",

        num_frames=NUM_FRAMES,

        image_size=IMAGE_SIZE,

        sample_rate=SAMPLE_RATE,

        audio_duration=AUDIO_DURATION
    )


    dev_dataset = LAVDFDataset(

        metadata_csv=(
            os.path.join(
                PROJECT_ROOT,
                "metadata",
                "lavdf_metadata.csv"
            )
        ),

        dataset_root=DATASET_ROOT,

        split="dev",

        num_frames=NUM_FRAMES,

        image_size=IMAGE_SIZE,

        sample_rate=SAMPLE_RATE,

        audio_duration=AUDIO_DURATION
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


    # ============================================================
    # EXPECTED DATASET SIZE
    # ============================================================

    EXPECTED_TRAIN = 78703

    EXPECTED_DEV = 31501

    if len(train_dataset) != EXPECTED_TRAIN:

        print(
            "WARNING: Training dataset size differs "
            f"from expected {EXPECTED_TRAIN}."
        )

    if len(dev_dataset) != EXPECTED_DEV:

        print(
            "WARNING: Dev dataset size differs "
            f"from expected {EXPECTED_DEV}."
        )


    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    metadata = pd.read_csv(
        os.path.join(
            PROJECT_ROOT,
            "metadata",
            "lavdf_metadata.csv"
        )
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

    print()

    train_loader_kwargs = {

        "batch_size": BATCH_SIZE,

        "shuffle": True,

        "num_workers": NUM_WORKERS,

        "pin_memory": PIN_MEMORY,
    }


    val_loader_kwargs = {

        "batch_size": BATCH_SIZE,

        "shuffle": False,

        "num_workers": NUM_WORKERS,

        "pin_memory": PIN_MEMORY,
    }


    if NUM_WORKERS > 0:

        train_loader_kwargs[
            "persistent_workers"
        ] = True

        val_loader_kwargs[
            "persistent_workers"
        ] = True

        train_loader_kwargs[
            "prefetch_factor"
        ] = PREFETCH_FACTOR

        val_loader_kwargs[
            "prefetch_factor"
        ] = PREFETCH_FACTOR


    train_loader = DataLoader(

        train_dataset,

        **train_loader_kwargs
    )


    val_loader = DataLoader(

        dev_dataset,

        **val_loader_kwargs
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

    print()

    model = MultimodalCrossAttentionModel()

    model = model.to(
        DEVICE
    )

    print(
        "Model loaded successfully."
    )

    print()


    # ============================================================
    # PARAMETER COUNT
    # ============================================================

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
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print()


    # ============================================================
    # LOSS
    # ============================================================

    criterion = nn.CrossEntropyLoss()


    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ============================================================
    # AMP
    # ============================================================

    if USE_AMP:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None


    # ============================================================
    # TRAINING HISTORY
    # ============================================================

    history = []

    best_val_auc = -1.0

    best_val_f1 = -1.0

    total_training_start = (
        time.time()
    )


    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(EPOCHS):

        print()

        print("=" * 80)

        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )

        print("=" * 80)

        print()


        # --------------------------------------------------------
        # TRAIN MODE
        # --------------------------------------------------------

        model.train()

        epoch_start = time.time()

        running_loss = 0.0

        train_predictions = []

        train_labels = []

        processed_batches = 0

        optimizer.zero_grad(
            set_to_none=True
        )


        # --------------------------------------------------------
        # TRAINING BATCHES
        # --------------------------------------------------------

        for batch_idx, batch in enumerate(
            train_loader
        ):

            if (
                MAX_TRAIN_BATCHES is not None
                and
                batch_idx >= MAX_TRAIN_BATCHES
            ):
                break


            # ----------------------------------------------------
            # MOVE DATA TO GPU
            # ----------------------------------------------------

            video = batch[
                "frames"
            ].to(
                DEVICE,
                non_blocking=True
            )


            audio = batch[
                "audio"
            ].to(
                DEVICE,
                non_blocking=True
            )


            labels = batch[
                "label"
            ].to(
                DEVICE,
                non_blocking=True
            )


            # ----------------------------------------------------
            # FORWARD
            # ----------------------------------------------------

            if USE_AMP:

                with torch.amp.autocast(

                    device_type="cuda",

                    dtype=torch.float16
                ):

                    logits = model(
                        video,
                        audio
                    )

                    loss = criterion(
                        logits,
                        labels
                    )

            else:

                logits = model(
                    video,
                    audio
                )

                loss = criterion(
                    logits,
                    labels
                )


            # ----------------------------------------------------
            # GRADIENT ACCUMULATION
            # ----------------------------------------------------

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


            # ----------------------------------------------------
            # OPTIMIZER STEP
            # ----------------------------------------------------

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
                    set_to_none=True
                )


            # ----------------------------------------------------
            # METRICS
            # ----------------------------------------------------

            running_loss += (
                loss.detach().item()
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            train_predictions.extend(

                predictions.detach()
                .cpu()
                .numpy()
                .tolist()
            )


            train_labels.extend(

                labels.detach()
                .cpu()
                .numpy()
                .tolist()
            )


            # ----------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------

            if (

                processed_batches % 100 == 0

                or

                (
                    MAX_TRAIN_BATCHES
                    is not None
                    and
                    processed_batches
                    == MAX_TRAIN_BATCHES
                )

            ):

                current_accuracy = (

                    accuracy_score(
                        train_labels,
                        train_predictions
                    )
                    * 100.0
                )


                if torch.cuda.is_available():

                    memory = (

                        torch.cuda.memory_allocated()
                        / 1024**3
                    )


                    print(

                        f"Batch "
                        f"{processed_batches:,}/"
                        f"{MAX_TRAIN_BATCHES if MAX_TRAIN_BATCHES is not None else len(train_loader):,} | "

                        f"Loss: "
                        f"{loss.item():.4f} | "

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
                        f"{loss.item():.4f} | "

                        f"Accuracy: "
                        f"{current_accuracy:.2f}%"
                    )


        # --------------------------------------------------------
        # HANDLE REMAINING GRADIENTS
        # --------------------------------------------------------

        if (

            processed_batches > 0

            and

            processed_batches
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
                set_to_none=True
            )


        # --------------------------------------------------------
        # TRAINING METRICS
        # --------------------------------------------------------

        if processed_batches > 0:

            epoch_loss = (

                running_loss
                / processed_batches
            )

        else:

            epoch_loss = 0.0


        train_accuracy = (

            accuracy_score(
                train_labels,
                train_predictions
            )
            * 100.0
        )


        epoch_training_time = (

            time.time()
            - epoch_start
        )


        print()

        print("-" * 80)

        print(
            f"EPOCH {epoch + 1} "
            "TRAINING COMPLETE"
        )

        print("-" * 80)

        print(
            f"Training Loss: "
            f"{epoch_loss:.4f}"
        )

        print(
            f"Training Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Training Batches: "
            f"{processed_batches:,}"
        )

        print(
            f"Training Samples: "
            f"{len(train_labels):,}"
        )

        print(
            f"Training Time: "
            f"{epoch_training_time:.2f} seconds"
        )

        print()


        # ========================================================
        # VALIDATION
        # ========================================================

        print("=" * 80)

        print(
            f"VALIDATION - EPOCH "
            f"{epoch + 1}"
        )

        print("=" * 80)

        model.eval()

        val_running_loss = 0.0

        val_predictions = []

        val_labels = []

        val_probabilities = []

        validation_batches = 0

        validation_start = time.time()


        with torch.no_grad():

            for batch_idx, batch in enumerate(
                val_loader
            ):

                if (

                    MAX_VAL_BATCHES is not None

                    and

                    batch_idx
                    >= MAX_VAL_BATCHES
                ):
                    break


                video = batch[
                    "frames"
                ].to(
                    DEVICE,
                    non_blocking=True
                )


                audio = batch[
                    "audio"
                ].to(
                    DEVICE,
                    non_blocking=True
                )


                labels = batch[
                    "label"
                ].to(
                    DEVICE,
                    non_blocking=True
                )


                if USE_AMP:

                    with torch.amp.autocast(

                        device_type="cuda",

                        dtype=torch.float16
                    ):

                        logits = model(
                            video,
                            audio
                        )

                        loss = criterion(
                            logits,
                            labels
                        )

                else:

                    logits = model(
                        video,
                        audio
                    )

                    loss = criterion(
                        logits,
                        labels
                    )


                probabilities = torch.softmax(
                    logits,
                    dim=1
                )


                predictions = torch.argmax(
                    logits,
                    dim=1
                )


                val_running_loss += (
                    loss.detach().item()
                )


                val_predictions.extend(

                    predictions
                    .cpu()
                    .numpy()
                    .tolist()
                )


                val_labels.extend(

                    labels
                    .cpu()
                    .numpy()
                    .tolist()
                )


                val_probabilities.extend(

                    probabilities[:, 1]
                    .float()
                    .cpu()
                    .numpy()
                    .tolist()
                )


                validation_batches += 1


                if (
                    validation_batches % 100 == 0
                ):

                    print(

                        f"Validation batch "
                        f"{validation_batches:,}/"
                        f"{MAX_VAL_BATCHES if MAX_VAL_BATCHES is not None else len(val_loader):,}"
                    )


        # --------------------------------------------------------
        # VALIDATION METRICS
        # --------------------------------------------------------

        if validation_batches > 0:

            val_loss = (

                val_running_loss
                / validation_batches
            )

        else:

            val_loss = 0.0


        val_accuracy = (

            accuracy_score(
                val_labels,
                val_predictions
            )
            * 100.0
        )


        val_precision = precision_score(

            val_labels,

            val_predictions,

            zero_division=0
        )


        val_recall = recall_score(

            val_labels,

            val_predictions,

            zero_division=0
        )


        val_f1 = f1_score(

            val_labels,

            val_predictions,

            zero_division=0
        )


        try:

            val_auc = roc_auc_score(

                val_labels,

                val_probabilities
            )

        except ValueError:

            val_auc = float("nan")


        validation_time = (

            time.time()
            - validation_start
        )


        # --------------------------------------------------------
        # PRINT VALIDATION
        # --------------------------------------------------------

        print()

        print("-" * 80)

        print(
            f"EPOCH {epoch + 1} "
            "VALIDATION RESULTS"
        )

        print("-" * 80)

        print(
            f"Validation Loss:      "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy:  "
            f"{val_accuracy:.2f}%"
        )

        print(
            f"Validation Precision: "
            f"{val_precision:.4f}"
        )

        print(
            f"Validation Recall:    "
            f"{val_recall:.4f}"
        )

        print(
            f"Validation F1:        "
            f"{val_f1:.4f}"
        )

        print(
            f"Validation ROC-AUC:   "
            f"{val_auc:.4f}"
        )

        print(
            f"Validation Batches:   "
            f"{validation_batches:,}"
        )

        print(
            f"Validation Samples:   "
            f"{len(val_labels):,}"
        )

        print(
            f"Validation Time:      "
            f"{validation_time:.2f} seconds"
        )

        print()


        # ========================================================
        # CHECKPOINT
        # ========================================================

        epoch_checkpoint_path = os.path.join(

            CHECKPOINT_DIR,

            f"multimodal_full_lavdf_epoch_{epoch + 1}.pth"
        )


        torch.save(

            {

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scaler_state_dict":
                    (
                        scaler.state_dict()
                        if scaler is not None
                        else None
                    ),

                "train_loss":
                    epoch_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "val_precision":
                    val_precision,

                "val_recall":
                    val_recall,

                "val_f1":
                    val_f1,

                "val_auc":
                    val_auc,

                "train_samples":
                    len(train_labels),

                "validation_samples":
                    len(val_labels),

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

                "max_train_batches":
                    MAX_TRAIN_BATCHES,

                "max_val_batches":
                    MAX_VAL_BATCHES,
            },

            epoch_checkpoint_path
        )


        print(
            "Epoch checkpoint saved:"
        )

        print(
            epoch_checkpoint_path
        )

        print()


        # ========================================================
        # BEST CHECKPOINT
        # ========================================================

        if not np.isnan(val_auc):

            is_better = (
                val_auc > best_val_auc
            )

        else:

            is_better = (
                val_f1 > best_val_f1
            )


        if is_better:

            best_val_auc = val_auc

            best_val_f1 = val_f1


            best_checkpoint_path = os.path.join(

                CHECKPOINT_DIR,

                "multimodal_full_lavdf_best.pth"
            )


            torch.save(

                {

                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scaler_state_dict":
                        (
                            scaler.state_dict()
                            if scaler is not None
                            else None
                        ),

                    "val_loss":
                        val_loss,

                    "val_accuracy":
                        val_accuracy,

                    "val_precision":
                        val_precision,

                    "val_recall":
                        val_recall,

                    "val_f1":
                        val_f1,

                    "val_auc":
                        val_auc,

                    "seed":
                        SEED,
                },

                best_checkpoint_path
            )


            print(
                "NEW BEST CHECKPOINT SAVED:"
            )

            print(
                best_checkpoint_path
            )

            print()


        # ========================================================
        # HISTORY
        # ========================================================

        epoch_record = {

            "epoch":
                epoch + 1,

            "train_loss":
                float(epoch_loss),

            "train_accuracy":
                float(train_accuracy),

            "val_loss":
                float(val_loss),

            "val_accuracy":
                float(val_accuracy),

            "val_precision":
                float(val_precision),

            "val_recall":
                float(val_recall),

            "val_f1":
                float(val_f1),

            "val_auc":
                float(val_auc),

            "training_time_seconds":
                float(
                    epoch_training_time
                ),

            "validation_time_seconds":
                float(
                    validation_time
                ),

            "train_samples":
                len(train_labels),

            "validation_samples":
                len(val_labels),
        }


        history.append(
            epoch_record
        )


        history_path = os.path.join(

            RESULTS_DIR,

            "training_history.json"
        )


        with open(

            history_path,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                history,

                f,

                indent=4
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


            print("=" * 80)

            print(
                "GPU MEMORY"
            )

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

            print()


    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    total_training_time = (

        time.time()
        - total_training_start
    )


    print()

    print("=" * 80)

    print(
        "TRAINING TEST COMPLETED"
    )

    print("=" * 80)

    print()

    print(
        f"Total time: "
        f"{total_training_time:.2f} seconds"
    )

    print(
        f"Total time: "
        f"{total_training_time / 3600:.2f} hours"
    )

    print()

    print(
        f"Full training dataset: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Full validation dataset: "
        f"{len(dev_dataset):,}"
    )

    print()

    print(
        f"Processed training samples: "
        f"{len(train_labels):,}"
    )

    print(
        f"Processed validation samples: "
        f"{len(val_labels):,}"
    )

    print()

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
            "training_history.json"
        )
    )

    print()

    print("=" * 80)

    print(
        "PIPELINE TEST FINISHED"
    )

    print("=" * 80)


# ================================================================
# WINDOWS SAFE ENTRY POINT
# ================================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()