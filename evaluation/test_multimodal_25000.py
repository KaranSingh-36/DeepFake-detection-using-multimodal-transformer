import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ================================================================
# PROJECT IMPORTS
# ================================================================

from dataset.lavdf_dataset import LAVDFDataset
from models.multimodel_model import MultimodalCrossAttentionModel


# ================================================================
# CONFIGURATION
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    "results/checkpoints/"
    "multimodal_25000_epoch_1.pth"
)

NUM_FRAMES = 8
IMAGE_SIZE = 224
SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0

BATCH_SIZE = 1

# None = evaluate the complete test set
TEST_LIMIT = None

RESULTS_DIR = "results/test"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("MULTIMODAL 25K CHECKPOINT - LAV-DF TEST EVALUATION")
print("=" * 80)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()


# ================================================================
# LOAD TEST DATASET
# ================================================================

print("=" * 80)
print("LOADING LAV-DF TEST DATASET")
print("=" * 80)

test_dataset = LAVDFDataset(
    metadata_csv="metadata/lavdf_metadata.csv",
    dataset_root="datasets/LAV-DF/LAV-DF",
    split="test",
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    sample_rate=SAMPLE_RATE,
    audio_duration=AUDIO_DURATION
)

print(
    f"Full test dataset: {len(test_dataset)} samples"
)


# ================================================================
# OPTIONAL TEST LIMIT
# ================================================================

if TEST_LIMIT is not None:

    test_count = min(
        TEST_LIMIT,
        len(test_dataset)
    )

    indices = range(test_count)

    from torch.utils.data import Subset

    test_dataset = Subset(
        test_dataset,
        indices
    )

    print(
        f"Test samples used: {len(test_dataset)}"
    )

else:

    print(
        f"Test samples used: {len(test_dataset)}"
    )


# ================================================================
# DATALOADER
# ================================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print()


# ================================================================
# LOAD MODEL
# ================================================================

print("=" * 80)
print("LOADING MULTIMODAL MODEL")
print("=" * 80)

model = MultimodalCrossAttentionModel()

model = model.to(DEVICE)


# ================================================================
# LOAD CHECKPOINT
# ================================================================

print(
    f"Loading checkpoint:\n{CHECKPOINT_PATH}"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint loaded successfully.")

if "epoch" in checkpoint:
    print(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

if "train_accuracy" in checkpoint:
    print(
        f"Checkpoint training accuracy: "
        f"{checkpoint['train_accuracy']:.2f}%"
    )

if "val_accuracy" in checkpoint:
    print(
        f"Checkpoint validation accuracy: "
        f"{checkpoint['val_accuracy']:.2f}%"
    )

print()


# ================================================================
# LOSS
# ================================================================

criterion = nn.CrossEntropyLoss()


# ================================================================
# EVALUATION
# ================================================================

print("=" * 80)
print("STARTING TEST EVALUATION")
print("=" * 80)

all_labels = []
all_predictions = []
all_probabilities = []
all_files = []

total_loss = 0.0
total_samples = 0

start_time = time.time()


with torch.inference_mode():

    for batch_idx, batch in enumerate(test_loader):

        video = batch["frames"].to(
            DEVICE,
            non_blocking=True
        )

        audio = batch["audio"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True
        )

        # --------------------------------------------------------
        # MODEL
        # --------------------------------------------------------

        logits = model(
            video,
            audio
        )

        loss = criterion(
            logits,
            labels
        )

        # --------------------------------------------------------
        # PROBABILITIES
        # --------------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

        if "file" in batch:
            all_files.extend(
                batch["file"]
            )
        else:
            all_files.extend(
                ["unknown"] * labels.size(0)
            )

        total_loss += loss.item()

        total_samples += labels.size(0)

        # --------------------------------------------------------
        # PROGRESS
        # --------------------------------------------------------

        if (
            (batch_idx + 1) % 100 == 0
            or
            (batch_idx + 1) == len(test_loader)
        ):

            current_accuracy = (
                100.0
                * np.mean(
                    np.array(all_predictions)
                    ==
                    np.array(all_labels)
                )
            )

            if torch.cuda.is_available():

                memory = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                print(
                    f"Test "
                    f"{batch_idx + 1}/"
                    f"{len(test_loader)} | "
                    f"Accuracy: "
                    f"{current_accuracy:.2f}% | "
                    f"GPU: {memory:.2f} GB"
                )

            else:

                print(
                    f"Test "
                    f"{batch_idx + 1}/"
                    f"{len(test_loader)} | "
                    f"Accuracy: "
                    f"{current_accuracy:.2f}%"
                )


# ================================================================
# CONVERT RESULTS
# ================================================================

y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)


# ================================================================
# METRICS
# ================================================================

test_loss = (
    total_loss /
    len(test_loader)
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

try:

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

except ValueError:

    roc_auc = None


# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ================================================================
# CLASS-WISE METRICS
# ================================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "REAL",
        "DEEPFAKE"
    ],
    output_dict=True,
    zero_division=0
)


# ================================================================
# TIME
# ================================================================

elapsed = time.time() - start_time


# ================================================================
# PRINT RESULTS
# ================================================================

print()
print("=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(
    f"Test samples:       {total_samples}"
)

print(
    f"Test loss:          {test_loss:.4f}"
)

print(
    f"Accuracy:           {accuracy * 100:.2f}%"
)

print(
    f"Precision:          {precision:.4f}"
)

print(
    f"Recall:             {recall:.4f}"
)

print(
    f"F1-score:           {f1:.4f}"
)

if roc_auc is not None:

    print(
        f"ROC-AUC:            {roc_auc:.4f}"
    )

print()

print("CONFUSION MATRIX")
print("-----------------")

print(
    "                 Pred REAL    Pred FAKE"
)

print(
    f"Actual REAL      {tn:9d}    {fp:9d}"
)

print(
    f"Actual DEEPFAKE  {fn:9d}    {tp:9d}"
)

print()

print("CLASSIFICATION REPORT")
print("----------------------")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "REAL",
            "DEEPFAKE"
        ],
        zero_division=0
    )
)

print(
    f"Evaluation time: {elapsed:.2f} seconds"
)

print(
    f"Evaluation time: {elapsed / 3600:.2f} hours"
)


# ================================================================
# SAVE PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({

    "file": all_files,

    "true_label": y_true,

    "predicted_label": y_pred,

    "deepfake_probability": y_prob,

})

prediction_df["true_class"] = (
    prediction_df["true_label"]
    .map({
        0: "REAL",
        1: "DEEPFAKE"
    })
)

prediction_df["predicted_class"] = (
    prediction_df["predicted_label"]
    .map({
        0: "REAL",
        1: "DEEPFAKE"
    })
)

prediction_path = os.path.join(
    RESULTS_DIR,
    "multimodal_25000_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ================================================================
# SAVE METRICS JSON
# ================================================================

metrics = {

    "checkpoint":
        CHECKPOINT_PATH,

    "test_samples":
        int(total_samples),

    "test_loss":
        float(test_loss),

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "roc_auc":
        None if roc_auc is None
        else float(roc_auc),

    "true_negative":
        int(tn),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_positive":
        int(tp),

    "evaluation_time_seconds":
        float(elapsed),

    "real_precision":
        float(report["REAL"]["precision"]),

    "real_recall":
        float(report["REAL"]["recall"]),

    "real_f1":
        float(report["REAL"]["f1-score"]),

    "deepfake_precision":
        float(report["DEEPFAKE"]["precision"]),

    "deepfake_recall":
        float(report["DEEPFAKE"]["recall"]),

    "deepfake_f1":
        float(report["DEEPFAKE"]["f1-score"]),

}


metrics_path = os.path.join(
    RESULTS_DIR,
    "multimodal_25000_test_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ================================================================
# CONFUSION MATRIX IMAGE
# ================================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Multimodal 25K - LAV-DF Test Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["REAL", "DEEPFAKE"]
)

plt.yticks(
    [0, 1],
    ["REAL", "DEEPFAKE"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "multimodal_25000_test_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ================================================================
# GPU MEMORY
# ================================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print()
    print("GPU MEMORY")
    print("----------------------")
    print(
        f"Current allocated: {allocated:.2f} GB"
    )

    print(
        f"Peak allocated:    {peak:.2f} GB"
    )


# ================================================================
# FILES
# ================================================================

print()
print("=" * 80)
print("RESULT FILES")
print("=" * 80)

print(
    f"Predictions CSV:\n{prediction_path}"
)

print(
    f"Metrics JSON:\n{metrics_path}"
)

print(
    f"Confusion matrix:\n{cm_path}"
)

print()
print("=" * 80)
print("MULTIMODAL TEST EVALUATION COMPLETED")
print("=" * 80)