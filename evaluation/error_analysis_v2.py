"""
LAV-DF Multimodal Transformer
Research Error Analysis

Analyzes the completed test predictions from:
    multimodal_25000_epoch2_test_predictions.csv

Uses:
    - Prediction CSV
    - LAV-DF metadata

Outputs:
    results/error_analysis/
"""

import os
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_FILE = (
    "results/test/multimodal_25000_epoch2_test_predictions.csv"
)

METADATA_FILE = (
    "metadata/lavdf_metadata.csv"
)

OUTPUT_DIR = "results/error_analysis_v2"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 80)
print("LAV-DF MULTIMODAL ERROR ANALYSIS")
print("=" * 80)


# ============================================================
# LOAD FILES
# ============================================================

print("\n[1] Loading prediction file...")

if not os.path.exists(PREDICTIONS_FILE):
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_FILE}"
    )

pred = pd.read_csv(PREDICTIONS_FILE)

print(f"Prediction rows: {len(pred):,}")
print("Prediction columns:")
print(list(pred.columns))


print("\n[2] Loading LAV-DF metadata...")

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

meta = pd.read_csv(METADATA_FILE)

print(f"Metadata rows: {len(meta):,}")
print("Metadata columns:")
print(list(meta.columns))


# ============================================================
# AUTOMATIC COLUMN DETECTION
# ============================================================

def find_column(df, possible_names, required=True):
    """
    Find a column using several possible names.
    """

    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    if required:
        raise ValueError(
            f"\nCould not find required column.\n"
            f"Tried: {possible_names}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


FILE_COL_PRED = find_column(
    pred,
    ["file", "filename", "video", "video_file", "path"]
)

TRUE_COL = find_column(
    pred,
    ["true_label", "actual", "actual_label", "label", "target"]
)

PRED_COL = find_column(
    pred,
    ["predicted_label", "prediction", "pred", "pred_label"]
)

# Optional confidence columns
REAL_PROB_COL = find_column(
    pred,
    [
        "real_probability",
        "real_prob",
        "prob_real",
        "real_confidence",
        "real_score"
    ],
    required=False
)

FAKE_PROB_COL = find_column(
    pred,
    [
        "deepfake_probability",
        "fake_probability",
        "fake_prob",
        "prob_fake",
        "fake_confidence",
        "fake_score"
    ],
    required=False
)


print("\nDetected prediction columns:")
print(f"File       : {FILE_COL_PRED}")
print(f"True label : {TRUE_COL}")
print(f"Pred label : {PRED_COL}")
print(f"Real prob  : {REAL_PROB_COL}")
print(f"Fake prob  : {FAKE_PROB_COL}")


# ============================================================
# NORMALIZE LABELS
# ============================================================

def normalize_label(value):

    value = str(value).strip().lower()

    if value in ["0", "real", "0.0"]:
        return 0

    if value in ["1", "fake", "deepfake", "1.0"]:
        return 1

    # Handle numeric strings
    try:
        numeric = float(value)

        if numeric == 0:
            return 0

        if numeric == 1:
            return 1

    except Exception:
        pass

    raise ValueError(
        f"Unknown label encountered: {value}"
    )


pred["true_label_norm"] = pred[TRUE_COL].apply(normalize_label)
pred["pred_label_norm"] = pred[PRED_COL].apply(normalize_label)


# ============================================================
# NORMALIZE FILE PATH
# ============================================================

def normalize_path(path):

    path = str(path).strip()

    # Convert Windows separators
    path = path.replace("\\", "/")

    # Remove possible leading ./ 
    if path.startswith("./"):
        path = path[2:]

    return path


pred["file_key"] = pred[FILE_COL_PRED].apply(normalize_path)
meta["file_key"] = meta["file"].apply(normalize_path)


# ============================================================
# MERGE PREDICTIONS WITH METADATA
# ============================================================

print("\n[3] Joining predictions with LAV-DF metadata...")

df = pred.merge(
    meta,
    on="file_key",
    how="left",
    suffixes=("", "_metadata")
)

matched = df["duration"].notna().sum() if "duration" in df.columns else 0

print(f"Prediction rows      : {len(df):,}")
print(f"Metadata matched     : {matched:,}")
print(f"Metadata unmatched   : {len(df) - matched:,}")

if matched != len(df):
    print(
        "\nWARNING:"
        "\nSome prediction files did not match the metadata."
        "\nWe will still analyze the available records."
    )


# ============================================================
# ERROR TYPE
# ============================================================

df["correct"] = (
    df["true_label_norm"] ==
    df["pred_label_norm"]
)

df["error_type"] = "CORRECT"

df.loc[
    (df["true_label_norm"] == 1) &
    (df["pred_label_norm"] == 0),
    "error_type"
] = "FALSE_NEGATIVE"

df.loc[
    (df["true_label_norm"] == 0) &
    (df["pred_label_norm"] == 1),
    "error_type"
] = "FALSE_POSITIVE"


# ============================================================
# BASIC SUMMARY
# ============================================================

total = len(df)

correct = int(df["correct"].sum())

false_negative = int(
    ((df["true_label_norm"] == 1) &
     (df["pred_label_norm"] == 0)).sum()
)

false_positive = int(
    ((df["true_label_norm"] == 0) &
     (df["pred_label_norm"] == 1)).sum()
)

true_negative = int(
    ((df["true_label_norm"] == 0) &
     (df["pred_label_norm"] == 0)).sum()
)

true_positive = int(
    ((df["true_label_norm"] == 1) &
     (df["pred_label_norm"] == 1)).sum()
)


accuracy = correct / total if total else 0

real_total = int(
    (df["true_label_norm"] == 0).sum()
)

fake_total = int(
    (df["true_label_norm"] == 1).sum()
)

fn_rate = (
    false_negative / fake_total
    if fake_total
    else 0
)

fp_rate = (
    false_positive / real_total
    if real_total
    else 0
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OVERALL ERROR SUMMARY")
print("=" * 80)

print(f"Total samples       : {total:,}")
print(f"Correct             : {correct:,}")
print(f"Incorrect           : {total - correct:,}")

print()
print(f"True Negative       : {true_negative:,}")
print(f"False Positive      : {false_positive:,}")
print(f"False Negative      : {false_negative:,}")
print(f"True Positive       : {true_positive:,}")

print()
print(f"Accuracy             : {accuracy * 100:.4f}%")
print(f"False Positive Rate  : {fp_rate * 100:.4f}%")
print(f"False Negative Rate  : {fn_rate * 100:.4f}%")

print("=" * 80)


# ============================================================
# SAVE FALSE POSITIVES
# ============================================================

fp = df[
    (df["true_label_norm"] == 0) &
    (df["pred_label_norm"] == 1)
].copy()

fp.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "false_positives.csv"
    ),
    index=False
)

print(
    f"\nFalse positives saved: {len(fp):,}"
)


# ============================================================
# SAVE FALSE NEGATIVES
# ============================================================

fn = df[
    (df["true_label_norm"] == 1) &
    (df["pred_label_norm"] == 0)
].copy()

fn.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "false_negatives.csv"
    ),
    index=False
)

print(
    f"False negatives saved: {len(fn):,}"
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

error_distribution = (
    df["error_type"]
    .value_counts()
    .rename_axis("error_type")
    .reset_index(name="count")
)

error_distribution["percentage"] = (
    error_distribution["count"] /
    total * 100
)

error_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "error_distribution.csv"
    ),
    index=False
)


# ============================================================
# ANALYSIS BY VIDEO/AUDIO MODIFICATION
# ============================================================

if (
    "modify_video" in df.columns and
    "modify_audio" in df.columns
):

    print("\n" + "=" * 80)
    print("ERROR RATE BY MODALITY")
    print("=" * 80)

    modality_df = (
        df.groupby(
            ["modify_video", "modify_audio"],
            dropna=False
        )
        .agg(
            total_samples=("correct", "size"),
            correct=("correct", "sum")
        )
        .reset_index()
    )

    modality_df["errors"] = (
        modality_df["total_samples"] -
        modality_df["correct"]
    )

    modality_df["accuracy"] = (
        modality_df["correct"] /
        modality_df["total_samples"]
    )

    modality_df["error_rate"] = (
        modality_df["errors"] /
        modality_df["total_samples"]
    )

    def normalize_binary(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, bool):
            return int(value)
        value = str(value).strip().lower()
        if value in {"1", "1.0", "true", "yes"}:
            return 1
        if value in {"0", "0.0", "false", "no"}:
            return 0
        try:
            value = float(value)
            if value in (0, 1):
                return int(value)
        except Exception:
            pass
        return np.nan

    df["modify_video_norm"] = df["modify_video"].apply(normalize_binary)
    df["modify_audio_norm"] = df["modify_audio"].apply(normalize_binary)

    def modality_name(row):
        video = row["modify_video_norm"]
        audio = row["modify_audio_norm"]
        if pd.isna(video) or pd.isna(audio):
            return "Unknown"
        if int(video) == 1 and int(audio) == 1:
            return "Video + Audio"
        if int(video) == 1 and int(audio) == 0:
            return "Video only"
        if int(video) == 0 and int(audio) == 1:
            return "Audio only"
        if int(video) == 0 and int(audio) == 0:
            return "Real / Unmodified"
        return "Unknown"

    modality_df["modality"] = modality_df.apply(
        modality_name,
        axis=1
    )

    modality_df = modality_df[
        [
            "modality",
            "total_samples",
            "correct",
            "errors",
            "accuracy",
            "error_rate"
        ]
    ]

    print(modality_df.to_string(index=False))

    modality_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "error_by_modality.csv"
        ),
        index=False
    )


# ============================================================
# ANALYSIS BY NUMBER OF FAKE PERIODS
# ============================================================

if "n_fakes" in df.columns:

    print("\n" + "=" * 80)
    print("ERROR RATE BY NUMBER OF FAKE SEGMENTS")
    print("=" * 80)

    fake_segment_df = (
        df.groupby("n_fakes", dropna=False)
        .agg(
            total_samples=("correct", "size"),
            correct=("correct", "sum")
        )
        .reset_index()
    )

    fake_segment_df["errors"] = (
        fake_segment_df["total_samples"] -
        fake_segment_df["correct"]
    )

    fake_segment_df["accuracy"] = (
        fake_segment_df["correct"] /
        fake_segment_df["total_samples"]
    )

    fake_segment_df["error_rate"] = (
        fake_segment_df["errors"] /
        fake_segment_df["total_samples"]
    )

    print(fake_segment_df.to_string(index=False))

    fake_segment_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "error_by_fake_segments.csv"
        ),
        index=False
    )


# ============================================================
# ANALYSIS BY VIDEO DURATION
# ============================================================

if "duration" in df.columns:

    print("\n" + "=" * 80)
    print("ERROR RATE BY VIDEO DURATION")
    print("=" * 80)

    bins = [
        0,
        2,
        4,
        6,
        8,
        10,
        15,
        30,
        np.inf
    ]

    labels = [
        "<=2s",
        "2-4s",
        "4-6s",
        "6-8s",
        "8-10s",
        "10-15s",
        "15-30s",
        ">30s"
    ]

    df["duration_group"] = pd.cut(
        df["duration"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    duration_df = (
        df.groupby(
            "duration_group",
            observed=False
        )
        .agg(
            total_samples=("correct", "size"),
            correct=("correct", "sum")
        )
        .reset_index()
    )

    duration_df["errors"] = (
        duration_df["total_samples"] -
        duration_df["correct"]
    )

    duration_df["accuracy"] = (
        duration_df["correct"] /
        duration_df["total_samples"]
    )

    duration_df["error_rate"] = (
        duration_df["errors"] /
        duration_df["total_samples"]
    )

    print(duration_df.to_string(index=False))

    duration_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "error_by_duration.csv"
        ),
        index=False
    )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

# The evaluator stores P(DEEPFAKE), so a real-probability column is not required.
confidence_available = (
    FAKE_PROB_COL is not None or REAL_PROB_COL is not None
)

if confidence_available:
    if FAKE_PROB_COL is not None:
        df["fake_probability"] = pd.to_numeric(df[FAKE_PROB_COL], errors="coerce")
        df["real_probability"] = 1.0 - df["fake_probability"]
    else:
        df["real_probability"] = pd.to_numeric(df[REAL_PROB_COL], errors="coerce")
        df["fake_probability"] = 1.0 - df["real_probability"]

    df["prediction_confidence"] = df[["real_probability", "fake_probability"]].max(axis=1)

    errors = df[~df["correct"]].copy()

    print("\n" + "=" * 80)
    print("INCORRECT PREDICTION CONFIDENCE")
    print("=" * 80)
    if len(errors) > 0:
        print(f"Mean confidence : {errors['prediction_confidence'].mean():.4f}")
        print(f"Max confidence  : {errors['prediction_confidence'].max():.4f}")
        print(f"Min confidence  : {errors['prediction_confidence'].min():.4f}")

    hardest_errors = errors.sort_values("prediction_confidence", ascending=False)
    hardest_errors.to_csv(os.path.join(OUTPUT_DIR, "highest_confidence_errors.csv"), index=False)

    fn_conf = df[(df["true_label_norm"] == 1) & (df["pred_label_norm"] == 0)].copy()
    fn_conf = fn_conf.sort_values("fake_probability", ascending=True)
    fn_conf.to_csv(os.path.join(OUTPUT_DIR, "highest_confidence_false_negatives.csv"), index=False)

    fp_conf = df[(df["true_label_norm"] == 0) & (df["pred_label_norm"] == 1)].copy()
    fp_conf = fp_conf.sort_values("fake_probability", ascending=False)
    fp_conf.to_csv(os.path.join(OUTPUT_DIR, "highest_confidence_false_positives.csv"), index=False)

    uncertain = df[(df["prediction_confidence"] >= 0.50) & (df["prediction_confidence"] <= 0.60)].copy()
    uncertain.to_csv(os.path.join(OUTPUT_DIR, "uncertain_predictions.csv"), index=False)

    print(f"Highest-confidence errors saved: {len(hardest_errors):,}")
    print(f"Highest-confidence false negatives saved: {len(fn_conf):,}")
    print(f"Highest-confidence false positives saved: {len(fp_conf):,}")
    print(f"Uncertain predictions saved: {len(uncertain):,}")

# ============================================================
# FAKE-PERIOD COUNT ANALYSIS
# ============================================================

if "fake_periods" in df.columns:
    def parse_fake_period_count(value):
        if pd.isna(value):
            return 0
        text = str(value).strip()
        if text in {"", "[]", "None", "nan", "null"}:
            return 0
        try:
            parsed = json.loads(text)
            return len(parsed) if isinstance(parsed, list) else 0
        except Exception:
            try:
                parsed = ast.literal_eval(text)
                return len(parsed) if isinstance(parsed, list) else 0
            except Exception:
                return 0

    df["fake_period_count"] = df["fake_periods"].apply(parse_fake_period_count)
    fake_period_df = (
        df.groupby("fake_period_count", dropna=False)
        .agg(total_samples=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    fake_period_df["errors"] = fake_period_df["total_samples"] - fake_period_df["correct"]
    fake_period_df["accuracy"] = fake_period_df["correct"] / fake_period_df["total_samples"]
    fake_period_df["error_rate"] = fake_period_df["errors"] / fake_period_df["total_samples"]
    fake_period_df.to_csv(os.path.join(OUTPUT_DIR, "error_by_fake_period_count.csv"), index=False)
    print("\n" + "=" * 80)
    print("ERROR RATE BY PARSED FAKE-PERIOD COUNT")
    print("=" * 80)
    print(fake_period_df.to_string(index=False))


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FALSE NEGATIVE ANALYSIS")
print("=" * 80)

print(
    f"Deepfake samples missed by model: "
    f"{len(fn):,}"
)

if len(fn) > 0:

    print("\nFalse negatives by video/audio modification:")

    if (
        "modify_video" in fn.columns and
        "modify_audio" in fn.columns
    ):

        fn_modality = (
            fn.groupby(
                ["modify_video", "modify_audio"],
                dropna=False
            )
            .size()
            .reset_index(name="false_negatives")
        )

        print(
            fn_modality.to_string(index=False)
        )

        fn_modality.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "false_negative_by_modality.csv"
            ),
            index=False
        )

    if "n_fakes" in fn.columns:

        print("\nFalse negatives by number of fake segments:")

        fn_segments = (
            fn.groupby("n_fakes")
            .size()
            .reset_index(
                name="false_negatives"
            )
        )

        print(
            fn_segments.to_string(index=False)
        )

        fn_segments.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "false_negative_by_fake_segments.csv"
            ),
            index=False
        )


# ============================================================
# FALSE POSITIVE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FALSE POSITIVE ANALYSIS")
print("=" * 80)

print(
    f"Real samples incorrectly classified as fake: "
    f"{len(fp):,}"
)

if len(fp) > 0:

    if (
        "modify_video" in fp.columns and
        "modify_audio" in fp.columns
    ):

        fp_modality = (
            fp.groupby(
                ["modify_video", "modify_audio"],
                dropna=False
            )
            .size()
            .reset_index(name="false_positives")
        )

        print(
            fp_modality.to_string(index=False)
        )

        fp_modality.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "false_positive_by_modality.csv"
            ),
            index=False
        )


# ============================================================
# SAVE ALL ERRORS
# ============================================================

all_errors = df[
    ~df["correct"]
].copy()

all_errors.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_errors.csv"
    ),
    index=False
)


# ============================================================
# CREATE PLOTS
# ============================================================

print("\n[4] Creating research plots...")


# ------------------------------------------------------------
# Plot 1: Error distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

error_counts = [
    true_negative,
    false_positive,
    false_negative,
    true_positive
]

error_labels = [
    "True Negative",
    "False Positive",
    "False Negative",
    "True Positive"
]

plt.bar(
    error_labels,
    error_counts
)

plt.ylabel("Number of Samples")
plt.title("LAV-DF Test Confusion Distribution")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_distribution.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Plot 2: Duration vs accuracy
# ------------------------------------------------------------

if "duration" in df.columns:

    plt.figure(figsize=(9, 5))

    plt.plot(
        duration_df["duration_group"].astype(str),
        duration_df["accuracy"] * 100,
        marker="o"
    )

    plt.xlabel("Video Duration")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy by Video Duration")
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "accuracy_by_duration.png"
        ),
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# Plot 3: Modality accuracy
# ------------------------------------------------------------

if "modality_df" in locals():

    plt.figure(figsize=(9, 5))

    plt.bar(
        modality_df["modality"],
        modality_df["accuracy"] * 100
    )

    plt.xlabel("Manipulation Modality")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Multimodal Transformer Accuracy by Modality"
    )

    plt.xticks(rotation=20)
    plt.ylim(
        max(0, modality_df["accuracy"].min() * 100 - 5),
        100
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "accuracy_by_modality.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# CREATE JSON SUMMARY
# ============================================================

summary = {
    "experiment": {
        "model": "Multimodal Transformer",
        "dataset": "LAV-DF",
        "training_samples": 25000,
        "epochs": 2,
        "test_samples": total
    },

    "metrics": {
        "accuracy": accuracy,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "false_positive_rate": fp_rate,
        "false_negative_rate": fn_rate
    },

    "prediction_file": PREDICTIONS_FILE,
    "metadata_file": METADATA_FILE,

    "outputs": {
        "false_positives": "false_positives.csv",
        "false_negatives": "false_negatives.csv",
        "all_errors": "all_errors.csv",
        "error_distribution": "error_distribution.csv",
        "error_by_modality": "error_by_modality.csv",
        "error_by_fake_segments": "error_by_fake_segments.csv",
        "error_by_duration": "error_by_duration.csv",
        "highest_confidence_errors":
            "highest_confidence_errors.csv",
        "uncertain_predictions":
            "uncertain_predictions.csv"
    }
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "error_analysis_summary.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("ERROR ANALYSIS COMPLETED")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nImportant files:")

print("  false_negatives.csv")
print("  false_positives.csv")
print("  all_errors.csv")
print("  error_by_modality.csv")
print("  error_by_fake_segments.csv")
print("  error_by_duration.csv")
print("  highest_confidence_errors.csv")
print("  highest_confidence_false_negatives.csv")
print("  highest_confidence_false_positives.csv")
print("  error_by_fake_period_count.csv")
print("  uncertain_predictions.csv")
print("  error_analysis_summary.json")

print("\nPlots:")

print("  confusion_distribution.png")
print("  accuracy_by_duration.png")
print("  accuracy_by_modality.png")

print("\n" + "=" * 80)