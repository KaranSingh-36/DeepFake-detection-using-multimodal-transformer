import json
from pathlib import Path
import pandas as pd


# ============================================================
# LAV-DF METADATA CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "LAV-DF"
    / "LAV-DF"
    / "metadata.min.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "lavdf_metadata.csv"
)


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("LAV-DF Metadata Creation")
print("=" * 60)

print(f"\nLoading metadata:")
print(METADATA_PATH)

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"\nMetadata file not found:\n{METADATA_PATH}"
    )

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nTotal entries: {len(data)}")


# ============================================================
# CONVERT METADATA
# ============================================================

rows = []

for item in data:

    file_path = item["file"]

    # LAV-DF labels
    modify_video = bool(item["modify_video"])
    modify_audio = bool(item["modify_audio"])

    # A video is considered DEEPFAKE if either
    # the video or audio has been manipulated.
    label = int(modify_video or modify_audio)

    rows.append(
        {
            "file": file_path,

            "split": item["split"],

            "label": label,

            "modify_video": int(modify_video),

            "modify_audio": int(modify_audio),

            "n_fakes": item["n_fakes"],

            "duration": item["duration"],

            "video_frames": item["video_frames"],

            "audio_channels": item["audio_channels"],

            "audio_frames": item["audio_frames"],

            "original": item["original"],

            "fake_periods": item["fake_periods"],
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("METADATA CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nOutput:")
print(OUTPUT_PATH)

print(f"\nTotal samples: {len(df)}")

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nVideo manipulation:")
print(df["modify_video"].value_counts())

print("\nAudio manipulation:")
print(df["modify_audio"].value_counts())

print("\nFirst 5 entries:")
print(df.head())

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)