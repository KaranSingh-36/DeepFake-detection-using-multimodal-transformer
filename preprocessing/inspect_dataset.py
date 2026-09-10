from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("../datasets/FakeAVCeleb")


# ============================================================
# VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# FIND VIDEOS
# ============================================================

def find_videos(dataset_dir):
    videos = []

    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return videos


# ============================================================
# PRINT DIRECTORY STRUCTURE
# ============================================================

def print_structure(dataset_dir, max_depth=3):

    print("\n" + "=" * 70)
    print("DATASET DIRECTORY STRUCTURE")
    print("=" * 70)

    dataset_dir = dataset_dir.resolve()

    for path in sorted(dataset_dir.rglob("*")):

        try:
            relative = path.relative_to(dataset_dir)
        except ValueError:
            continue

        depth = len(relative.parts)

        if depth <= max_depth:

            prefix = "    " * (depth - 1)

            if path.is_dir():
                print(f"{prefix}[DIR]  {path.name}")
            else:
                print(f"{prefix}[FILE] {path.name}")


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATASET_DIR.exists():

        print("ERROR: Dataset directory not found.")
        print()
        print("Expected location:")
        print(DATASET_DIR.resolve())
        print()

        return

    print("=" * 70)
    print("FakeAVCeleb Dataset Inspector")
    print("=" * 70)

    print(f"\nDataset location:")
    print(DATASET_DIR.resolve())

    # --------------------------------------------------------
    # DIRECTORY STRUCTURE
    # --------------------------------------------------------

    print_structure(DATASET_DIR)

    # --------------------------------------------------------
    # FIND VIDEOS
    # --------------------------------------------------------

    print("\nSearching for videos...")

    videos = find_videos(DATASET_DIR)

    print(f"\nTotal video files found: {len(videos)}")

    # --------------------------------------------------------
    # FILE EXTENSIONS
    # --------------------------------------------------------

    extension_counter = Counter(
        video.suffix.lower()
        for video in videos
    )

    print("\nVideo extensions:")

    for extension, count in extension_counter.items():
        print(f"  {extension}: {count}")

    # --------------------------------------------------------
    # SHOW FIRST 20 VIDEOS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST 20 VIDEO FILES")
    print("=" * 70)

    for index, video in enumerate(videos[:20], start=1):

        relative_path = video.relative_to(DATASET_DIR)

        print(f"{index:02d}. {relative_path}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()