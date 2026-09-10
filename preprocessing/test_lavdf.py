import cv2
from pathlib import Path


# ============================================================
# FIND ONE LAV-DF VIDEO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "LAV-DF"
    / "LAV-DF"
    / "test"
    / "000000.mp4"
)


print("=" * 60)
print("LAV-DF VIDEO TEST")
print("=" * 60)

print(f"\nVideo:")
print(VIDEO_PATH)


# ============================================================
# CHECK FILE
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"\nVideo not found:\n{VIDEO_PATH}"
    )

print("\nVideo file exists.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("OpenCV could not open the video.")

print("OpenCV opened the video successfully.")


# ============================================================
# VIDEO INFORMATION
# ============================================================

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

duration = frame_count / fps if fps > 0 else 0


print("\nVideo information:")
print(f"Frames   : {frame_count}")
print(f"FPS      : {fps:.2f}")
print(f"Resolution: {width} x {height}")
print(f"Duration : {duration:.2f} seconds")


# ============================================================
# READ FIRST FRAME
# ============================================================

success, frame = cap.read()

if not success:
    cap.release()
    raise RuntimeError("Could not read the first frame.")

print("\nFirst frame read successfully.")

print(f"Frame shape: {frame.shape}")


# ============================================================
# SAVE TEST FRAME
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "results" / "test"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_IMAGE = OUTPUT_DIR / "test_frame.jpg"

cv2.imwrite(
    str(OUTPUT_IMAGE),
    frame
)

print(f"\nTest frame saved to:")
print(OUTPUT_IMAGE)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

print("\n" + "=" * 60)
print("VIDEO TEST SUCCESSFUL")
print("=" * 60)