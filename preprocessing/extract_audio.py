import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "LAV-DF"
    / "LAV-DF"
    / "test"
    / "000000.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_PATH = OUTPUT_DIR / "test_audio.wav"


print("=" * 60)
print("LAV-DF AUDIO TEST")
print("=" * 60)

print(f"\nInput video:")
print(VIDEO_PATH)

print(f"\nOutput audio:")
print(AUDIO_PATH)


if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )


# Extract audio using FFmpeg
command = [
    "ffmpeg",
    "-y",
    "-i", str(VIDEO_PATH),
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    str(AUDIO_PATH),
]


print("\nExtracting audio...")

result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)


if result.returncode != 0:
    print("\nFFmpeg error:")
    print(result.stderr)
    raise RuntimeError("Audio extraction failed.")


if not AUDIO_PATH.exists():
    raise RuntimeError(
        "FFmpeg finished but audio file was not created."
    )


file_size = AUDIO_PATH.stat().st_size

print("\nAudio extraction successful.")

print(f"Audio file size: {file_size / 1024:.2f} KB")

print("\nAudio settings:")
print("Channels : 1 (mono)")
print("Sample rate: 16000 Hz")
print("Format   : WAV PCM")


print("\n" + "=" * 60)
print("AUDIO TEST SUCCESSFUL")
print("=" * 60)