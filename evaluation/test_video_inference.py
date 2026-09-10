import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from dataset.lavdf_dataset import LAVDFDataset
from models.video_model import VideoDeepfakeDetector

def main():

    print("=" * 70)
    print("REAL LAV-DF VIDEO INFERENCE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = LAVDFDataset(
        split="test",
        num_frames=8,
        image_size=224,
        sample_rate=16000,
        audio_duration=4.0,
    )

    # Only test 4 videos
    subset = torch.utils.data.Subset(
        dataset,
        range(min(4, len(dataset)))
    )

    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print()
    print("Loading video model...")

    model = VideoDeepfakeDetector(
        freeze_vit=True
    )

    model = model.to(device)
    model.eval()

    print("Model ready.")

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    print()
    print("Running inference...")
    print()

    with torch.no_grad():

        for batch_number, batch in enumerate(dataloader):

            frames = batch["frames"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )

            files = batch["file"]

            # Forward pass
            logits = model(frames)

            # Convert logits to probabilities
            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            # -------------------------------------------------
            # Print each video
            # -------------------------------------------------

            for i in range(len(files)):

                predicted_label = predictions[i].item()
                actual_label = labels[i].item()

                real_probability = (
                    probabilities[i][0].item()
                )

                fake_probability = (
                    probabilities[i][1].item()
                )

                predicted_text = (
                    "DEEPFAKE"
                    if predicted_label == 1
                    else "REAL"
                )

                actual_text = (
                    "DEEPFAKE"
                    if actual_label == 1
                    else "REAL"
                )

                print("File:", files[i])
                print(
                    f"Actual    : {actual_text}"
                )
                print(
                    f"Prediction: {predicted_text}"
                )
                print(
                    f"Real probability: "
                    f"{real_probability:.4f}"
                )
                print(
                    f"Fake probability: "
                    f"{fake_probability:.4f}"
                )
                print("-" * 50)

    print()
    print("=" * 70)
    print("REAL VIDEO INFERENCE TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()