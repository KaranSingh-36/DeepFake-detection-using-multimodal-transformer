import torch
from torch.utils.data import DataLoader

from lavdf_dataset import LAVDFDataset


def main():

    print("=" * 60)
    print("LAV-DF DATALOADER + GPU TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Select device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )
    else:
        print("WARNING: CUDA is not available.")

    # ---------------------------------------------------------
    # 2. Load a SMALL subset for testing
    # ---------------------------------------------------------

    dataset = LAVDFDataset(
        split="train",
        num_frames=8,
        image_size=224,
        sample_rate=16000,
        audio_duration=4.0,
    )

    # Only use 8 samples for this test
    indices = list(range(min(8, len(dataset))))

    subset = torch.utils.data.Subset(
        dataset,
        indices
    )

    print()
    print("Test samples:", len(subset))

    # ---------------------------------------------------------
    # 3. Create DataLoader
    # ---------------------------------------------------------

    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("Batch size: 2")

    # ---------------------------------------------------------
    # 4. Load one batch
    # ---------------------------------------------------------

    print()
    print("Loading first batch...")

    batch = next(iter(dataloader))

    print()
    print("CPU batch shapes:")
    print("Frames:", batch["frames"].shape)
    print("Audio :", batch["audio"].shape)
    print("Labels:", batch["label"].shape)

    # ---------------------------------------------------------
    # 5. Move batch to GPU
    # ---------------------------------------------------------

    print()
    print("Moving batch to GPU...")

    frames = batch["frames"].to(
        device,
        non_blocking=True
    )

    audio = batch["audio"].to(
        device,
        non_blocking=True
    )

    labels = batch["label"].to(
        device,
        non_blocking=True
    )

    # ---------------------------------------------------------
    # 6. Verify GPU tensors
    # ---------------------------------------------------------

    print()
    print("GPU batch:")
    print("Frames:", frames.shape)
    print("Audio :", audio.shape)
    print("Labels:", labels.shape)

    print()
    print("Tensor devices:")
    print("Frames device:", frames.device)
    print("Audio device :", audio.device)
    print("Labels device:", labels.device)

    # ---------------------------------------------------------
    # 7. GPU memory test
    # ---------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        )

        print()
        print("GPU memory:")
        print(
            f"Allocated: {allocated:.2f} MB"
        )
        print(
            f"Reserved : {reserved:.2f} MB"
        )

    # ---------------------------------------------------------
    # 8. Final result
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DATALOADER + GPU TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()