import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from models.multimodel_model import (
    MultimodalCrossAttentionModel
)


def main():

    print("=" * 60)
    print("MULTIMODAL BACKWARD PASS TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM: {total_memory:.2f} GB"
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------

    print("\nLoading model...")

    model = MultimodalCrossAttentionModel()

    model = model.to(device)

    print("Model loaded successfully.")

    # ------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------

    model.train()

    # ------------------------------------------------------------
    # Test input
    # ------------------------------------------------------------

    batch_size = 1
    num_frames = 8

    video = torch.randn(
        batch_size,
        num_frames,
        3,
        224,
        224,
        device=device
    )

    audio = torch.randn(
        batch_size,
        64000,
        device=device
    )

    labels = torch.tensor(
        [1],
        dtype=torch.long,
        device=device
    )

    print("\nInput shapes:")

    print(f"Video:  {video.shape}")
    print(f"Audio:  {audio.shape}")
    print(f"Labels: {labels.shape}")

    # ------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5
    )

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    print("\nRunning forward pass...")

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        video,
        audio
    )

    print(
        f"Logits: {logits.shape}"
    )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    loss = criterion(
        logits,
        labels
    )

    print(
        f"Loss: {loss.item():.6f}"
    )

    # ------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------

    print("\nRunning backward pass...")

    loss.backward()

    print("Backward pass completed successfully.")

    # ------------------------------------------------------------
    # Check gradients
    # ------------------------------------------------------------

    gradient_count = 0
    parameters_with_gradients = 0

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            parameters_with_gradients += 1

            gradient_count += (
                parameter.grad.abs().sum().item()
            )

    print(
        f"\nParameters with gradients: "
        f"{parameters_with_gradients}"
    )

    print(
        f"Total gradient magnitude: "
        f"{gradient_count:.6f}"
    )

    # ------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------

    print("\nRunning optimizer step...")

    optimizer.step()

    print("Optimizer step completed successfully.")

    # ------------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        )

        peak = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print("\nGPU memory:")

        print(
            f"Current allocated: "
            f"{allocated:.2f} MB"
        )

        print(
            f"Current reserved: "
            f"{reserved:.2f} MB"
        )

        print(
            f"Peak allocated: "
            f"{peak:.2f} MB"
        )

    print("\n" + "=" * 60)
    print("MULTIMODAL BACKWARD PASS TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()