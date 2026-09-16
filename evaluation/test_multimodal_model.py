import os
import sys

# Make project root available
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import torch

from models.multimodel_model import (
    MultimodalCrossAttentionModel
)


def main():

    print("=" * 60)
    print("MULTIMODAL CROSS-ATTENTION MODEL TEST")
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
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM: {total_memory:.2f} GB"
        )

    # ------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------

    print("\nLoading multimodal model...")

    model = MultimodalCrossAttentionModel()

    model = model.to(device)

    print("Model created successfully.")

    # ------------------------------------------------------------
    # Dummy input
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

    # 4 seconds of 16 kHz audio
    audio = torch.randn(
        batch_size,
        64000,
        device=device
    )

    print("\nInput shapes:")

    print(
        f"Video: {video.shape}"
    )

    print(
        f"Audio: {audio.shape}"
    )

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    print("\nRunning forward pass...")

    with torch.no_grad():

        output = model(
            video,
            audio,
            return_attention=True
        )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("\nOutput information:")

    print(
        f"Logits: "
        f"{output['logits'].shape}"
    )

    print(
        f"Visual tokens: "
        f"{output['visual_tokens'].shape}"
    )

    print(
        f"Audio tokens: "
        f"{output['audio_tokens'].shape}"
    )

    print(
        f"Fused embedding: "
        f"{output['fused_embedding'].shape}"
    )

    if output["visual_attention"] is not None:

        print(
            f"Visual attention: "
            f"{output['visual_attention'].shape}"
        )

    if output["audio_attention"] is not None:

        print(
            f"Audio attention: "
            f"{output['audio_attention'].shape}"
        )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    probabilities = torch.softmax(
        output["logits"],
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    )

    print("\nPrediction:")

    print(
        f"Logits: {output['logits']}"
    )

    print(
        f"Probabilities: {probabilities}"
    )

    print(
        f"Predicted class: {prediction.item()}"
    )

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

        print("\nGPU memory:")

        print(
            f"Allocated: {allocated:.2f} MB"
        )

        print(
            f"Reserved: {reserved:.2f} MB"
        )

    print("\n" + "=" * 60)
    print("MULTIMODAL MODEL TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()