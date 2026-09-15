"""
Audio Encoder for Multimodal Deepfake Detection

Uses a pretrained Wav2Vec2 encoder to extract
speech/audio representations from LAV-DF audio.

Input:
    audio waveform -> [B, T]
    T = 64000 samples for approximately 4 seconds at 16 kHz

Output:
    audio sequence -> [B, T', 768]
    pooled audio embedding -> [B, 768]
    classification logits -> [B, 2]
"""

import torch
import torch.nn as nn

from transformers import AutoFeatureExtractor, Wav2Vec2Model


class AudioModel(nn.Module):
    """
    Pretrained Wav2Vec2 audio encoder + classification head.

    The Wav2Vec2 sequence representation is retained because
    the final multimodal model will use it for cross-attention.
    """

    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        num_classes=2,
        freeze_encoder=False,
    ):
        super().__init__()

        print(f"Loading audio model: {model_name}")

        # Audio preprocessing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name
        )

        # Pretrained Wav2Vec2 encoder
        self.encoder = Wav2Vec2Model.from_pretrained(
            model_name
        )

        self.hidden_size = self.encoder.config.hidden_size

        # Optional freezing for initial experiments
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )

        print(f"Audio hidden size: {self.hidden_size}")
        print("Audio model created successfully.")

    def forward(self, audio):
        """
        Parameters
        ----------
        audio : torch.Tensor
            Shape [B, T]
            Raw waveform sampled at 16 kHz.

        Returns
        -------
        logits : torch.Tensor
            Shape [B, num_classes]

        sequence_output : torch.Tensor
            Shape [B, T', hidden_size]

        pooled_output : torch.Tensor
            Shape [B, hidden_size]
        """

        # Ensure correct floating-point type
        audio = audio.float()

        # Wav2Vec2 feature extraction
        outputs = self.encoder(
            input_values=audio
        )

        # Full temporal audio representation
        sequence_output = outputs.last_hidden_state

        # Temporal mean pooling
        pooled_output = sequence_output.mean(dim=1)

        # Classification
        logits = self.classifier(pooled_output)

        return logits, sequence_output, pooled_output


def test_audio_model():
    """
    Basic CUDA/model test.

    This does NOT train the model.
    It verifies that:
      1. Wav2Vec2 downloads correctly
      2. The model loads
      3. Audio can be processed
      4. CUDA works
      5. Output dimensions are correct
    """

    print("=" * 70)
    print("AUDIO MODEL TEST")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    print("\nCreating audio model...")

    model = AudioModel(
        model_name="facebook/wav2vec2-base",
        num_classes=2,
        freeze_encoder=False,
    )

    model = model.to(device)
    model.eval()

    # Simulate one LAV-DF audio sample.
    # LAV-DF audio is 16 kHz.
    # 64000 samples ≈ 4 seconds.
    print("\nCreating test audio...")

    audio = torch.randn(
        1,
        64000,
        dtype=torch.float32,
        device=device,
    )

    print(f"Input shape: {audio.shape}")

    print("\nRunning forward pass...")

    with torch.no_grad():
        logits, sequence, pooled = model(audio)

    print("\nOutput shapes:")
    print(f"Logits:   {logits.shape}")
    print(f"Sequence: {sequence.shape}")
    print(f"Pooled:   {pooled.shape}")

    predictions = torch.argmax(logits, dim=1)

    print(f"\nLogits:\n{logits}")
    print(f"\nPrediction: {predictions}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2

        print("\nGPU memory:")
        print(f"Allocated: {allocated:.2f} MB")
        print(f"Reserved:  {reserved:.2f} MB")

    print("\n" + "=" * 70)
    print("AUDIO MODEL TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    test_audio_model()  