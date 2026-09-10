import torch
import torch.nn as nn
from transformers import ViTModel


class VideoDeepfakeDetector(nn.Module):
    """
    Video-only deepfake detector.

    Input:
        frames: [B, T, 3, 224, 224]

    Output:
        logits: [B, 2]

    B = batch size
    T = number of frames
    """

    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        num_classes=2,
        freeze_vit=True,
        dropout=0.2,
    ):
        super().__init__()

        # --------------------------------------------------
        # Pretrained Vision Transformer
        # --------------------------------------------------

        self.vit = ViTModel.from_pretrained(
            model_name
        )

        hidden_size = self.vit.config.hidden_size

        # --------------------------------------------------
        # Freeze ViT initially
        # --------------------------------------------------

        if freeze_vit:
            for parameter in self.vit.parameters():
                parameter.requires_grad = False

        # --------------------------------------------------
        # Temporal attention
        # --------------------------------------------------

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),

            nn.Linear(
                hidden_size,
                512
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, frames):

        # --------------------------------------------------
        # Input:
        #
        # [B, T, C, H, W]
        # --------------------------------------------------

        batch_size, num_frames, channels, height, width = (
            frames.shape
        )

        # --------------------------------------------------
        # Process every frame through ViT
        #
        # [B, T, C, H, W]
        #
        # becomes
        #
        # [B*T, C, H, W]
        # --------------------------------------------------

        frames = frames.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        # --------------------------------------------------
        # ViT
        # --------------------------------------------------

        outputs = self.vit(
            pixel_values=frames
        )

        # ViT CLS token
        #
        # [B*T, sequence, hidden]
        #
        # -> [B*T, hidden]
        # --------------------------------------------------

        frame_embeddings = outputs.last_hidden_state[:, 0]

        # --------------------------------------------------
        # Restore temporal dimension
        #
        # [B*T, hidden]
        #
        # -> [B, T, hidden]
        # --------------------------------------------------

        frame_embeddings = frame_embeddings.reshape(
            batch_size,
            num_frames,
            -1
        )

        # --------------------------------------------------
        # Temporal self-attention
        # --------------------------------------------------

        temporal_features, _ = self.temporal_attention(
            frame_embeddings,
            frame_embeddings,
            frame_embeddings
        )

        # --------------------------------------------------
        # Temporal average pooling
        # --------------------------------------------------

        video_embedding = temporal_features.mean(
            dim=1
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.classifier(
            video_embedding
        )

        return logits


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VIDEO MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ------------------------------------------------------
    # Create model
    # ------------------------------------------------------

    model = VideoDeepfakeDetector(
        freeze_vit=True
    )

    model = model.to(device)

    print()
    print("Model created successfully.")

    # ------------------------------------------------------
    # Fake test batch
    # ------------------------------------------------------

    test_frames = torch.randn(
        2,
        8,
        3,
        224,
        224,
        device=device
    )

    print()
    print("Input shape:")
    print(test_frames.shape)

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------

    print()
    print("Running forward pass...")

    with torch.no_grad():

        logits = model(
            test_frames
        )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print()
    print("Output shape:")
    print(logits.shape)

    print()
    print("Logits:")
    print(logits)

    predictions = logits.argmax(
        dim=1
    )

    print()
    print("Predictions:")
    print(predictions)

    print()
    print("=" * 60)
    print("VIDEO MODEL TEST SUCCESSFUL")
    print("=" * 60)