import torch
import torch.nn as nn
from transformers import ViTModel, Wav2Vec2Model


class MultimodalCrossAttentionModel(nn.Module):
    """
    Multimodal Deepfake Detector

    Visual branch:
        Video frames -> ViT

    Audio branch:
        Audio waveform -> Wav2Vec2

    Fusion:
        Bidirectional cross-attention between
        visual and audio representations

    Output:
        2 classes -> REAL / DEEPFAKE
    """

    def __init__(
        self,
        video_model_name="google/vit-base-patch16-224",
        audio_model_name="facebook/wav2vec2-base",
        num_classes=2,
        num_attention_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        # ============================================================
        # 1. PRETRAINED VISUAL ENCODER
        # ============================================================

        self.video_encoder = ViTModel.from_pretrained(
            video_model_name
        )

        video_dim = self.video_encoder.config.hidden_size

        # ============================================================
        # 2. PRETRAINED AUDIO ENCODER
        # ============================================================

        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            audio_model_name
        )

        audio_dim = self.audio_encoder.config.hidden_size

        # ============================================================
        # 3. COMMON FUSION DIMENSION
        # ============================================================

        # ViT-Base = 768
        # Wav2Vec2-Base = 768
        #
        # Keeping a common 768-dimensional representation
        # allows direct multimodal attention.

        fusion_dim = 768

        self.video_projection = nn.Linear(
            video_dim,
            fusion_dim
        )

        self.audio_projection = nn.Linear(
            audio_dim,
            fusion_dim
        )

        # ============================================================
        # 4. VISUAL -> AUDIO CROSS ATTENTION
        # ============================================================

        self.visual_to_audio = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # ============================================================
        # 5. AUDIO -> VISUAL CROSS ATTENTION
        # ============================================================

        self.audio_to_visual = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # ============================================================
        # 6. NORMALIZATION
        # ============================================================

        self.visual_norm = nn.LayerNorm(fusion_dim)
        self.audio_norm = nn.LayerNorm(fusion_dim)

        # ============================================================
        # 7. FEED-FORWARD TRANSFORMATION
        # ============================================================

        self.visual_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

        self.audio_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

        self.visual_ffn_norm = nn.LayerNorm(fusion_dim)
        self.audio_ffn_norm = nn.LayerNorm(fusion_dim)

        # ============================================================
        # 8. MULTIMODAL FUSION
        # ============================================================

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ============================================================
        # 9. CLASSIFIER
        # ============================================================

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def encode_video(self, video_frames):
        """
        video_frames:
            [B, T, 3, 224, 224]

        Returns:
            visual_tokens:
            [B, T, 768]
        """

        batch_size, num_frames, channels, height, width = (
            video_frames.shape
        )

        # Combine batch and temporal dimensions
        frames = video_frames.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        # ViT processing
        video_output = self.video_encoder(
            pixel_values=frames
        )

        # CLS token
        cls_features = video_output.last_hidden_state[:, 0]

        # Restore temporal dimension
        visual_tokens = cls_features.reshape(
            batch_size,
            num_frames,
            -1
        )

        visual_tokens = self.video_projection(
            visual_tokens
        )

        return visual_tokens

    def encode_audio(self, audio_waveform):
        """
        audio_waveform:
            [B, samples]

        Returns:
            audio_tokens:
            [B, audio_sequence_length, 768]
        """

        audio_output = self.audio_encoder(
            input_values=audio_waveform
        )

        audio_tokens = audio_output.last_hidden_state

        audio_tokens = self.audio_projection(
            audio_tokens
        )

        return audio_tokens

    def forward(
        self,
        video_frames,
        audio_waveform,
        return_attention=False
    ):
        """
        Parameters
        ----------
        video_frames:
            [B, T, 3, 224, 224]

        audio_waveform:
            [B, samples]

        return_attention:
            Whether to return cross-attention weights.

        Returns
        -------
        logits:
            [B, 2]

        optionally:
            attention information
        """

        # ============================================================
        # VIDEO ENCODER
        # ============================================================

        visual_tokens = self.encode_video(
            video_frames
        )

        # ============================================================
        # AUDIO ENCODER
        # ============================================================

        audio_tokens = self.encode_audio(
            audio_waveform
        )

        # ============================================================
        # VISUAL -> AUDIO CROSS ATTENTION
        #
        # Query  = visual information
        # Key    = audio information
        # Value  = audio information
        # ============================================================

        visual_attended, visual_attention = (
            self.visual_to_audio(
                query=visual_tokens,
                key=audio_tokens,
                value=audio_tokens,
                need_weights=return_attention
            )
        )

        visual_tokens = self.visual_norm(
            visual_tokens + visual_attended
        )

        # ============================================================
        # AUDIO -> VISUAL CROSS ATTENTION
        #
        # Query  = audio information
        # Key    = visual information
        # Value  = visual information
        # ============================================================

        audio_attended, audio_attention = (
            self.audio_to_visual(
                query=audio_tokens,
                key=visual_tokens,
                value=visual_tokens,
                need_weights=return_attention
            )
        )

        audio_tokens = self.audio_norm(
            audio_tokens + audio_attended
        )

        # ============================================================
        # FEED-FORWARD TRANSFORMATIONS
        # ============================================================

        visual_ffn_output = self.visual_ffn(
            visual_tokens
        )

        visual_tokens = self.visual_ffn_norm(
            visual_tokens + visual_ffn_output
        )

        audio_ffn_output = self.audio_ffn(
            audio_tokens
        )

        audio_tokens = self.audio_ffn_norm(
            audio_tokens + audio_ffn_output
        )

        # ============================================================
        # TEMPORAL / SEQUENCE POOLING
        # ============================================================

        visual_embedding = visual_tokens.mean(
            dim=1
        )

        audio_embedding = audio_tokens.mean(
            dim=1
        )

        # ============================================================
        # MULTIMODAL FUSION
        # ============================================================

        fused = torch.cat(
            [
                visual_embedding,
                audio_embedding
            ],
            dim=-1
        )

        fused = self.fusion(fused)

        # ============================================================
        # CLASSIFICATION
        # ============================================================

        logits = self.classifier(
            fused
        )

        if return_attention:
            return {
                "logits": logits,
                "visual_tokens": visual_tokens,
                "audio_tokens": audio_tokens,
                "visual_attention": visual_attention,
                "audio_attention": audio_attention,
                "fused_embedding": fused
            }

        return logits