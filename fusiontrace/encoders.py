"""Pretrained ViT and Wav2Vec2 encoders producing the per-step features that
the fusion model consumes. The same module is used to build the feature cache
(fully frozen) and for end-to-end fine-tuning (top blocks unfrozen).

Visual feature per frame : [CLS token ; mean of patch tokens] per visual stream
                           ("face", "mouth")                     -> 2 * hidden * streams
Audio feature per step   : hidden states of the chosen layers   -> [L, hidden]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, Wav2Vec2Model
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
try:
    hf_logging.disable_progress_bar()
except Exception:  # pragma: no cover
    pass

try:
    from transformers import AutoImageProcessor
except ImportError:  # pragma: no cover
    AutoImageProcessor = None

WAV2VEC2_HOP = 320          # samples between consecutive Wav2Vec2 frames at 16 kHz
WAV2VEC2_RATE = 16000 / WAV2VEC2_HOP   # 50 Hz


def _set_trainable_blocks(blocks, final_norm, count):
    for block in blocks:
        for parameter in block.parameters():
            parameter.requires_grad = False
    if count > 0:
        for block in blocks[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        if final_norm is not None:
            for parameter in final_norm.parameters():
                parameter.requires_grad = True


class FrozenEncoders(nn.Module):
    def __init__(
        self,
        video_model_name="google/vit-base-patch16-224",
        audio_model_name="facebook/wav2vec2-base",
        audio_layers=(6, 12),
        audio_pool=2,
        visual_streams=("face", "mouth"),
        trainable_video_blocks=0,
        trainable_audio_blocks=0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.video_model_name = video_model_name
        self.audio_model_name = audio_model_name
        self.audio_layers = [int(layer) for layer in audio_layers]
        self.audio_pool = int(audio_pool)
        self.visual_streams = [str(name) for name in visual_streams]
        if not self.visual_streams:
            raise ValueError("visual_streams must name at least one stream")
        self.trainable_video_blocks = int(trainable_video_blocks)
        self.trainable_audio_blocks = int(trainable_audio_blocks)

        self.vit = ViTModel.from_pretrained(video_model_name, add_pooling_layer=False).float()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(audio_model_name).float()

        # No stochastic regularisation inside the pretrained audio encoder.
        self.wav2vec2.config.mask_time_prob = 0.0
        self.wav2vec2.config.mask_feature_prob = 0.0
        self.wav2vec2.config.layerdrop = 0.0
        self.wav2vec2.config.apply_spec_augment = False

        depth = len(self.wav2vec2.encoder.layers)
        for layer in self.audio_layers:
            if layer < 0 or layer > depth:
                raise ValueError(f"audio layer {layer} out of range 0..{depth}")

        mean, std = self._image_stats(video_model_name)
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        for parameter in self.parameters():
            parameter.requires_grad = False
        _set_trainable_blocks(self.vit_blocks, self.vit.layernorm, self.trainable_video_blocks)
        _set_trainable_blocks(self.wav2vec2.encoder.layers, None, self.trainable_audio_blocks)

        if gradient_checkpointing:
            if self.trainable_video_blocks > 0:
                self.vit.gradient_checkpointing_enable()
            if self.trainable_audio_blocks > 0:
                self.wav2vec2.gradient_checkpointing_enable()

    @property
    def vit_blocks(self):
        # transformers >= 5 exposes ViT blocks as `layers`; 4.x nests them under `encoder.layer`.
        if hasattr(self.vit, "layers"):
            return self.vit.layers
        return self.vit.encoder.layer

    @staticmethod
    def _image_stats(model_name):
        if AutoImageProcessor is not None:
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                return list(processor.image_mean), list(processor.image_std)
            except Exception:
                pass
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # ------------------------------------------------------------------ dims
    @property
    def visual_dim(self):
        return 2 * self.vit.config.hidden_size * len(self.visual_streams)

    @property
    def audio_dim(self):
        return self.wav2vec2.config.hidden_size

    @property
    def audio_hz(self):
        return WAV2VEC2_RATE / self.audio_pool

    @property
    def is_trainable(self):
        return self.trainable_video_blocks > 0 or self.trainable_audio_blocks > 0

    def train(self, mode=True):
        # Frozen parts stay in eval mode (no dropout) even while fine-tuning.
        super().train(mode)
        if self.trainable_video_blocks == 0:
            self.vit.eval()
        if self.trainable_audio_blocks == 0:
            self.wav2vec2.eval()
        return self

    # ------------------------------------------------------------- preprocess
    def preprocess_frames(self, frames_uint8):
        """numpy/tensor uint8 [N, H, W, 3] -> float tensor [N, 3, H, W] normalised."""
        if not torch.is_tensor(frames_uint8):
            frames_uint8 = torch.from_numpy(frames_uint8)
        x = frames_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        x = x.to(self.image_mean.device)
        return (x - self.image_mean) / self.image_std

    @staticmethod
    def normalize_waveform(wave):
        """Per-utterance zero mean / unit variance, as the Wav2Vec2 processor does."""
        mean = wave.mean(dim=-1, keepdim=True)
        var = wave.var(dim=-1, keepdim=True, unbiased=False)
        return (wave - mean) / torch.sqrt(var + 1e-7)

    # ------------------------------------------------------------------ video
    def encode_frames(self, pixel_values, chunk_size=64):
        """[N, 3, H, W] -> [N, 2 * hidden]."""
        outputs = []
        grad = torch.is_grad_enabled() and self.trainable_video_blocks > 0
        with torch.set_grad_enabled(grad):
            for start in range(0, pixel_values.shape[0], chunk_size):
                chunk = pixel_values[start:start + chunk_size]
                hidden = self.vit(pixel_values=chunk).last_hidden_state
                cls = hidden[:, 0]
                patches = hidden[:, 1:].mean(dim=1)
                outputs.append(torch.cat([cls, patches], dim=-1))
        return torch.cat(outputs, dim=0)

    def encode_visual(self, streams):
        """{"face": uint8 [N, S, S, 3], "mouth": ...} -> [N, visual_dim].

        Streams are encoded separately with the same ViT and concatenated in the
        order of `self.visual_streams`.
        """
        parts = []
        for name in self.visual_streams:
            if name not in streams:
                raise KeyError(f"visual stream {name!r} missing from the decoded clip")
            frames = streams[name]
            if not torch.is_tensor(frames):
                frames = torch.from_numpy(np.ascontiguousarray(frames))
            parts.append(self.encode_frames(self.preprocess_frames(frames.to(self.image_mean.device))))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ audio
    def encode_audio(self, waveform):
        """[B, S] raw waveform -> [B, L, T_audio, hidden] at `audio_hz`."""
        stacked = self._encode_audio_raw(waveform)
        if self.audio_pool > 1:
            b, l, t, d = stacked.shape
            usable = (t // self.audio_pool) * self.audio_pool
            stacked = stacked[:, :, :usable].reshape(b, l, usable // self.audio_pool, self.audio_pool, d).mean(dim=3)
        return stacked

    def encode_audio_long(self, waveform, chunk_seconds=4.0, hop_seconds=3.2, batch_chunks=8):
        """Whole-clip audio [S] -> [L, T_audio, hidden] with bounded temporal context.

        Wav2Vec2 attends over everything it is given, so a fake segment at 5 s
        would otherwise leak into the features at 1 s. The clip is encoded in
        `chunk_seconds` chunks with `hop_seconds` hop and only the central
        tokens of each chunk are kept, so every token sees at most one chunk.
        """
        rate = WAV2VEC2_RATE
        hop_tokens = int(round(hop_seconds * rate))
        chunk_tokens = int(round(chunk_seconds * rate))
        margin = (chunk_tokens - hop_tokens) // 2
        chunk_samples = (chunk_tokens - 1) * WAV2VEC2_HOP + 400
        hop_samples = hop_tokens * WAV2VEC2_HOP

        wave = waveform.float().flatten()
        total_tokens = max(1, (wave.numel() - 400) // WAV2VEC2_HOP + 1) if wave.numel() >= 400 else 1
        if wave.numel() < 400:
            wave = F.pad(wave, (0, 400 - wave.numel()))

        starts = list(range(0, max(1, wave.numel() - margin * WAV2VEC2_HOP), hop_samples))
        pieces = []
        full = [s for s in starts if s + chunk_samples <= wave.numel()]
        partial = [s for s in starts if s + chunk_samples > wave.numel()]

        def keep(tokens, start_index, is_first, is_last):
            lo = 0 if is_first else margin
            hi = tokens.shape[2] if is_last else min(tokens.shape[2], margin + hop_tokens)
            return tokens[:, :, lo:hi], start_index + lo

        outputs = {}
        for b in range(0, len(full), batch_chunks):
            group = full[b:b + batch_chunks]
            batch = torch.stack([wave[s:s + chunk_samples] for s in group])
            feats = self._encode_audio_raw(batch)              # [B, L, T, D]
            for i, s in enumerate(group):
                outputs[s] = feats[i:i + 1]
        for s in partial:
            outputs[s] = self._encode_audio_raw(wave[s:].unsqueeze(0))

        pieces = []
        for n, s in enumerate(starts):
            tokens, _ = keep(outputs[s], s // WAV2VEC2_HOP, n == 0, n == len(starts) - 1)
            pieces.append(tokens)
        stacked = torch.cat(pieces, dim=2)[:, :, :total_tokens]   # [1, L, T, D]

        if self.audio_pool > 1:
            _, l, t, d = stacked.shape
            usable = (t // self.audio_pool) * self.audio_pool
            if usable == 0:
                stacked = F.pad(stacked, (0, 0, 0, self.audio_pool - t))
                usable = self.audio_pool
            stacked = stacked[:, :, :usable].reshape(1, l, usable // self.audio_pool, self.audio_pool, d).mean(dim=3)
        return stacked[0]

    def _encode_audio_raw(self, wave_batch):
        grad = torch.is_grad_enabled() and self.trainable_audio_blocks > 0
        with torch.set_grad_enabled(grad):
            wave = self.normalize_waveform(wave_batch.float())
            outputs = self.wav2vec2(input_values=wave, output_hidden_states=True)
            states = outputs.hidden_states
            return torch.stack([states[layer] for layer in self.audio_layers], dim=1)

    def manifest(self):
        return {
            "video_model_name": self.video_model_name,
            "audio_model_name": self.audio_model_name,
            "audio_layers": self.audio_layers,
            "audio_pool": self.audio_pool,
            "visual_streams": list(self.visual_streams),
            "visual_dim": self.visual_dim,
            "audio_dim": self.audio_dim,
            "audio_hz": self.audio_hz,
        }
