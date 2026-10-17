"""Turn a decoded clip into the cached feature format using FrozenEncoders."""

import numpy as np
import torch

from fusiontrace.media import probe_video, read_audio, read_frames


def decode_clip(video_path, visual_fps, image_size, face_crop, sample_rate=16000):
    """Decode everything the encoders need for a whole clip."""
    info = probe_video(video_path)
    frames, _ = read_frames(video_path, visual_fps, image_size=image_size, face_crop=face_crop)
    wave = read_audio(video_path, sample_rate)
    has_audio = wave.size > 0
    if not has_audio:
        wave = np.zeros(int(round(info["duration"] * sample_rate)), dtype=np.float32)
    return {
        "frames": frames,                 # uint8 [Tv, S, S, 3]
        "waveform": wave,                 # float32 [S]
        "duration": float(info["duration"]),
        "has_audio": has_audio,
    }


@torch.inference_mode()
def encode_clip(encoders, clip, visual_fps, device, amp=True, audio_chunk_seconds=4.0,
                audio_hop_seconds=3.2):
    """-> dict(visual float16 [Tv, Dv], audio float16 [L, Ta, Da], visual_fps, audio_hz, duration)."""
    use_amp = amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pixels = encoders.preprocess_frames(torch.as_tensor(clip["frames"]).to(device))
        visual = encoders.encode_frames(pixels)
        wave = torch.as_tensor(clip["waveform"]).to(device)
        audio = encoders.encode_audio_long(wave, chunk_seconds=audio_chunk_seconds,
                                           hop_seconds=audio_hop_seconds)
    return {
        "visual": visual.float().cpu().numpy().astype(np.float16),
        "audio": audio.float().cpu().numpy().astype(np.float16),
        "visual_fps": float(visual_fps),
        "audio_hz": float(encoders.audio_hz),
        "duration": float(clip["duration"]),
    }


def build_encoders(cfg, device, trainable_video_blocks=0, trainable_audio_blocks=0,
                   gradient_checkpointing=False):
    from fusiontrace.encoders import FrozenEncoders

    encoders = FrozenEncoders(
        video_model_name=cfg.encoders.video_model_name,
        audio_model_name=cfg.encoders.audio_model_name,
        audio_layers=cfg.data.audio_layers,
        audio_pool=cfg.data.audio_pool,
        trainable_video_blocks=trainable_video_blocks,
        trainable_audio_blocks=trainable_audio_blocks,
        gradient_checkpointing=gradient_checkpointing,
    )
    return encoders.to(device)


def cache_manifest(cfg, encoders):
    manifest = encoders.manifest()
    manifest.update({
        "visual_fps": float(cfg.data.visual_fps),
        "image_size": int(cfg.data.image_size),
        "face_crop": bool(cfg.data.face_crop),
        "sample_rate": int(cfg.data.sample_rate),
        "audio_chunk_seconds": float(cfg.data.audio_chunk_seconds),
        "audio_hop_seconds": float(cfg.data.audio_hop_seconds),
    })
    return manifest
