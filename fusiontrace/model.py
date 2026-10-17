"""Window-level fusion model with per-modality heads.

Inputs (from the cache or from FrozenEncoders):
    visual [B, Tv, Dv]   audio [B, L, Ta, Da]   masks [B, T] (True = valid)

Outputs (window level, one logit per window):
    video_logit  - evidence from frames only
    audio_logit  - evidence from audio only
    fused_logit  - joint decision (cross-attention / concat / unimodal variant)
Outputs (step level, for localisation inside the window):
    video_step_logit [B, Tv]  audio_step_logit [B, Ta]  fused_step_logit [B, Tv]

The unimodal heads are computed before any cross-modal exchange so that they
remain honest per-modality detectors. `fusion` selects the research variant:
    cross_attention | concat | video_only | audio_only
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_VARIANTS = ("cross_attention", "concat", "video_only", "audio_only")


def sinusoidal_positions(length, dim, device):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)[:, : dim // 2]
    return table


def masked_mean(x, mask):
    if mask is None:
        return x.mean(dim=1)
    weights = mask.float().unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class InputProjection(nn.Module):
    def __init__(self, in_dim, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(self.norm(x))
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device).unsqueeze(0)
        return self.drop(self.out_norm(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context, context_mask=None, need_weights=False):
        key_padding = None if context_mask is None else ~context_mask
        attended, weights = self.attn(
            self.q_norm(x), self.kv_norm(context), self.kv_norm(context),
            key_padding_mask=key_padding, need_weights=need_weights, average_attn_weights=True,
        )
        x = x + self.drop(attended)
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x, weights


class Head(nn.Module):
    def __init__(self, in_dim, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WindowFusionModel(nn.Module):
    def __init__(self, visual_dim, audio_dim, n_audio_layers, dim=256, heads=4,
                 temporal_layers=1, dropout=0.1, fusion="cross_attention"):
        super().__init__()
        if fusion not in FUSION_VARIANTS:
            raise ValueError(f"fusion must be one of {FUSION_VARIANTS}, got {fusion}")
        if dim % 2:
            raise ValueError("dim must be even")
        self.fusion = fusion
        self.dim = dim
        self.use_video = fusion != "audio_only"
        self.use_audio = fusion != "video_only"

        def temporal_encoder():
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, temporal_layers, enable_nested_tensor=False)

        if self.use_video:
            self.visual_in = InputProjection(visual_dim, dim, dropout)
            self.visual_temporal = temporal_encoder()
            self.video_head = Head(dim, dim, dropout)
            self.video_step_head = Head(dim, dim, dropout)
        if self.use_audio:
            self.audio_layer_weights = nn.Parameter(torch.zeros(n_audio_layers))
            self.audio_in = InputProjection(audio_dim, dim, dropout)
            self.audio_temporal = temporal_encoder()
            self.audio_head = Head(dim, dim, dropout)
            self.audio_step_head = Head(dim, dim, dropout)

        fused_dim = 2 * dim if fusion in ("cross_attention", "concat") else dim
        if fusion == "cross_attention":
            self.v_from_a = CrossAttentionBlock(dim, heads, dropout)
            self.a_from_v = CrossAttentionBlock(dim, heads, dropout)
        self.fused_head = Head(fused_dim, dim, dropout)
        self.fused_step_head = Head(fused_dim, dim, dropout)

    @staticmethod
    def _to_visual_rate(a, audio_mask, n_steps):
        """Average-pool audio steps [B, Ta, D] onto the visual grid [B, Tv, D]."""
        if audio_mask is not None:
            a = a * audio_mask.unsqueeze(-1).float()
            counts = F.adaptive_avg_pool1d(audio_mask.float().unsqueeze(1), n_steps).squeeze(1)
            pooled = F.adaptive_avg_pool1d(a.transpose(1, 2), n_steps).transpose(1, 2)
            return pooled / counts.clamp(min=1e-6).unsqueeze(-1)
        return F.adaptive_avg_pool1d(a.transpose(1, 2), n_steps).transpose(1, 2)

    def forward(self, visual=None, audio=None, visual_mask=None, audio_mask=None, return_attention=False):
        out = {"video_logit": None, "audio_logit": None, "fused_logit": None}
        v = a = None

        if self.use_video:
            v = self.visual_in(visual)
            v = self.visual_temporal(v, src_key_padding_mask=None if visual_mask is None else ~visual_mask)
            out["video_logit"] = self.video_head(masked_mean(v, visual_mask))

        if self.use_audio:
            weights = torch.softmax(self.audio_layer_weights, dim=0)
            audio_mixed = (audio * weights.view(1, -1, 1, 1)).sum(dim=1)
            a = self.audio_in(audio_mixed)
            a = self.audio_temporal(a, src_key_padding_mask=None if audio_mask is None else ~audio_mask)
            out["audio_logit"] = self.audio_head(masked_mean(a, audio_mask))

        n_steps = visual.shape[1]
        if self.use_video:
            out["video_step_logit"] = self.video_step_head(v)
        if self.use_audio:
            out["audio_step_logit"] = self.audio_step_head(a)

        if self.fusion == "cross_attention":
            v2, attn_v = self.v_from_a(v, a, audio_mask, need_weights=return_attention)
            a2, attn_a = self.a_from_v(a, v, visual_mask, need_weights=return_attention)
            fused = torch.cat([masked_mean(v2, visual_mask), masked_mean(a2, audio_mask)], dim=-1)
            fused_steps = torch.cat([v2, self._to_visual_rate(a2, audio_mask, n_steps)], dim=-1)
            if return_attention:
                out["visual_to_audio_attention"] = attn_v
                out["audio_to_visual_attention"] = attn_a
        elif self.fusion == "concat":
            fused = torch.cat([masked_mean(v, visual_mask), masked_mean(a, audio_mask)], dim=-1)
            fused_steps = torch.cat([v, self._to_visual_rate(a, audio_mask, n_steps)], dim=-1)
        elif self.fusion == "video_only":
            fused = masked_mean(v, visual_mask)
            fused_steps = v
        else:
            fused = masked_mean(a, audio_mask)
            fused_steps = self._to_visual_rate(a, audio_mask, n_steps)

        out["fused_logit"] = self.fused_head(fused)
        out["fused_step_logit"] = self.fused_step_head(fused_steps)
        return out


class EndToEndModel(nn.Module):
    """FrozenEncoders (optionally with trainable top blocks) + WindowFusionModel."""

    def __init__(self, encoders, fusion):
        super().__init__()
        self.encoders = encoders
        self.fusion_model = fusion

    def forward(self, frames_uint8, waveform, return_attention=False):
        """frames_uint8: {"face": uint8 [B, T, S, S, 3], "mouth": ...} (or a single tensor)."""
        if torch.is_tensor(frames_uint8):
            frames_uint8 = {self.encoders.visual_streams[0]: frames_uint8}
        device = self.encoders.image_mean.device
        b, t = next(iter(frames_uint8.values())).shape[:2]
        flat = {name: frames.to(device).flatten(0, 1) for name, frames in frames_uint8.items()}
        visual = self.encoders.encode_visual(flat).view(b, t, -1)
        audio = self.encoders.encode_audio(waveform.to(device))
        return self.fusion_model(visual, audio, None, None, return_attention=return_attention)


def _fit_length(target, length):
    """Step targets and logits can differ by a step (raw vs cached audio); trim or pad."""
    if target.shape[1] == length:
        return target
    if target.shape[1] > length:
        return target[:, :length]
    pad = target[:, -1:].expand(-1, length - target.shape[1])
    return torch.cat([target, pad], dim=1)


def apply_modality_dropout(batch, probability, generator=None):
    """Zero one modality for a share of the windows and relabel the fused targets.

    Without this the fused head can solve LAV-DF from audio alone and never
    learns to use the frames. For each window, with probability `probability`
    one modality is removed (half the time audio, half the time video): its
    features are zeroed, the fused window / step targets become the labels of
    the remaining modality, and `audio_present` / `video_present` flags tell
    `multitask_loss` to skip the unimodal head of the missing modality.
    """
    batch = dict(batch)
    n = batch["visual"].shape[0]
    if probability <= 0.0:
        return batch
    u = torch.rand(n, generator=generator)
    drop_audio = u < probability / 2.0
    drop_video = (u >= probability / 2.0) & (u < probability)
    device = batch["visual"].device
    drop_audio, drop_video = drop_audio.to(device), drop_video.to(device)

    if drop_audio.any():
        audio = batch["audio"].clone()
        audio[drop_audio] = 0
        batch["audio"] = audio
    if drop_video.any():
        visual = batch["visual"].clone()
        visual[drop_video] = 0
        batch["visual"] = visual

    y_video = batch["y_video"].float().to(device)
    y_audio = batch["y_audio"].float().to(device)
    y_any = batch["y_any"].float().to(device)
    batch["y_any"] = torch.where(drop_audio, y_video, torch.where(drop_video, y_audio, y_any))
    if "y_any_steps" in batch:
        steps = batch["y_any_steps"].float().to(device)
        keep = torch.where(drop_audio, y_video, torch.where(drop_video, y_audio, torch.ones_like(y_any)))
        batch["y_any_steps"] = steps * keep.unsqueeze(1)
        if "y_video_steps" in batch:
            video_steps = _fit_length(batch["y_video_steps"].float().to(device), steps.shape[1])
            batch["y_any_steps"] = torch.where(drop_audio.unsqueeze(1), video_steps, batch["y_any_steps"])
    batch["audio_present"] = ~drop_audio
    batch["video_present"] = ~drop_video
    return batch


def multitask_loss(outputs, batch, weights=None):
    """Masked BCE over the window heads and the step heads. Returns (total, parts)."""
    weights = {"video": 1.0, "audio": 1.0, "fused": 1.0, "steps": 1.0, **(weights or {})}
    parts = {}
    total = 0.0
    valid = (~batch["ambiguous"]).float() if "ambiguous" in batch else None

    def present(name, device):
        flag = batch.get(f"{name}_present")
        return None if flag is None else flag.float().to(device)

    window_targets = {"video": "y_video", "audio": "y_audio", "fused": "y_any"}
    for name, target_key in window_targets.items():
        logit = outputs.get(f"{name}_logit")
        if logit is None or weights.get(name, 0.0) == 0.0:
            continue
        target = batch[target_key].float().to(logit.device)
        loss = F.binary_cross_entropy_with_logits(logit.float(), target, reduction="none")
        mask = torch.ones_like(loss)
        if valid is not None:
            mask = mask * valid.to(logit.device)
        if present(name, logit.device) is not None:
            mask = mask * present(name, logit.device)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        parts[name] = loss.detach()
        total = total + weights[name] * loss

    if weights.get("steps", 0.0) > 0.0:
        step_specs = [("video_step_logit", "y_video_steps", "visual_mask", "video"),
                      ("audio_step_logit", "y_audio_steps", "audio_mask", "audio"),
                      ("fused_step_logit", "y_any_steps", "visual_mask", None)]
        step_losses = []
        for logit_key, target_key, mask_key, modality in step_specs:
            logit = outputs.get(logit_key)
            if logit is None or target_key not in batch:
                continue
            target = _fit_length(batch[target_key].float().to(logit.device), logit.shape[1])
            loss = F.binary_cross_entropy_with_logits(logit.float(), target, reduction="none")
            mask = torch.ones_like(loss)
            if batch.get(mask_key) is not None:
                mask = mask * _fit_length(batch[mask_key].float().to(logit.device), logit.shape[1])
            if valid is not None:
                mask = mask * valid.to(logit.device).unsqueeze(1)
            if modality is not None and present(modality, logit.device) is not None:
                mask = mask * present(modality, logit.device).unsqueeze(1)
            step_losses.append((loss * mask).sum() / mask.sum().clamp(min=1.0))
        if step_losses:
            step_loss = torch.stack(step_losses).mean()
            parts["steps"] = step_loss.detach()
            total = total + weights["steps"] * step_loss
    return total, parts
