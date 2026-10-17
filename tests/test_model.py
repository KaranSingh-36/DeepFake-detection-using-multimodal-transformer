import numpy as np
import pytest
import torch

from fusiontrace.data import slice_steps, step_targets, window_tensors
from fusiontrace.windows import VideoMeta
from fusiontrace.model import FUSION_VARIANTS, WindowFusionModel, multitask_loss


def make_batch(b=3, tv=8, ta=40, dv=1536, da=768, layers=2):
    visual_mask = torch.ones(b, tv, dtype=torch.bool)
    audio_mask = torch.ones(b, ta, dtype=torch.bool)
    visual_mask[1, 5:] = False       # a short clip
    audio_mask[1, 25:] = False
    return {
        "visual": torch.randn(b, tv, dv),
        "audio": torch.randn(b, layers, ta, da),
        "visual_mask": visual_mask,
        "audio_mask": audio_mask,
        "y_video": torch.tensor([1.0, 0.0, 0.0]),
        "y_audio": torch.tensor([0.0, 1.0, 0.0]),
        "y_any": torch.tensor([1.0, 1.0, 0.0]),
        "ambiguous": torch.tensor([False, False, False]),
        "y_video_steps": (torch.rand(b, tv) > 0.5).float(),
        "y_audio_steps": (torch.rand(b, ta) > 0.5).float(),
        "y_any_steps": (torch.rand(b, tv) > 0.5).float(),
    }


@pytest.mark.parametrize("fusion", FUSION_VARIANTS)
def test_forward_backward_all_variants(fusion):
    torch.manual_seed(0)
    model = WindowFusionModel(1536, 768, 2, dim=64, heads=4, fusion=fusion)
    batch = make_batch()
    out = model(batch["visual"], batch["audio"], batch["visual_mask"], batch["audio_mask"], return_attention=True)
    assert out["fused_logit"].shape == (3,)
    assert torch.isfinite(out["fused_logit"]).all()
    if fusion == "audio_only":
        assert out["video_logit"] is None
    else:
        assert out["video_logit"].shape == (3,)
    if fusion == "video_only":
        assert out["audio_logit"] is None
    else:
        assert out["audio_logit"].shape == (3,)
    if fusion == "cross_attention":
        assert out["visual_to_audio_attention"].shape == (3, 8, 40)
    assert out["fused_step_logit"].shape == (3, 8)
    if fusion != "audio_only":
        assert out["video_step_logit"].shape == (3, 8)
    if fusion != "video_only":
        assert out["audio_step_logit"].shape == (3, 40)
    loss, parts = multitask_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert "steps" in parts and set(parts) <= {"video", "audio", "fused", "steps"}


def test_padding_does_not_change_valid_output():
    torch.manual_seed(0)
    model = WindowFusionModel(16, 8, 1, dim=32, heads=2, dropout=0.0).eval()
    visual = torch.randn(1, 8, 16)
    audio = torch.randn(1, 1, 20, 8)
    out_full = model(visual, audio)
    full = out_full["fused_logit"]
    # keep the audio/visual step ratio (2.5) so the step pooling bins line up
    padded_visual = torch.cat([visual, torch.zeros(1, 4, 16)], dim=1)
    padded_audio = torch.cat([audio, torch.zeros(1, 1, 10, 8)], dim=2)
    vmask = torch.cat([torch.ones(1, 8, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool)], dim=1)
    amask = torch.cat([torch.ones(1, 20, dtype=torch.bool), torch.zeros(1, 10, dtype=torch.bool)], dim=1)
    out_masked = model(padded_visual, padded_audio, vmask, amask)
    assert torch.allclose(full, out_masked["fused_logit"], atol=1e-4)
    assert torch.allclose(out_full["fused_step_logit"], out_masked["fused_step_logit"][:, :8], atol=1e-4)


def test_ambiguous_windows_are_excluded_from_loss():
    model = WindowFusionModel(16, 8, 1, dim=32, heads=2, dropout=0.0)
    batch = make_batch(dv=16, da=8, layers=1)
    out = model(batch["visual"], batch["audio"], batch["visual_mask"], batch["audio_mask"])
    loss_all, _ = multitask_loss(out, batch)
    batch["ambiguous"] = torch.tensor([True, True, True])
    loss_none, _ = multitask_loss(out, batch)
    assert float(loss_none) == 0.0 and float(loss_all) > 0.0


def test_slice_steps_pads_and_masks():
    seq = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    piece, mask = slice_steps(seq, rate=5.0, start=1.2, window_seconds=1.6)   # steps 6..13
    assert piece.shape == (8, 3)
    assert mask.tolist() == [True] * 4 + [False] * 4
    assert np.allclose(piece[:4], seq[6:10]) and np.all(piece[4:] == 0)
    features = {"visual": seq, "visual_fps": 5.0, "audio": np.zeros((2, 50, 4), np.float16), "audio_hz": 25.0}
    tensors = window_tensors(features, 0.0, 1.6)
    assert tensors["visual"].shape == (8, 3) and tensors["audio"].shape == (2, 40, 4)


def test_step_targets_follow_periods_and_flags():
    meta = VideoMeta("x", "train", 6.0, modify_video=False, modify_audio=True, fake_periods=[(2.0, 2.7)])
    video, audio, any_ = step_targets(meta, 1.6, 1.6, 5.0)      # centres 1.7, 1.9, ..., 3.1
    assert video.sum() == 0
    assert audio.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]          # 2.7 is the exclusive end
    assert any_.tolist() == audio.tolist()
