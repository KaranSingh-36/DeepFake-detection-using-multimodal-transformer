
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Configuration - MATCHES THE 25K / 2-EPOCH MODEL PIPELINE
# ---------------------------------------------------------------------
NUM_FRAMES = 8
IMAGE_SIZE = 224
SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0
NUM_AUDIO_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "results" / "checkpoints" / "multimodal_25000_2epoch_epoch_2.pth",
    PROJECT_ROOT / "results" / "checkpoints" / "multimodal_25000_2epoch_epoch_2.pt",
    PROJECT_ROOT / "results" / "checkpoints" / "multimodal_25000_epoch2.pth",
]

# Benchmark metrics from the completed 25K / 2-epoch LAV-DF test.
BENCHMARK = {
    "test_samples": 26100,
    "accuracy": 97.65,
    "precision": 99.95,
    "recall": 96.85,
    "f1": 98.38,
    "roc_auc": 99.94,
    "tn": 6897,
    "fp": 9,
    "fn": 604,
    "tp": 18590,
}

st.set_page_config(
    page_title="Deepfake Multimodal Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 {margin: 0 0 .25rem 0; font-size: 2.1rem;}
.hero p {margin: 0; color: #d1d5db;}
.stage {
    border: 1px solid #374151;
    border-radius: 12px;
    padding: .8rem;
    background: #111827;
    color: #f9fafb;
}
.stage small {color: #9ca3af;}
.metric-card {
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: .8rem 1rem;
    background: #ffffff;
}
.result-real {
    border: 2px solid #16a34a;
    border-radius: 16px;
    padding: 1.2rem;
    background: #dcfce7 !important;
    color: #14532d !important;
}

.result-real * {
    color: #14532d !important;
}

.result-fake {
    border: 2px solid #dc2626;
    border-radius: 16px;
    padding: 1.2rem;
    background: #fee2e2 !important;
    color: #7f1d1d !important;
}

.result-fake * {
    color: #7f1d1d !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The project has used both spellings in older scripts. Prefer the
    # current multimodal_model.py and fall back to the older filename.
    try:
        from models.multimodal_model import MultimodalCrossAttentionModel
    except ImportError:
        from models.multimodal_model import MultimodalCrossAttentionModel

    model = MultimodalCrossAttentionModel().to(device)

    checkpoint_path = next(
        (p for p in CHECKPOINT_CANDIDATES if p.exists()),
        None,
    )
    if checkpoint_path is None:
        # Also allow any 25K epoch-2 checkpoint to be discovered.
        candidates = sorted(
            (PROJECT_ROOT / "results" / "checkpoints").glob(
                "*25000*epoch*2*.pth"
            )
        )
        if candidates:
            checkpoint_path = candidates[-1]

    if checkpoint_path is None:
        raise FileNotFoundError(
            "25K epoch-2 checkpoint was not found in results/checkpoints.\n"
            "Expected: multimodal_25000_2epoch_epoch_2.pth"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, checkpoint, str(checkpoint_path), str(device)


def run_ffmpeg_audio(video_path: str):
    """Extract mono 16 kHz float32 PCM exactly like LAVDFDataset."""
    command = [
        "ffmpeg",
        "-v", "error",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if len(audio) == 0:
        raise RuntimeError("No audio stream was found in the uploaded video.")

    if len(audio) < NUM_AUDIO_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_AUDIO_SAMPLES - len(audio)),
            mode="constant",
        )
    else:
        audio = audio[:NUM_AUDIO_SAMPLES]
    return audio


def extract_video_frames(video_path: str):
    """Match LAVDFDataset frame sampling: 8 equally spaced RGB 224x224 frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Uploaded video contains no readable frames.")

    duration = total_frames / fps if fps > 0 else 0.0
    frame_indices = np.linspace(
        0, total_frames - 1, NUM_FRAMES, dtype=int
    )

    frames_rgb = []
    timestamps = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if frames_rgb:
                frame = cv2.cvtColor(frames_rgb[-1], cv2.COLOR_RGB2BGR)
            else:
                cap.release()
                raise RuntimeError("Could not read the first sampled frame.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        frames_rgb.append(resized)
        timestamps.append(float(idx / fps) if fps > 0 else 0.0)

    cap.release()

    # Exact tensor preprocessing used by LAVDFDataset.
    arr = np.stack(frames_rgb).astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    frames = torch.from_numpy(arr.copy()).float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    return {
        "display_frames": frames_rgb,
        "model_frames": frames,
        "timestamps": timestamps,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


def make_waveform(audio):
    fig = plt.figure(figsize=(10, 2.4))
    ax = fig.add_subplot(111)
    t = np.arange(len(audio)) / SAMPLE_RATE
    ax.plot(t, audio, linewidth=0.7)
    ax.set_title("Extracted Audio Waveform — 16 kHz Mono")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_spectrogram(audio):
    fig = plt.figure(figsize=(10, 3.0))
    ax = fig.add_subplot(111)
    ax.specgram(
        audio,
        NFFT=1024,
        Fs=SAMPLE_RATE,
        noverlap=768,
        cmap="magma",
    )
    ax.set_title("Audio Spectrogram")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    return fig


def frame_grid(frames, timestamps):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(frames[i])
        ax.set_title(f"Frame {i+1}  •  {timestamps[i]:.2f}s")
        ax.axis("off")
    fig.tight_layout()
    return fig


def make_probability_chart(real_p, fake_p):
    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(111)
    labels = ["REAL", "DEEPFAKE"]
    values = [real_p * 100, fake_p * 100]
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Model probability (%)")
    ax.set_title("Multimodal prediction")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2, 98),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    return fig


def format_checkpoint_info(checkpoint):
    keys = {
        "epoch": checkpoint.get("epoch"),
        "train_accuracy": checkpoint.get("train_accuracy"),
        "val_accuracy": checkpoint.get("val_accuracy"),
        "val_f1": checkpoint.get("val_f1"),
        "val_auc": checkpoint.get("val_auc"),
        "train_samples": checkpoint.get("train_samples"),
        "validation_samples": checkpoint.get("validation_samples"),
    }
    return keys


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown("""
<div class="hero">
<h1>🛡️ Deepfake Detection — Multimodal Transformer</h1>
<p>Local demonstration dashboard • ViT + Wav2Vec2 + Cross-Attention • 25K / 2-Epoch LAV-DF checkpoint</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("System")
    st.write("**Model:** Multimodal Cross-Attention Transformer")
    st.write("**Visual encoder:** ViT")
    st.write("**Audio encoder:** Wav2Vec2")
    st.write("**Fusion:** Cross-attention")
    st.write("**Input:** MP4 → 8 frames + 4 s audio")
    st.write("**Device:** " + ("NVIDIA CUDA" if torch.cuda.is_available() else "CPU"))

    st.divider()
    st.header("25K / 2-Epoch Benchmark")
    st.metric("LAV-DF Test Accuracy", f"{BENCHMARK['accuracy']:.2f}%")
    st.metric("ROC-AUC", f"{BENCHMARK['roc_auc']:.2f}%")
    st.metric("F1 Score", f"{BENCHMARK['f1']:.2f}%")
    st.caption(
        "These are benchmark metrics measured on the 26,100-video LAV-DF "
        "test set. They are not the accuracy of an individual uploaded video."
    )

# Load model
try:
    with st.spinner("Loading the trained 25K / 2-epoch checkpoint..."):
        model, checkpoint, checkpoint_path, device = load_model()
    ck = format_checkpoint_info(checkpoint)
except Exception as e:
    st.error("Could not load the trained model.")
    st.code(str(e))
    st.stop()

# Status
c1, c2, c3, c4 = st.columns(4)
c1.metric("Checkpoint epoch", str(ck["epoch"] if ck["epoch"] is not None else 2))
c2.metric("Trainable params", "192.77M")
c3.metric("Test samples", f"{BENCHMARK['test_samples']:,}")
c4.metric("Inference device", "RTX 3050" if "cuda" in device else "CPU")

st.caption(f"Loaded checkpoint: `{checkpoint_path}`")

st.divider()

# Architecture visualization
st.subheader("How the uploaded video is analyzed")
arch_cols = st.columns(5)
arch_cols[0].markdown("**1. MP4**\n\nVideo + audio")
arch_cols[1].markdown("**2. Video branch**\n\n8 sampled frames → ViT")
arch_cols[2].markdown("**3. Audio branch**\n\n4 s mono / 16 kHz → Wav2Vec2")
arch_cols[3].markdown("**4. Fusion**\n\nCross-attention Transformer")
arch_cols[4].markdown("**5. Decision**\n\nREAL / DEEPFAKE")

st.info(
    "The dashboard uses the same core preprocessing dimensions as the 25K training run: "
    "8 equally spaced 224×224 frames and a 4-second, 16-kHz mono audio segment."
)

st.divider()

# Upload
st.subheader("Upload a video for live demonstration")
uploaded = st.file_uploader(
    "Choose an MP4/MOV/AVI video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded is None:
    st.markdown(
        "Upload a video above. The dashboard will show the actual extraction pipeline, "
        "the visual frames sent to the model, the extracted audio, and the final multimodal prediction."
    )
    st.stop()

# Save uploaded file
work_dir = Path(tempfile.mkdtemp(prefix="deepfake_dashboard_"))
video_path = work_dir / uploaded.name
video_path.write_bytes(uploaded.getbuffer())

st.video(str(video_path))

# ---------------------------------------------------------------------
# Processing stages
# ---------------------------------------------------------------------
st.subheader("Live detection pipeline")

progress = st.progress(0)
status = st.empty()

stage_cols = st.columns(5)
for col, title in zip(
    stage_cols,
    ["MP4 loaded", "Video extracted", "Audio extracted", "Features ready", "Prediction"],
):
    col.markdown(f'<div class="stage"><b>{title}</b><br><small>Waiting</small></div>', unsafe_allow_html=True)

try:
    status.info("Stage 1/5 — Reading MP4 container...")
    progress.progress(10)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    probe = cv2.VideoCapture(str(video_path))
    probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
    probe_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    probe_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    st.write(
        f"**Input:** `{uploaded.name}`  •  **Size:** {file_size_mb:.2f} MB  •  "
        f"**Resolution:** {probe_w}×{probe_h}  •  **FPS:** {probe_fps:.2f}  •  "
        f"**Frames:** {probe_frames:,}"
    )

    status.info("Stage 2/5 — Sampling and preprocessing 8 visual frames...")
    progress.progress(30)
    video_info = extract_video_frames(str(video_path))

    st.success(
        f"Video branch ready: {NUM_FRAMES} frames sampled across "
        f"{video_info['duration']:.2f} seconds."
    )

    status.info("Stage 3/5 — Extracting mono 16-kHz audio with FFmpeg...")
    progress.progress(50)
    audio = run_ffmpeg_audio(str(video_path))

    st.success(
        f"Audio branch ready: {len(audio):,} samples "
        f"({len(audio)/SAMPLE_RATE:.2f} seconds at {SAMPLE_RATE:,} Hz)."
    )

    status.info("Stage 4/5 — Sending visual + audio tensors through the trained model...")
    progress.progress(70)

    frames = video_info["model_frames"].unsqueeze(0).to(device)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(frames, audio_tensor)
        probabilities = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    inference_seconds = time.perf_counter() - start

    real_p = float(probabilities[0])
    fake_p = float(probabilities[1])
    prediction = "DEEPFAKE" if fake_p >= real_p else "REAL"
    confidence = max(real_p, fake_p) * 100.0

    status.info("Stage 5/5 — Final classification ready.")
    progress.progress(100)

except Exception as e:
    progress.empty()
    status.error("Detection pipeline failed.")
    st.exception(e)
    st.stop()

# ---------------------------------------------------------------------
# Visual/audio inspection
# ---------------------------------------------------------------------
st.divider()
st.subheader("1. Video → visual frames")

v1, v2 = st.columns([1.6, 1])
with v1:
    st.pyplot(
        frame_grid(
            video_info["display_frames"],
            video_info["timestamps"],
        ),
        clear_figure=True,
    )
with v2:
    st.markdown("### Visual tensor")
    st.write(f"Shape: `{tuple(frames.shape)}`")
    st.write("8 equally spaced frames")
    st.write("RGB")
    st.write("224 × 224 pixels")
    st.write("ImageNet normalization")
    st.write("→ ViT visual encoder")

st.divider()
st.subheader("2. MP4 → audio")

a1, a2 = st.columns([1.6, 1])
with a1:
    st.pyplot(make_waveform(audio), clear_figure=True)
    st.pyplot(make_spectrogram(audio), clear_figure=True)
with a2:
    st.markdown("### Audio tensor")
    st.write(f"Shape: `{tuple(audio_tensor.shape)}`")
    st.write("Mono")
    st.write("16,000 Hz")
    st.write("4.0 second analysis window")
    st.write("→ Wav2Vec2 audio encoder")

st.divider()
st.subheader("3. Multimodal cross-attention → classification")

p1, p2 = st.columns([1.15, 1])
with p1:
    st.pyplot(make_probability_chart(real_p, fake_p), clear_figure=True)
with p2:
    if prediction == "DEEPFAKE":
        st.markdown(
            f"""
            <div class="result-fake">
            <h2>DEEPFAKE</h2>
            <h3>{confidence:.2f}% confidence</h3>
            <p>Real probability: <b>{real_p*100:.2f}%</b><br>
            Deepfake probability: <b>{fake_p*100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-real">
            <h2>REAL</h2>
            <h3>{confidence:.2f}% confidence</h3>
            <p>Real probability: <b>{real_p*100:.2f}%</b><br>
            Deepfake probability: <b>{fake_p*100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.metric("Single-video inference time", f"{inference_seconds:.2f} s")
    st.caption(
        "Confidence is the model's softmax probability for this uploaded video. "
        "It is different from the 97.65% LAV-DF benchmark accuracy."
    )

st.divider()
st.subheader("Research benchmark used for this demonstration")

m = st.columns(6)
m[0].metric("Accuracy", f"{BENCHMARK['accuracy']:.2f}%")
m[1].metric("Precision", f"{BENCHMARK['precision']:.2f}%")
m[2].metric("Recall", f"{BENCHMARK['recall']:.2f}%")
m[3].metric("F1", f"{BENCHMARK['f1']:.2f}%")
m[4].metric("ROC-AUC", f"{BENCHMARK['roc_auc']:.2f}%")
m[5].metric("Test videos", f"{BENCHMARK['test_samples']:,}")

st.write(
    f"**Confusion matrix:** TN={BENCHMARK['tn']:,}, "
    f"FP={BENCHMARK['fp']:,}, FN={BENCHMARK['fn']:,}, TP={BENCHMARK['tp']:,}"
)

st.caption(
    "Benchmark result: 25,000 training samples, 2 epochs, batch size 1, "
    "the saved epoch-2 checkpoint, evaluated on the full 26,100-video LAV-DF test set."
)

st.divider()
st.subheader("What the dashboard is demonstrating")

st.markdown("""
- **Video path:** MP4 → OpenCV → 8 temporal samples → 224×224 RGB → ImageNet normalization → ViT.
- **Audio path:** MP4 → FFmpeg → mono 16-kHz PCM → 4-second tensor → Wav2Vec2.
- **Fusion:** visual and audio representations are combined by the trained cross-attention multimodal model.
- **Decision:** the classifier outputs REAL vs DEEPFAKE probabilities.
- **Research metric:** 97.65% is the measured LAV-DF test-set accuracy of the 25K/2-epoch checkpoint, not a claim that every uploaded video is correct with 97.65% probability.
""")
