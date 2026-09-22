Deepfake Detection Using Multimodal Transformer Architecture

Overview

This project develops a multimodal deepfake detection system for
cybersecurity applications. The proposed architecture jointly analyzes
visual and audio information from videos using pretrained Transformer
models and bidirectional cross-attention.

Core architecture

                    Input Video
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Frame Extraction          Audio Extraction
             |                       |
             v                       v
       Vision Transformer        Wav2Vec2
             |                       |
             v                       v
       Visual Tokens             Audio Tokens
             |                       |
             +-----------+-----------+
                         |
                  Bidirectional
                  Cross-Attention
                         |
                    Feature Fusion
                         |
                    Classifier
                     /       \
                  REAL     DEEPFAKE

Objectives

Develop video-only and audio-only deepfake detection baselines.

Develop a multimodal Transformer architecture.

Investigate bidirectional audio-visual cross-attention.

Compare cross-attention fusion with simpler fusion methods.

Perform rigorous evaluation and ablation studies.

Investigate Grad-CAM and SHAP for explainability.

Evaluate cross-dataset generalization.

Deploy the trained model through an API and interactive dashboard.

Models

Visual branch

Vision Transformer

google/vit-base-patch16-224

Video frames are processed individually and their CLS representations
are organized as a temporal visual sequence.

Expected representation:

[B, T, 768]

Audio branch

Wav2Vec2

facebook/wav2vec2-base

Audio is processed as a 16 kHz waveform.

Expected representation:

[B, audio_sequence_length, 768]

Multimodal fusion

The model performs:

Visual-to-audio cross-attention.

Audio-to-visual cross-attention.

Residual connections and normalization.

Feed-forward transformations.

Temporal pooling.

Feature fusion.

Binary classification.

Classes:

0 = REAL
1 = DEEPFAKE

Dataset

The primary development dataset is LAV-DF.

Current extracted dataset:

Split     Samples

Train      78,703
Dev        31,501
Test       26,100
Total     136,304

The metadata includes video paths, fake periods, duration, modification
information, split information, and audio/video properties.

Additional datasets such as FakeAVCeleb, DFDC-related benchmarks, and
AV-Deepfake1M may be used later for cross-dataset evaluation where
access and licensing permit.

Project Structure

major project/
├── .venv/
├── api/
├── dataset/
│   ├── lavdf_dataset.py
│   └── test_dataloader.py
├── datasets/
│   └── LAV-DF/
├── evaluation/
│   ├── evaluation.py
│   ├── test_video_inference.py
│   ├── test_multimodal_model.py
│   └── test_multimodal_backward.py
├── explainability/
├── metadata/
│   └── lavdf_metadata.csv
├── models/
│   ├── audio_model.py
│   ├── multimodal_model.py
│   └── video_model.py
├── preprocessing/
│   ├── create_metadata.py
│   ├── extract_audio.py
│   ├── extract_frames.py
│   ├── inspect_dataset.py
│   └── test_lavdf.py
├── results/
├── training/
│   ├── train_baseline.py
│   ├── train_multimodal.py
│   └── train_video_baseline.py
├── .gitignore
├── README.md
└── requirements.txt

Environment

Primary development hardware:

NVIDIA GeForce RTX 3050 Laptop GPU

4 GB dedicated VRAM

CUDA-enabled PyTorch

Python 3.12.x

Windows

VS Code

The multimodal model is memory-intensive. Training may use mixed
precision, gradient accumulation, gradient checkpointing, and
appropriate batch sizing while preserving the proposed architecture.

Setup

Create and activate the virtual environment:

python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Verify FFmpeg:

ffmpeg -version

Dataset Verification

Verify LAV-DF video loading:

python preprocessing/test_lavdf.py

Verify audio extraction:

python preprocessing/extract_audio.py

Create metadata:

python preprocessing/create_metadata.py

Test the DataLoader:

python dataset/test_dataloader.py

Model Tests

Video model:

python evaluation/test_video_inference.py

Multimodal forward pass:

python evaluation/test_multimodal_model.py

Multimodal backward and optimizer test:

python evaluation/test_multimodal_backward.py

The current development tests have successfully verified the multimodal
forward pass, cross-attention, backward propagation, gradients, and
optimizer update on the RTX 3050.

Experimental Plan

The research experiments will compare:

Video-only baseline

Video → ViT → Classifier

Audio-only baseline

Audio → Wav2Vec2 → Classifier

Simple multimodal fusion

Video → ViT ─────┐
                 ├→ Concatenation → Classifier
Audio → Wav2Vec2 ┘

Proposed model

Video → ViT ─────────────┐
                         ↓
                  Cross-Attention
                         ↑
Audio → Wav2Vec2 ────────┘
                         ↓
                      Fusion
                         ↓
                    Classifier

Evaluation

Final experiments will report:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion matrix

Per-class metrics

Training and validation curves

Additional measurements may include inference time and GPU memory usage.

Ablation Studies

Planned ablations include:

Video-only vs audio-only vs multimodal.

Simple concatenation vs cross-attention.

Different numbers of sampled frames.

Different audio durations.

Frozen vs fine-tuned pretrained encoders.

Different fusion configurations.

Explainability

The project plans to investigate:

Grad-CAM for visual evidence.

SHAP for feature and multimodal contribution analysis.

The objective is to provide evidence supporting model predictions rather
than only returning a binary label.

Deployment

After research experiments are completed, a FastAPI service and
interactive dashboard are planned.

Target workflow:

Upload Video
     ↓
Preprocessing
     ↓
Visual + Audio Extraction
     ↓
Multimodal Transformer
     ↓
REAL / DEEPFAKE
     ↓
Confidence + Explainability

Current Development Status

Completed

LAV-DF dataset acquisition and extraction

Dataset inspection

Metadata generation

Video preprocessing

Audio extraction

CUDA/PyTorch configuration

RTX 3050 detection

ViT visual model

Wav2Vec2 audio model

LAV-DF DataLoader

Video baseline smoke test

Multimodal model implementation

Multimodal forward-pass test

Multimodal backward-pass test

Optimizer-step test

Next

Real-data multimodal training smoke test

Full baseline experiments

Proposed-model training

Final evaluation

Ablation studies

Explainability

Cross-dataset testing

FastAPI deployment

Dashboard

Research paper

Reproducibility

Development smoke tests are not final research results.

Final experiments will save checkpoints, metrics, plots, confusion
matrices, ablation results, cross-dataset results, and explainability
outputs under the results/ directory so that reported research metrics
can be reproduced from clean runs.

Research Direction

The central research question is whether bidirectional multimodal
cross-attention between visual and audio representations provides useful
information beyond unimodal models and simple feature concatenation for
audio-visual deepfake detection.

Technologies

Category            Technology

Language            Python
Deep Learning       PyTorch
Visual Model        Vision Transformer
Audio Model         Wav2Vec2
Multimodal Fusion   Cross-Attention
Computer Vision     OpenCV
Audio Processing    FFmpeg
Transformers        Hugging Face Transformers
Explainability      Grad-CAM, SHAP
API                 FastAPI
Development         VS Code
Version Control     Git / GitHub

Author

Karan Singh
Tushar

MCA --- Artificial Intelligence & Machine Learning

GitHub: https://github.com/KaranSingh-36

Project Status

Research and Development --- In Progress
