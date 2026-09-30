# 🎭 Deepfake Detection Using Multimodal Transformers

### Audio-Visual Deepfake Detection for Cybersecurity Applications

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/ViT-Vision%20Transformer-purple?style=for-the-badge" alt="ViT">
  <img src="https://img.shields.io/badge/Wav2Vec2-Audio%20Transformer-green?style=for-the-badge" alt="Wav2Vec2">
  <img src="https://img.shields.io/badge/Dataset-LAV--DF-red?style=for-the-badge" alt="LAV-DF">
</p>

<p align="center">
  <b>A research-oriented multimodal deepfake detection system that jointly analyzes visual and audio information from videos.</b>
</p>

---

## 📌 Overview

Deepfakes are AI-generated or manipulated media that can convincingly alter a person's appearance, speech, or behavior.

Traditional deepfake detection systems often focus on only one modality, such as video frames or audio. However, modern deepfakes can manipulate multiple modalities simultaneously.

This project investigates a **multimodal Transformer-based approach** that analyzes both:

- 🎥 **Visual information** from video frames
- 🎙️ **Audio information** from speech/audio tracks

The extracted representations are combined using **bidirectional cross-attention**, allowing the visual and audio modalities to exchange information before classification.

The final system predicts whether the input video is:

> **REAL** or **DEEPFAKE**

---

## 🎯 Project Objectives

The main objectives of this project are:

- Build a **video-only deepfake detection baseline**
- Build an **audio-only deepfake detection baseline**
- Develop a **multimodal audio-visual Transformer architecture**
- Use **Vision Transformer (ViT)** for visual representation learning
- Use **Wav2Vec2** for audio representation learning
- Investigate **bidirectional cross-attention** between audio and visual features
- Compare cross-attention fusion with simpler feature-fusion methods
- Perform systematic evaluation using multiple classification metrics
- Conduct ablation studies to understand the contribution of different components
- Investigate multimodal explainability using techniques such as **Grad-CAM** and **SHAP**
- Study the potential for **cross-dataset generalization**
- Develop an API and interactive interface for future deployment

---

## 🔬 Research Question

> **Does bidirectional cross-attention between visual and audio representations provide useful information for deepfake detection beyond unimodal models and simple feature concatenation?**

The project is designed to investigate this question experimentally through:

1. Video-only baseline
2. Audio-only baseline
3. Simple multimodal feature fusion
4. Proposed multimodal cross-attention architecture
5. Ablation studies
6. Cross-dataset evaluation

---

# 🧠 System Architecture

The proposed system processes the video through two separate modalities.

```mermaid
flowchart LR

    A[Input Video]

    A --> B[Video Frame Extraction]
    A --> C[Audio Extraction]

    B --> D[Vision Transformer<br/>ViT]
    C --> E[Wav2Vec2]

    D --> F[Visual Tokens]
    E --> G[Audio Tokens]

    F --> H[Visual-to-Audio<br/>Cross Attention]
    G --> I[Audio-to-Visual<br/>Cross Attention]

    H --> J[Multimodal Feature Fusion]
    I --> J

    J --> K[Temporal Pooling]
    K --> L[Binary Classifier]

    L --> M{Prediction}

    M --> N[REAL]
    M --> O[DEEPFAKE]
