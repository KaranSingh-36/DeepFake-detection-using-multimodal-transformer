import os
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LAVDFDataset(Dataset):
    """LAV-DF loader optimized for Windows multi-worker training."""

    def __init__(
        self,
        metadata_csv="metadata/lavdf_metadata.csv",
        dataset_root=r"C:\AI_DATA\LAV-DF\LAV-DF",
        split="train",
        num_frames=8,
        image_size=224,
        sample_rate=16000,
        audio_duration=4.0,
    ):
        self.metadata_csv = metadata_csv
        self.dataset_root = dataset_root
        self.split = split
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.sample_rate = int(sample_rate)
        self.audio_duration = float(audio_duration)
        self.num_audio_samples = int(self.sample_rate * self.audio_duration)

        metadata = pd.read_csv(self.metadata_csv)
        metadata = metadata[metadata["split"] == self.split].reset_index(drop=True)

        if len(metadata) == 0:
            raise ValueError(f"No samples found for split='{self.split}'")

        # Avoid pandas.iloc() for every sample.
        self.files = metadata["file"].astype(str).tolist()
        self.labels = metadata["label"].astype(np.int64).to_numpy(copy=True)

        # Reused by every sample.
        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1)

        self._std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(1, 3, 1, 1)

        print(f"LAV-DF {self.split} dataset loaded: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def _get_video_path(self, relative_path):
        return os.path.join(
            self.dataset_root,
            relative_path.replace("/", os.sep),
        )

    def _load_frames(self, video_path):
        """
        Sequentially decode once and retrieve only requested frames.
        This avoids repeated cap.set() seeks.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video contains no frames: {video_path}")

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=np.int64,
        )

        frames = []
        target_ptr = 0
        frame_index = 0
        last_frame = None

        while target_ptr < len(frame_indices):
            success = cap.grab()

            if not success:
                break

            target_index = int(frame_indices[target_ptr])

            if frame_index == target_index:
                success, frame = cap.retrieve()

                if not success:
                    if last_frame is None:
                        cap.release()
                        raise RuntimeError(
                            f"Could not read frame from: {video_path}"
                        )
                    frame = last_frame.copy()
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(
                        frame,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    frame = frame.astype(np.float32) / 255.0
                    last_frame = frame

                frames.append(frame)
                target_ptr += 1

            frame_index += 1

            if frame_index >= total_frames:
                break

        cap.release()

        if not frames:
            raise RuntimeError(f"Could not read frames from: {video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        frames = np.stack(frames[:self.num_frames])
        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = torch.from_numpy(frames.copy()).float()
        frames = (frames - self._mean) / self._std

        return frames

    def _load_audio(self, video_path):
        """
        Ask FFmpeg for only the 4-second window needed by Wav2Vec2.
        """
        command = [
            "ffmpeg",
            "-v", "error",
            "-i", video_path,
            "-t", f"{self.audio_duration:.3f}",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-f", "f32le",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(
                "FFmpeg audio extraction failed:\n"
                f"{video_path}\n{error_message}"
            ) from e

        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()

        if len(audio) == 0:
            raise RuntimeError(f"No audio found in: {video_path}")

        if len(audio) < self.num_audio_samples:
            audio = np.pad(
                audio,
                (0, self.num_audio_samples - len(audio)),
                mode="constant",
            )
        else:
            audio = audio[:self.num_audio_samples]

        return torch.from_numpy(audio).float()

    def __getitem__(self, index):
        relative_path = self.files[index]
        label = int(self.labels[index])
        video_path = self._get_video_path(relative_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video does not exist:\n{video_path}")

        return {
            "frames": self._load_frames(video_path),
            "audio": self._load_audio(video_path),
            "label": torch.tensor(label, dtype=torch.long),
            "file": relative_path,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LAV-DF Dataset Loader")
    print("=" * 60)

    dataset = LAVDFDataset(
        split="test",
        num_frames=8,
        image_size=224,
        sample_rate=16000,
        audio_duration=4.0,
    )

    print("Dataset size:", len(dataset))
    sample = dataset[0]
    print("File:", sample["file"])
    print("Frames shape:", sample["frames"].shape)
    print("Audio shape:", sample["audio"].shape)
    print("Label:", sample["label"].item())
    print("Frame dtype:", sample["frames"].dtype)
    print("Audio dtype:", sample["audio"].dtype)
    print(
        "Audio min/max:",
        sample["audio"].min().item(),
        sample["audio"].max().item(),
    )
    print("=" * 60)
    print("DATASET LOADER TEST SUCCESSFUL")
    print("=" * 60)
