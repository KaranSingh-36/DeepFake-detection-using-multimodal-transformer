import os
import subprocess
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class LAVDFDataset(Dataset):
    """
    LAV-DF dataset loader.

    Returns:
        frames: [num_frames, 3, 224, 224]
        audio:  [audio_samples]
        label:  0 = real, 1 = deepfake
    """

    def __init__(
        self,
        metadata_csv="metadata/lavdf_metadata.csv",
        dataset_root="datasets/LAV-DF/LAV-DF",
        split="train",
        num_frames=8,
        image_size=224,
        sample_rate=16000,
        audio_duration=4.0,
    ):
        self.metadata_csv = metadata_csv
        self.dataset_root = dataset_root
        self.split = split

        self.num_frames = num_frames
        self.image_size = image_size
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration

        # Number of audio samples returned for every video
        self.num_audio_samples = int(
            self.sample_rate * self.audio_duration
        )

        # Load metadata
        self.metadata = pd.read_csv(self.metadata_csv)

        # Keep only requested split
        self.metadata = self.metadata[
            self.metadata["split"] == self.split
        ].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                f"No samples found for split='{self.split}'"
            )

        print(
            f"LAV-DF {self.split} dataset loaded: "
            f"{len(self.metadata)} samples"
        )

    def __len__(self):
        return len(self.metadata)

    def _get_video_path(self, relative_path):
        """
        Convert metadata path such as:
            train/000001.mp4

        into:
            datasets/LAV-DF/LAV-DF/train/000001.mp4
        """

        return os.path.join(
            self.dataset_root,
            relative_path.replace("/", os.sep)
        )

    def _load_frames(self, video_path):
        """
        Sample equally spaced frames from the video.

        Returns:
            Tensor [num_frames, 3, 224, 224]
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(
                f"Video contains no frames: {video_path}"
            )

        # Select equally spaced frame indices
        frame_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int
        )

        frames = []

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:
                # If a frame cannot be read,
                # use the previous frame if available.
                if len(frames) > 0:
                    frame = frames[-1].copy()
                else:
                    cap.release()
                    raise RuntimeError(
                        f"Could not read frame from: {video_path}"
                    )

            # OpenCV: BGR
            # Convert to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize
            frame = cv2.resize(
                frame,
                (self.image_size, self.image_size)
            )

            # Convert uint8 [0,255] -> float [0,1]
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

        cap.release()

        # [T, H, W, C]
        frames = np.stack(frames)

        # [T, C, H, W]
        frames = np.transpose(
            frames,
            (0, 3, 1, 2)
        )

        # Convert to tensor
        frames = torch.from_numpy(
            frames.copy()
        ).float()

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        frames = (frames - mean) / std

        return frames

    def _load_audio(self, video_path):
        """
        Extract mono 16-kHz audio directly from the MP4
        using FFmpeg.

        Returns:
            Tensor [64000] for 4 seconds at 16 kHz.
        """

        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "f32le",
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
            error_message = e.stderr.decode(
                "utf-8",
                errors="ignore"
            )

            raise RuntimeError(
                f"FFmpeg audio extraction failed:\n"
                f"{video_path}\n"
                f"{error_message}"
            )

        # Convert raw bytes to float32 audio
        audio = np.frombuffer(
            result.stdout,
            dtype=np.float32
        ).copy()

        if len(audio) == 0:
            raise RuntimeError(
                f"No audio found in: {video_path}"
            )

        # Normalize length
        if len(audio) < self.num_audio_samples:

            padding = np.zeros(
                self.num_audio_samples - len(audio),
                dtype=np.float32
            )

            audio = np.concatenate(
                [audio, padding]
            )

        else:
            audio = audio[:self.num_audio_samples]

        audio = torch.from_numpy(
            audio
        ).float()

        return audio

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        relative_path = row["file"]

        video_path = self._get_video_path(
            relative_path
        )

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video does not exist:\n{video_path}"
            )

        # Load video frames
        frames = self._load_frames(
            video_path
        )

        # Load audio
        audio = self._load_audio(
            video_path
        )

        # Label
        label = int(row["label"])

        return {
            "frames": frames,
            "audio": audio,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
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

    print()
    print("Dataset size:", len(dataset))

    print()
    print("Loading first sample...")

    sample = dataset[0]

    print()
    print("File:")
    print(sample["file"])

    print()
    print("Frames shape:")
    print(sample["frames"].shape)

    print()
    print("Audio shape:")
    print(sample["audio"].shape)

    print()
    print("Label:")
    print(sample["label"].item())

    print()
    print("Frame dtype:")
    print(sample["frames"].dtype)

    print()
    print("Audio dtype:")
    print(sample["audio"].dtype)

    print()
    print("Audio min/max:")
    print(
        sample["audio"].min().item(),
        sample["audio"].max().item()
    )

    print()
    print("=" * 60)
    print("DATASET LOADER TEST SUCCESSFUL")
    print("=" * 60)