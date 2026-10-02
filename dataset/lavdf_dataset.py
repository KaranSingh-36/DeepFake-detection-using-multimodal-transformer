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

        self.num_frames = num_frames
        self.image_size = image_size
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration

        self.num_audio_samples = int(
            self.sample_rate * self.audio_duration
        )

        # ImageNet normalization.
        # Store as NumPy arrays so they are not recreated
        # for every frame/sample.
        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        ).reshape(1, 1, 3)

        # --------------------------------------------------------
        # LOAD METADATA
        # --------------------------------------------------------

        self.metadata = pd.read_csv(
            self.metadata_csv
        )

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

    # ============================================================
    # VIDEO PATH
    # ============================================================

    def _get_video_path(self, relative_path):

        return os.path.join(
            self.dataset_root,
            relative_path.replace("/", os.sep)
        )

    # ============================================================
    # VIDEO LOADING
    # ============================================================

    def _load_frames(self, video_path):
        """
        Sequentially decode the video and keep approximately
        equally-spaced frames.

        This avoids repeated random seeking using:

            cap.set(CAP_PROP_POS_FRAMES, ...)

        which was used by the previous implementation.

        Returns:
            Tensor [num_frames, 3, image_size, image_size]
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

        # --------------------------------------------------------
        # SELECT TARGET FRAME INDICES
        # --------------------------------------------------------

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=np.int64
        )

        target_positions = set(
            int(x)
            for x in frame_indices
        )

        frames = []

        frame_index = 0

        # --------------------------------------------------------
        # SEQUENTIAL DECODE
        # --------------------------------------------------------

        while frame_index < total_frames:

            success, frame = cap.read()

            if not success:
                break

            if frame_index in target_positions:

                # BGR -> RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Resize
                frame = cv2.resize(
                    frame,
                    (
                        self.image_size,
                        self.image_size
                    ),
                    interpolation=cv2.INTER_LINEAR
                )

                # uint8 -> float32 [0, 1]
                frame = frame.astype(
                    np.float32
                ) / 255.0

                # ImageNet normalization
                frame = (
                    frame - self.mean
                ) / self.std

                frames.append(frame)

                # We already have all requested frames.
                if len(frames) >= self.num_frames:
                    break

            frame_index += 1

        cap.release()

        # --------------------------------------------------------
        # FALLBACK IF VIDEO DECODING ENDED EARLY
        # --------------------------------------------------------

        if len(frames) == 0:
            raise RuntimeError(
                f"Could not read frames from: {video_path}"
            )

        while len(frames) < self.num_frames:

            frames.append(
                frames[-1].copy()
            )

        frames = np.stack(
            frames,
            axis=0
        )

        # [T, H, W, C] -> [T, C, H, W]
        frames = np.transpose(
            frames,
            (0, 3, 1, 2)
        )

        frames = np.ascontiguousarray(
            frames
        )

        return torch.from_numpy(
            frames
        ).float()

    # ============================================================
    # AUDIO LOADING
    # ============================================================

    def _load_audio(self, video_path):
        """
        Extract mono 16-kHz audio from the MP4 using FFmpeg.

        Only the required audio duration is decoded.

        Returns:
            Tensor [audio_samples]
        """

        command = [
            "ffmpeg",

            "-v",
            "error",

            "-i",
            video_path,

            # Only decode the required duration.
            "-t",
            str(self.audio_duration),

            # Disable video/subtitle/data streams.
            "-vn",
            "-sn",
            "-dn",

            # Mono
            "-ac",
            "1",

            # 16 kHz
            "-ar",
            str(self.sample_rate),

            # Raw float32 PCM
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
                "FFmpeg audio extraction failed:\n"
                f"{video_path}\n"
                f"{error_message}"
            )

        audio = np.frombuffer(
            result.stdout,
            dtype=np.float32
        ).copy()

        if len(audio) == 0:
            raise RuntimeError(
                f"No audio found in: {video_path}"
            )

        # --------------------------------------------------------
        # FIX AUDIO LENGTH
        # --------------------------------------------------------

        if len(audio) < self.num_audio_samples:

            audio = np.pad(
                audio,
                (
                    0,
                    self.num_audio_samples - len(audio)
                ),
                mode="constant"
            )

        else:

            audio = audio[
                :self.num_audio_samples
            ]

        audio = np.ascontiguousarray(
            audio
        )

        return torch.from_numpy(
            audio
        ).float()

    # ============================================================
    # GET ITEM
    # ============================================================

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        relative_path = row["file"]

        video_path = self._get_video_path(
            relative_path
        )

        if not os.path.exists(video_path):

            raise FileNotFoundError(
                f"Video does not exist:\n"
                f"{video_path}"
            )

        # --------------------------------------------------------
        # VIDEO
        # --------------------------------------------------------

        frames = self._load_frames(
            video_path
        )

        # --------------------------------------------------------
        # AUDIO
        # --------------------------------------------------------

        audio = self._load_audio(
            video_path
        )

        # --------------------------------------------------------
        # LABEL
        # --------------------------------------------------------

        label = int(
            row["label"]
        )

        return {
            "frames": frames,

            "audio": audio,

            "label": torch.tensor(
                label,
                dtype=torch.long
            ),

            "file": relative_path,
        }


# ================================================================
# DATASET TEST
# ================================================================

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