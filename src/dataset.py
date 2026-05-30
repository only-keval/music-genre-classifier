from pathlib import Path

import librosa
import numpy as np
import torch

from torch.utils.data import Dataset

from contants import SAMPLE_RATE, TARGET_FRAMES, N_MELS


class FMADataset(Dataset):
    def __init__(
        self,
        tracks_df,
        genre_to_idx,
        audio_root,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
    ):
        self.tracks = tracks_df
        self.genre_to_idx = genre_to_idx

        self.audio_root = Path(audio_root)

        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        sample = self.tracks.iloc[idx]

        track_id = sample.name
        genre = sample["track"]["genre_top"]

        track_str = f"{track_id:06d}"

        audio_path = (
            self.audio_root
            / track_str[:3]
            / f"{track_str}.mp3"
        )

        y_audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
        )

        mel = librosa.feature.melspectrogram(
            y=y_audio,
            sr=sr,
            n_mels=self.n_mels,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        x = torch.tensor(
            mel_db,
            dtype=torch.float32,
        ).unsqueeze(0)

        if x.shape[2] > TARGET_FRAMES:
            x = x[:, :, :TARGET_FRAMES]

        y = torch.tensor(
            self.genre_to_idx[genre],
            dtype=torch.long,
        )

        return x, y