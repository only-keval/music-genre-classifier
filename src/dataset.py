from pathlib import Path

import torch

from torch.utils.data import Dataset


class FMADataset(Dataset):
    def __init__(
        self,
        tracks_df,
        genre_to_idx,
        preprocessed_root,
    ):
        self.tracks = tracks_df
        self.genre_to_idx = genre_to_idx

        self.preprocessed_root = Path(
            preprocessed_root
        )

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        sample = self.tracks.iloc[idx]

        track_id = sample.name
        genre = sample["track"]["genre_top"]

        tensor_path = (
            self.preprocessed_root
            / f"{track_id:06d}.pt"
        )

        x = torch.load(
            tensor_path,
            weights_only=True,
        )

        y = torch.tensor(
            self.genre_to_idx[genre],
            dtype=torch.long,
        )

        return x, y