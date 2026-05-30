from pathlib import Path

import pandas as pd
import torch

from dataset import FMADataset
from model import GenreCNN
from metadata import load_fma_small_metadata, build_label_mapping


def main():
    print("Loading metadata...")

    small_tracks = load_fma_small_metadata()
    genre_to_idx, idx_to_genre = build_label_mapping(small_tracks)

    print(f"Tracks: {len(small_tracks)}")
    print(f"Genres: {len(genre_to_idx)}")

    dataset = FMADataset(
        tracks_df=small_tracks,
        genre_to_idx=genre_to_idx,
        audio_root="data/fma_small",
    )

    print(f"Dataset size: {len(dataset)}")

    print("\nLoading first sample...")

    x, y = dataset[0]

    print(f"x shape: {x.shape}")
    print(f"x shape: {x.shape}")
    print(f"label: {y.item()}")
    print(f"genre: {idx_to_genre[y.item()]}")

    model = GenreCNN(
        num_classes=len(genre_to_idx)
    )

    print("\nRunning forward pass...")

    x_batch = x.unsqueeze(0)

    print(f"batch shape: {x_batch.shape}")

    out = model(x_batch)

    print(f"output shape: {out.shape}")
    print(f"logits:\n{out}")

    pred = torch.argmax(
        out,
        dim=1,
    )

    print(
        f"\npredicted genre: "
        f"{idx_to_genre[pred.item()]}"
    )


if __name__ == "__main__":
    main()