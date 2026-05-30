from pathlib import Path

import pandas as pd


def load_fma_small_metadata(
    metadata_path="data/fma_metadata/tracks.csv",
    audio_root="data/fma_small",
):
    tracks = pd.read_csv(
        metadata_path,
        header=[0, 1],
        index_col=0,
    )

    audio_files = list(
        Path(audio_root).rglob("*.mp3")
    )

    track_ids = {
        int(file.stem)
        for file in audio_files
    }

    small_tracks = tracks.loc[
        tracks.index.isin(track_ids)
    ]

    return small_tracks


def build_label_mapping(tracks_df):
    genres = sorted(
        tracks_df["track"]["genre_top"]
        .dropna()
        .unique()
    )

    genre_to_idx = {
        genre: idx
        for idx, genre in enumerate(genres)
    }

    idx_to_genre = {
        idx: genre
        for genre, idx in genre_to_idx.items()
    }

    return genre_to_idx, idx_to_genre