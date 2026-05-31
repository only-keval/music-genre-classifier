import torch
from torch.utils.data import DataLoader

from dataset import FMADataset
from metadata import (
    load_fma_small_metadata,
    build_label_mapping,
)
from model import GenreCNN


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 5


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Load metadata
    tracks = load_fma_small_metadata()

    genre_to_idx, idx_to_genre = (
        build_label_mapping(tracks)
    )

    print(f"Tracks: {len(tracks)}")
    print(f"Genres: {len(genre_to_idx)}")

    # Dataset
    dataset = FMADataset(
        tracks_df=tracks,
        genre_to_idx=genre_to_idx,
        audio_root="data/fma_small",
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = GenreCNN(
        num_classes=len(genre_to_idx)
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Batch {batch_idx}/{len(loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch + 1} complete "
            f"- Avg Loss: {avg_loss:.4f}"
        )

    print("Training complete")


if __name__ == "__main__":
    main()