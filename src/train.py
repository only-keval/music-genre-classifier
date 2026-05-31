import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import FMADataset
from metadata import (
    load_fma_small_metadata,
    build_label_mapping,
)
from model import GenreCNN


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50


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

    train_tracks, test_tracks = train_test_split(
        tracks,
        test_size=0.2,
        random_state=42,
        stratify=tracks["track"]["genre_top"],
    )

    # Dataset
    train_dataset = FMADataset(
        tracks_df=train_tracks,
        genre_to_idx=genre_to_idx,
        preprocessed_root="data/preprocessed",
    )

    test_dataset = FMADataset(
        tracks_df=test_tracks,
        genre_to_idx=genre_to_idx,
        preprocessed_root="data/preprocessed",
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
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

        for batch_idx, (x, y) in enumerate(train_loader):
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

            # if batch_idx % 20 == 0:
            #     print(
            #         f"Epoch {epoch + 1}/{EPOCHS} "
            #         f"Batch {batch_idx}/{len(train_loader)} "
            #         f"Loss: {loss.item():.4f}"
            #     )

        avg_loss = running_loss / len(train_loader)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)

                predictions = logits.argmax(dim=1)

                correct += (
                    predictions == y
                ).sum().item()

                total += y.size(0)

        test_accuracy = correct / total

        print(
            f"Epoch {epoch + 1} complete"
            f" - Avg Loss: {avg_loss:.4f}"
            f" - Test Accuracy: {test_accuracy:.4f}"
        )

    print("Training complete")


if __name__ == "__main__":
    main()