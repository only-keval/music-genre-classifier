import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import FMADataset
from metadata import (
    load_fma_small_metadata,
    build_label_mapping,
)
from model import GenreCNN

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100


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
        # weight_decay=1e-3,
    )

    print("Starting training...")

    loss_history = []
    train_acc_history = []
    test_acc_history = []

    plt.style.use("dark_background")

    plt.rcParams["figure.facecolor"] = "#1e1e1e"
    # plt.rcParams["axes.facecolor"] = "#252526"
    plt.rcParams["axes.facecolor"] = "#1e1e1e"
    plt.rcParams["savefig.facecolor"] = "#1e1e1e"

    plt.rcParams["grid.color"] = "#3c3c3c"
    plt.rcParams["grid.alpha"] = 0.6

    plt.rcParams["axes.edgecolor"] = "#808080"
    plt.rcParams["xtick.color"] = "#d4d4d4"
    plt.rcParams["ytick.color"] = "#d4d4d4"
    plt.rcParams["text.color"] = "#d4d4d4"
    plt.rcParams["axes.labelcolor"] = "#d4d4d4"

    loss_color = "#4e8cff"
    train_color = "#00c29b"
    test_color = "#c3890d"

    plt.ion()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    fig.tight_layout(pad=3.0)

    fig.show()
    plt.show(block=False)
    plt.pause(0.1)

    best_test_accuracy = 0.0
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        train_correct = 0
        train_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            predictions = logits.argmax(dim=1)

            train_correct += (
                predictions == y
            ).sum().item()

            train_total += y.size(0)

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
            if batch_idx % 20 == 0:
                plt.pause(0.001)

        avg_loss = running_loss / len(train_loader)

        train_accuracy = (
            train_correct / train_total
        )

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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                MODELS_DIR / "best_model.pt",
            )

            print(
                f"New best model saved "
                f"(accuracy={best_test_accuracy:.4f})"
            )

        loss_history.append(avg_loss)
        train_acc_history.append(train_accuracy)
        test_acc_history.append(test_accuracy)

        print(
            f"Epoch {epoch + 1} complete"
            f" - Avg Loss: {avg_loss:.4f}"
            f" - Train Accuracy: {train_accuracy:.4f}"
            f" - Test Accuracy: {test_accuracy:.4f}"
        )

        epochs = range(
            1,
            len(loss_history) + 1,
        )

        ax1.clear()
        ax1.plot(
            epochs,
            loss_history,
            color=loss_color,
        )
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        ax2.clear()
        ax2.plot(
            epochs,
            train_acc_history,
            label="Train Accuracy",
            color=train_color,
        )
        ax2.plot(
            epochs,
            test_acc_history,
            label="Test Accuracy",
            color=test_color,
        )
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.draw()
        plt.pause(0.01)

    print("Training complete")
    print(
        f"Best test accuracy: "
        f"{best_test_accuracy:.4f}"
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()