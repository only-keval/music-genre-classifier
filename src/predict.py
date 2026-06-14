import argparse
from pathlib import Path

import torch

from inference import load_model, predict

def main():
    parser = argparse.ArgumentParser(
        description="Predict music genre"
    )

    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to audio file",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, idx_to_genre = load_model(
        device
    )

    predictions = predict(
        model,
        idx_to_genre,
        args.audio_path,
        device,
    )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("Predictions:")
    print()

    for genre, prob in predictions:
        print(
            f"{genre:<12}"
            f"{prob * 100:.2f}%"
        )


if __name__ == "__main__":
    main()