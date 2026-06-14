from pathlib import Path

import torch

from model import GenreCNN
from preprocess import create_spectrogram

MODEL_PATH = Path("models/best_model_v3_61.pt")

def load_model(device):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    idx_to_genre = checkpoint["idx_to_genre"]

    model = GenreCNN(
        num_classes=len(idx_to_genre),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, idx_to_genre


def predict(
    model,
    idx_to_genre,
    audio_path,
    device,
):
    x = create_spectrogram(audio_path)

    x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1,
        )

    return [
        (
            idx_to_genre[i],
            probs[0, i].item(),
        )
        for i in range(
            len(idx_to_genre)
        )
    ]
