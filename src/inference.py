from pathlib import Path
import math

import torch
import librosa

from model import GenreCNN
from preprocess import create_spectrogram_from_audio

from constants import SAMPLE_RATE, CLIP_DURATION

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
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    clip_logits = []
    duration = int(librosa.get_duration(y=audio))
    for i in range(math.ceil(duration / CLIP_DURATION)):
        end_sec = min((i + 1) * CLIP_DURATION, duration)
        start_sec = max(0, end_sec - CLIP_DURATION)
        start, end = start_sec * SAMPLE_RATE, end_sec * SAMPLE_RATE
        x = create_spectrogram_from_audio(audio[start:end])

        x = x.unsqueeze(0)
        x = x.to(device)

        with torch.no_grad():
            logits = model(x)
            clip_logits.append(logits)


    mean_logits = torch.stack(clip_logits).mean(dim=0)

    probs = torch.softmax(
        mean_logits,
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
