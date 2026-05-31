from pathlib import Path

import librosa
import pandas as pd
import torch
import torch.nn.functional as F

from metadata import load_fma_small_metadata
from constants import N_MELS, SAMPLE_RATE, TARGET_FRAMES

AUDIO_ROOT = Path("data/fma_small")
OUTPUT_ROOT = Path("data/preprocessed")


def create_spectrogram(audio_path: Path) -> torch.Tensor:
    y_audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
    )

    mel = librosa.feature.melspectrogram(
        y=y_audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=mel.max(),
    )

    x = torch.tensor(
        mel_db,
        dtype=torch.float32,
    )

    x = x.unsqueeze(0)

    frames = x.shape[2]

    if frames > TARGET_FRAMES:
        x = x[:, :, :TARGET_FRAMES]

    elif frames < TARGET_FRAMES:
        pad = TARGET_FRAMES - frames

        x = F.pad(
            x,
            (0, pad),
        )

    return x


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracks = load_fma_small_metadata()

    processed = 0
    failed = 0

    for track_id in tracks.index:
        audio_path = (
            AUDIO_ROOT
            / f"{track_id:06d}"[:3]
            / f"{track_id:06d}.mp3"
        )

        output_path = (
            OUTPUT_ROOT
            / f"{track_id:06d}.pt"
        )

        if output_path.exists():
            continue

        try:
            x = create_spectrogram(
                audio_path
            )

            torch.save(
                x,
                output_path,
            )

            processed += 1

            if processed % 100 == 0:
                print(
                    f"Processed {processed}"
                )

        except Exception as e:
            failed += 1

            print(
                f"Failed {track_id}: {e}"
            )

    print()
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()