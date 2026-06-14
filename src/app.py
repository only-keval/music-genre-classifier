from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import streamlit as st
import torch

from inference import (
    load_model,
    predict,
)
from preprocess import create_spectrogram

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
)


@st.cache_resource
def get_model():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, idx_to_genre = load_model(
        device
    )

    return (
        model,
        idx_to_genre,
        device,
    )


st.title("🎵 Music Genre Classifier")

st.caption(
    "Upload an MP3 or WAV file and let the CNN predict its genre."
)

uploaded_file = st.file_uploader(
    "Audio File",
    type=["mp3", "wav"],
)

if uploaded_file is not None:
    (
        model,
        idx_to_genre,
        device,
    ) = get_model()

    with NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ) as tmp:
        tmp.write(
            uploaded_file.getbuffer()
        )

        audio_path = Path(tmp.name)

    try:
        st.audio(uploaded_file)

        with st.spinner(
            "Generating spectrogram..."
        ):
            spectrogram = create_spectrogram(
                audio_path
            )

        with st.spinner(
            "Running inference..."
        ):
            predictions = predict(
                model,
                idx_to_genre,
                audio_path,
                device,
            )

        predictions.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        top_genre, top_prob = (
            predictions[0]
        )

        st.markdown("---")

        col1, col2 = st.columns(
            [3, 2]
        )

        with col1:
            st.subheader(
                "Mel Spectrogram"
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.imshow(
                spectrogram.squeeze().numpy(),
                origin="lower",
                aspect="auto",
            )

            ax.set_xlabel(
                "Time Frames"
            )

            ax.set_ylabel(
                "Mel Bands"
            )

            ax.set_title(
                "Input Spectrogram"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        with col2:
            st.subheader(
                "Prediction"
            )

            st.metric(
                label="Predicted Genre",
                value=top_genre,
            )

            st.metric(
                label="Confidence",
                value=f"{top_prob * 100:.2f}%",
            )

            st.markdown(
                "### Genres"
            )

            for genre, prob in predictions:
                genre_col, bar_col, pct_col = st.columns(
                    [2, 6, 1]
                )

                with genre_col:
                    st.write(
                        f"**{genre}**"
                    )

                with bar_col:
                    st.progress(
                        float(prob)
                    )

                with pct_col:
                    st.write(
                        f"{prob * 100:.1f}%"
                    )

    finally:
        audio_path.unlink(
            missing_ok=True
        )