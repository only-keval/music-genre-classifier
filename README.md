# Music Genre Classifier

A deep learning project that classifies music tracks into genres using Mel spectrograms and a Convolutional Neural Network (CNN) trained on the Free Music Archive (FMA) Small dataset.

## Live Demo

🎵 https://music-genre-classifier-keval.streamlit.app/

Upload an MP3 or WAV file and the model will predict its genre, display confidence scores, and visualize the generated Mel spectrogram.

## Features

* Music genre classification using PyTorch
* Mel spectrogram preprocessing with Librosa
* Interactive Streamlit web application
* Real-time genre prediction
* Confidence score visualization
* Mel spectrogram visualization
* Support for MP3 and WAV audio files

## How It Works

1. Audio is loaded and converted into a Mel spectrogram using Librosa.
2. The spectrogram is converted to decibel scale and resized to a fixed shape.
3. The spectrogram is passed through a Convolutional Neural Network.
4. The model outputs genre probabilities for the 8 supported genres.
5. The Streamlit app displays the top predictions and confidence scores.

## Dataset

This project uses the Free Music Archive (FMA) Small dataset.

Dataset repository:

https://github.com/mdeff/fma

The model is trained to classify the following genres:

* Electronic
* Experimental
* Folk
* Hip-Hop
* Instrumental
* International
* Pop
* Rock

## Model Architecture

```text
Input (Mel Spectrogram)
    ↓
Conv2D(1 → 32)
    ↓
ReLU
    ↓
MaxPool2D

Conv2D(32 → 64)
    ↓
ReLU
    ↓
MaxPool2D

Conv2D(64 → 128)
    ↓
ReLU
    ↓
MaxPool2D

AdaptiveAvgPool2D(1×1)
    ↓
Flatten
    ↓
Linear(128 → 128)
    ↓
ReLU
    ↓
Dropout(0.1)
    ↓
Linear(128 → 8)
```

## Results

The best model achieved approximately **61% test accuracy** on the FMA Small dataset.

The model was trained from scratch using PyTorch on full 30-second tracks represented as Mel spectrograms.

## Project Structure

```text
.
├── data/
├── models/
│   └── best_model.pt
├── src/
│   ├── app.py
│   ├── constants.py
│   ├── dataset.py
│   ├── inference.py
│   ├── metadata.py
│   ├── model.py
│   ├── preprocess.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd music-genre-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Web App

```bash
streamlit run src/app.py
```

Then open the local URL displayed by Streamlit in your browser.

## Running Inference from the Command Line

```bash
python src/inference.py path/to/song.mp3
```

Example:

```bash
python src/inference.py my_song.mp3
```

## Training

Train the model:

```bash
python src/train.py
```

The best-performing model checkpoint is automatically saved during training.

## Technologies Used

* Python
* PyTorch
* Streamlit
* Librosa
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Future Improvements

* Experiment with clip-based training and prediction aggregation
* Data augmentation for improved generalization
* Hyperparameter optimization
* Confusion matrix and evaluation dashboard
* Transfer learning using pretrained audio models

## License

This project is intended for educational and portfolio purposes.
