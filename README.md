# Audio-Language Pretraining Engine

**Project 15: Weakly Supervised Audio-Language Pretraining on Noisy Captions**

![UI Screenshot](static/notebook.html) *(See live interactive UI by running the server)*

This project explores weakly supervised audio-language pretraining using contrastive learning. By leveraging noisy captions from datasets like WavCaps, the model maps raw audio waveforms and natural language text into a shared multimodal latent space. This allows for zero-shot text-to-audio retrieval.

**Course:** Machine Learning  
**Team Members:** Alok, Apoorv, Dennis  
**Date:** 28 June 2026

## Features

- **Text-to-Audio Retrieval:** Find the most relevant audio samples in the dataset using natural language queries (e.g., "A dog is barking loudly").
- **Contrastive Learning (InfoNCE):** A dual-encoder architecture utilizing a CNN14/HTSAT audio encoder and a DistilBERT text encoder.
- **Matrix-Themed Web UI:** A fully custom, interactive terminal-style frontend complete with visualizations, analytics, and an integrated dataset gallery.
- **FastAPI Backend:** A lightweight, high-performance Python backend serving the model and UI.
- **Zero-Shot Evaluation:** Robust metrics including Recall@1, Recall@5, Recall@10, and mAP@10 on validation splits.

## Project Structure

- `app.py`: The FastAPI web server that handles API endpoints, model caching, and static file serving.
- `run_server.py`: The entry point to start the local web server using Uvicorn.
- `static/`: Contains the frontend HTML (`index.html`), CSS styling, and JavaScript logic, along with dynamically generated spectrograms and audio clips.
- `15_wavecaps.ipynb` / `Project_15_WavCaps.ipynb`: The core Jupyter notebooks containing the data processing pipeline, model architecture, training loop, and evaluation scripts.
- `build_notebook.py`: A utility script to programmatically generate the project notebook format.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Alok-LOKI/machine-learning-audio-language-engine.git
   cd machine-learning-audio-language-engine
   ```

2. **Install Dependencies:**
   Make sure you have Python installed, then install the required libraries:
   ```bash
   pip install fastapi uvicorn torch torchaudio soundfile numpy matplotlib
   ```
   *(Note: Depending on your environment, you may also need `transformers` and `huggingface_hub` for the language models).*

3. **Run the Server:**
   Start the local backend server:
   ```bash
   python run_server.py
   ```
   
4. **View the Application:**
   Open your browser and navigate to `http://127.0.0.1:8000` to interact with the Audio-Language Engine.

## Methodology

Our system utilizes a weakly supervised dual-encoder contrastive learning framework. By learning from noisy web captions, we encode raw waveforms into a shared multimodal acoustic-semantic latent space.
- **Audio Encoder:** Extracts frame-level acoustic features mapped to an aggregate embedding vector.
- **Text Encoder:** Produces sequence-pooled text embeddings.
- **Objective Function:** InfoNCE loss to maximize cosine similarity of matched audio-text pairs while pushing negative pairs apart.

## License

This project was developed for an academic Machine Learning course.
