# Audio-Language Pretraining Engine
**Project Assignment Report**

## 1. Project Objective
The goal of this project was to develop a weakly supervised Audio-Language Engine capable of performing complex multimodal tasks: **Text-to-Audio Retrieval** and **Zero-Shot Audio Captioning**. Rather than relying on perfectly annotated datasets, the model is designed to learn from noisy, weakly aligned captions to map acoustic features into the same latent space as natural language.

## 2. System Architecture
The application is structured into a modern full-stack pipeline:
* **Frontend UI**: A custom-designed, dark-themed robotic interface built using HTML5, CSS3, and Vanilla JavaScript. It features dynamic API querying, real-time audio visualizers, and a computationally generated HTML5 Canvas background displaying a neural mesh network.
* **Backend API**: A highly optimized **FastAPI** Python server that manages asynchronous requests, processes audio data, and routes it through our neural networks.
* **Core Model**: The machine learning backbone leverages a Dual-Encoder architecture (combining a DistilBERT text encoder with a specialized Audio Spectrogram Transformer) trained using Contrastive Loss.

## 3. Methodology & Implementation
1. **Data Preprocessing**: We utilized a subset of the WavCaps/SoundBible dataset. The JSON metadata was parsed to extract Mel-Spectrograms and align them with both their true descriptions and computationally generated noisy captions.
2. **Training & Contrastive Learning**: A training loop was established inside a Jupyter Notebook environment. By utilizing contrastive loss (InfoNCE), the model iteratively learned to maximize the cosine similarity between corresponding audio-text pairs while pushing apart mismatched pairs.
3. **Web Integration**: The Jupyter Notebook containing the data science workflow was exported, programmatically styled for a transparent dark-mode aesthetic, and seamlessly embedded into the frontend dashboard for peer review.

## 4. Evaluation & Results
The model was evaluated using standard retrieval metrics:
* **Recall@1 (R@1)**
* **Recall@5 (R@5)**
* **Recall@10 (R@10)**

By analyzing the Top-K retrieval accuracy against a random baseline, the engine successfully demonstrated that acoustic properties can be effectively mapped to semantic textual embeddings, allowing users to type a natural language query (e.g., "A dog barking loudly") and instantly retrieve the correct audio file.
