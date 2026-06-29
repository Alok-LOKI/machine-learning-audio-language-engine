# Audio-Language Pretraining Engine

## Presentation Slide Outline

You can copy and paste the following content directly into your PowerPoint slides.

---

### Slide 1: Title Slide

**Title:** Audio-Language Pretraining Engine
**Subtitle:** Weakly Supervised Audio-Language Retrieval & Captioning
**Team Members:** ALOK, Dennis, Apoorv
*(Suggested Visual: A screenshot of the glowing green robotic mesh background from the web app)*

---

### Slide 2: Project Objective

**Heading:** The Problem We Are Solving
**Bullet Points:**

- Modern AI often focuses solely on text or vision.
- We built an engine that understands the semantic meaning of **Audio**.
- **Goal:** Allow users to search for audio using natural text queries (Retrieval) and generate text descriptions of audio files (Captioning).
- Built using Weak Supervision, meaning the AI learns from noisy, real-world data without perfect human labels.

---

### Slide 3: System Architecture

**Heading:** Full-Stack Deep Learning Pipeline
**Bullet Points:**

- **Frontend (Web App):** Modern, interactive dashboard with a custom robotic mesh background (HTML5/Vanilla JS).
- **Backend (FastAPI):** High-speed asynchronous Python server handling model inference.
- **AI Model (Dual-Encoder):** Combines a DistilBERT text encoder with an Audio Spectrogram Transformer.
  *(Suggested Visual: Screenshot of the Analytics Dashboard or the Text-to-Audio Retrieval search bar)*

---

### Slide 4: Data & Training (The Jupyter Notebook)

**Heading:** Model Training & Contrastive Loss
**Bullet Points:**

- Processed the WavCaps/SoundBible dataset.
- Converted raw audio waveforms into Mel-Spectrograms.
- **Contrastive Learning:** The model is trained to push matching Audio/Text vector embeddings together and push mismatched pairs apart.
  *(Suggested Visual: Screenshot of the transparent Jupyter Notebook embedded inside the web app)*

---

### Slide 5: Evaluation & Metrics

**Heading:** How We Measure Success
**Bullet Points:**

- We evaluated the model using **Zero-Shot Retrieval Metrics**.
- **Recall@1, Recall@5, Recall@10:** Measures what percentage of the time the correct audio file was in the top 1, 5, or 10 search results.
- **Mean Rank:** The average position of the correct file.
- Outperformed a random baseline significantly, proving successful multimodal mapping!
  *(Suggested Visual: Screenshot of the live Evaluation Metrics bar charts)*

---

### Slide 6: Demonstration & Conclusion

**Heading:** Live Web Application Demo
**Bullet Points:**

- Seamless UI for real-time Audio Retrieval.
- Transparent integration of data science workflows into a user-friendly product.
- **Future Work:** Scaling to larger datasets, deploying to the cloud, and adding real-time microphone support.
  **Footer:** Thank You! Questions?
