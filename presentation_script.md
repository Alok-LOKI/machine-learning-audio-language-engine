# Presentation Script (3 Members)
**Team Members:** ALOK, Dennis, Apoorv

---

## Slide 1: Title Slide
**(Speaker: ALOK)**
"Hello everyone, our team consists of myself, Alok, alongside Dennis and Apoorv. Today we are excited to present our project: The Audio-Language Pretraining Engine. This is a fully integrated, full-stack machine learning application designed to bridge the gap between acoustic sound and natural human language."

---

## Slide 2: Project Objective
**(Speaker: ALOK)**
"To understand the problem we are solving: most modern AI systems are very good at understanding images and text, but audio is often left behind. Our goal was to build a system that can understand the semantic meaning of sound. 
By using weakly supervised learning on a dataset of noisy captions, we built an engine that lets you type a natural sentence—like 'A dog barking'—and the AI will instantly search its database to retrieve the matching audio file based purely on its acoustic features."

---

## Slide 3: System Architecture
**(Speaker: Apoorv)**
"I'll take over to explain how we actually built this. We didn't just train a model; we built a complete production pipeline.
On the frontend, we designed a custom 'robotic style' web interface using HTML and Vanilla JavaScript, featuring dynamic background animations and live API querying.
On the backend, we used FastAPI to run a high-speed Python server. This server loads our Dual-Encoder neural network into memory—which combines DistilBERT for text processing and an Audio Spectrogram Transformer for the sound processing—allowing for real-time inference when a user submits a search."

---

## Slide 4: Data & Training (The Jupyter Notebook)
**(Speaker: Dennis)**
"For the actual Machine Learning, everything was prototyped and trained inside a Jupyter environment, which we've transparently embedded into our web app.
We used a subset of the WavCaps and SoundBible datasets. We first had to convert raw audio files into Mel-Spectrograms so the neural network could 'see' the audio.
We then trained the model using a technique called Contrastive Loss. Basically, during training, the model mathematically pushes the embeddings of matching audio and text pairs together, while pushing the embeddings of mismatched pairs apart in a shared latent space."

---

## Slide 5: Evaluation & Metrics
**(Speaker: Dennis)**
"To prove the model actually works, we evaluated it using Zero-Shot Retrieval Metrics. 
Because simple accuracy doesn't work for search engines, we used Recall@1, Recall@5, and Recall@10. This measures what percentage of the time the correct audio file appeared in the top 1, top 5, or top 10 search results.
Our trained model significantly outperformed random chance baselines, proving that our contrastive learning loop successfully taught the AI to map sound to language."

---

## Slide 6: Demonstration & Conclusion
**(Speaker: ALOK)**
"To wrap things up, we've deployed this model into the live web application you've seen in the screenshots. It offers a seamless, interactive experience where users can view the dataset, run live text-to-audio retrievals, and inspect the Jupyter source code directly from the browser.
In the future, we hope to scale this to much larger datasets and potentially add real-time microphone support so the AI can caption the world around it live. 
Thank you for listening, we are happy to take any questions!"
