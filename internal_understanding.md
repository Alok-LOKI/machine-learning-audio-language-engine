# Audio-Language Engine: Internal Guide

This document is for your team to understand exactly how the project is structured, what technologies are used, and what we fixed to get it working perfectly.

## 1. Overall Architecture
The project is a **Full-Stack Deep Learning Application**. 
- **Backend:** A Python web server built with **FastAPI**. It loads a PyTorch Dual-Encoder neural network into memory. It has API endpoints (like `/generate_caption` or `/search`) that accept requests from the website, run the AI model, and return the results.
- **Frontend:** A pure HTML/CSS/JavaScript interface. It sends requests to the backend API without reloading the page, giving it a modern, app-like feel.
- **Data & Training:** The Jupyter Notebook (`Project_15_WavCaps.ipynb`) is where the actual model architecture, data analysis, and training/evaluation loops live. 

## 2. What We Fixed & Upgraded
During our session, we solved several critical issues:

### A. The Data Loading Bugs (Jupyter Notebook)
- **The Issue:** The notebook was originally crashing because it was looking for `true_caption` and `noisy_caption` in the JSON dataset, but the dataset actually used the keys `caption` and `description`. This resulted in empty strings, causing a variance of `0` and breaking the Seaborn density plots.
- **The Fix:** We wrote a script to correct the keys inside the notebook cells and removed a duplicate code block that was causing the graphs to render twice. The notebook now executes perfectly from start to finish.

### B. UI/UX Design & Aesthetic
- **The Issue:** The original matrix effect and styling didn't quite hit the "premium robotic" aesthetic required for a modern AI showcase. 
- **The Fix:** We completely overhauled the CSS to a sleek monochrome palette (Black, White, Gray) and programmed a custom HTML5 Canvas animation. The background now features subtle falling data streams overlaid with a glowing green "Robotic Neural Mesh" that connects floating particles.

### C. Transparency & Notebook Integration
- **The Issue:** We wanted to embed the Jupyter Notebook directly into the website so viewers could read the source code. However, Jupyter's default HTML export has solid white backgrounds that clashed with our dark robotic theme.
- **The Fix:** We ran a custom Python script that forced the injected Jupyter HTML to use `data-theme="dark"` and completely stripped away its background colors (`background: transparent !important`). This allowed our beautiful canvas animation to show *through* the source code without breaking the Python syntax highlighting.

## 3. How the AI Works (Simple Explanation)
The model is a **Dual-Encoder**.
1. **Audio Encoder:** Listens to an audio file and converts its features (Mel-Spectrograms) into a mathematical vector (an embedding).
2. **Text Encoder:** Reads a caption and converts it into a vector.
3. **Contrastive Loss:** During training, the model learns to push the Audio Vector and the Text Vector of matching pairs close together in mathematical space.
When a user searches for "A dog barking", the model converts that text into a vector, and finds the nearest audio vector in its database!
