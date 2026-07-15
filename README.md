# 🎬 Cross-Domain Movie Recommender System (Books → Movies)

A deep learning-based Cross-Domain Recommendation System that transfers user preferences and tastes from the **Book Domain** (source) to the **Movie Domain** (target) using a trained neural network (**Branch B Model**). 

The project includes an interactive web interface built with **Streamlit**, featuring speech-to-text input, dynamic movie title resolution, and robust session state form management.

---

## 🚀 Key Features

* **Cross-Domain Transfer Learning**: Recommends movies by encoding your book reviews and processing them through a dual-encoder Branch B neural network model.
* **Dual Speech Input (Groq Whisper API)**: Speak your book titles and reviews directly into the app using your microphone for instant, highly accurate transcription.
* **Smart Movie Title Resolution**:
  * Resolves raw Amazon ASINs (product IDs) to official movie titles (e.g. *Madea Goes to Jail (The Tyler Perry Collection)*).
  * Uses a pre-populated offline cache parsed from the `meta_Movies_and_TV.jsonl` dataset.
  * Falls back to dynamic direct Amazon page fetching or DuckDuckGo search to handle unseen products without breaking the interface.
* **Robust Form State Syncing**: Supports adding and deleting book reviews dynamically, preserving text changes and shifting all widget states properly when items are removed.

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application
├── model.ipynb                 # Jupyter notebook for model training & evaluation
├── preprocess.ipynb            # Jupyter notebook for dataset preprocessing
├── best_branch_b_v2.pt         # Trained Branch B model checkpoint (PyTorch)
├── .gitignore                  # Git exclusions (large datasets, secrets, cache)
├── data/
│   └── processed/
│       └── asin_to_movie_title.json  # Pre-compiled local movie title cache
└── final_test_results_v2.json  # Model performance metrics (RMSE, MAE, etc.)
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/naveen27022005/Cross-Domain-Recommendation-System.git
cd Cross-Domain-Recommendation-System
```

### 2. Install dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install streamlit torch numpy pandas transformers groq beautifulsoup4
```

### 3. Add API Keys & Large Datasets (Not in Git)
* **Groq API Key**: Create a file named `api_key.txt` in the root folder and write:
  ```text
  api_key_groq = YOUR_GROQ_API_KEY
  ```
  *(This file is git-ignored to prevent leaking your credentials).*
* **Metadata file**: If you wish to rebuild the title mapping cache from scratch, place the 1.28 GB `meta_Movies_and_TV.jsonl` file in the root directory.

---

## 🏃 Running the Application

Launch the Streamlit interface locally by running:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Model Information

The core neural network architecture implements **Branch B** of a cross-domain collaborative filtering network. It takes user embeddings generated from reviews (via a pre-trained `distilbert-base-uncased` transformer model) and item embeddings to predict target domain ratings.

**Performance Metrics** (trained model `best_branch_b_v2.pt`):
* **Validation RMSE**: 0.8229
* **Validation MAE**: 0.6130
* **Prediction Std Dev**: 0.2944
