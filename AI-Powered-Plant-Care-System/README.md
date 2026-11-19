# 🌿 AI-Powered Plant Care System

A multi-page Streamlit application that combines machine learning, content-based recommendations, and a retrieval-augmented chatbot to support everyday plant care.

## Features

- **Plant Symptom Diagnosis:** Predicts a likely plant issue from selected symptoms using a Random Forest model and displays care information.
- **Photo-Based Diagnosis:** Accepts plant photos and uses a Groq vision-capable model to describe or classify visible issues. Requires a Groq API key.
- **Fertilizer Recommendation:** Uses TF-IDF and cosine similarity to find similar fertilizer or plant-care products.
- **Plant Stress Risk Assessment:** Estimates stress risk from plant environment and care inputs using a Random Forest pipeline and feature-importance information.
- **PlantBot:** Answers plant-care questions using a RAG workflow built with LangChain, FAISS, Hugging Face embeddings, and Groq-hosted Llama 3.3.
- **User Accounts and Garden Dashboard:** Provides login/registration and stores garden profiles and activity history locally.

## Tech Stack

Python · Streamlit · Pandas · NumPy · Scikit-learn · Random Forest · TF-IDF · Cosine Similarity · LangChain · FAISS · Hugging Face Embeddings · Groq API

## Project Structure

```text
AI-Powered-Plant-Care-System/
├── 0_home.py
├── auth.py
├── garden_db.py
├── pages/
├── plantbot/
├── models/
├── data/
├── issue_images/
├── scripts/
├── utils/
├── requirements.txt
├── .env.example
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Powered-Plant-Care-System.git
cd AI-Powered-Plant-Care-System
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API keys

Copy `.env.example` to `.env` and add your own keys:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

`GROQ_API_KEY` is required for PlantBot and photo-based diagnosis. Keep `.env` private; never commit API keys to GitHub. `HF_TOKEN` is only needed for Hugging Face-hosted model access as described in the code.

### 5. Start the application

```bash
streamlit run 0_home.py
```

Open the local URL shown in the terminal, register an account, and sign in.

## Models and Data

The repository includes the datasets and trained model artifacts used by the application under `data/` and `models/`. The model-building script is available at `scripts/build_models.py`.

To rebuild generated datasets and models:

```bash
python scripts/build_models.py
```

The PlantBot FAISS index is created locally when needed from the bundled plant-care knowledge files. The first run may download the embedding model.

## Important Notes

- This is an educational project, not professional horticultural or agricultural advice.
- The stress-risk training data and fertilizer recommendation data are project datasets; outputs should be treated as indicative suggestions, not validated scientific predictions.
- User account and garden-history data are stored locally. Do not commit personal user data or local database files.
- Do not upload real API keys, `.env` files, or private credentials.

## Author

**Smit Gadhiya**  
MCA, SRM Institute of Science and Technology  
LinkedIn: https://www.linkedin.com/in/smit-gadhiya/
