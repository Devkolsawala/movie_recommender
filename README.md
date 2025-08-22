
#  🌐 Live Demo: Movie Recommender Demo

# 🎬 AI-Based Multi-Genre Movie Recommendation System

An advanced content-based movie recommender system that uses deep semantic understanding, genre alignment, popularity metrics, and rating similarity to generate highly accurate and diverse movie recommendations.

---

## 🚀 Features

- ✅ **Semantic Matching with Transformers** using `all-mpnet-base-v2`
- 🎭 **Multi-Genre Awareness** for better diversity
- 🌍 **Language-Aware Filtering**
- 🔢 **Weighted Scoring System** (semantic, genre, rating, popularity)
- 📈 **Popularity Boost with Log-View Normalization**
- 🔍 **Custom Embedding Generation Pipeline**
- 🧠 **Description & Genre Enrichment**
- 🗂️ Organized by Media Type: Movies, Series, and Short Dramas

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Devkolsawala/movie-recommender.git
cd movie-recommender
```

### 2. Create a Python Environment

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Make sure you have the following files in your working directory:

- `final_dataset.csv` — Your full media dataset
- `language_tbl.csv` — Language mapping table

---

## 🧠 Embedding Generation

Run the training script to generate semantic embeddings:

```bash
python enhanced_train_model_mpnet.py
```

This will create `.pt` files with precomputed embeddings for movies, series, and short dramas — separated by language.

---

## 🤖 Recommendation Engine

Run the enhanced recommender system:

```bash
python enhanced_test_improved_mpnet.py
```

You can configure:

- `CLICKED_IDS` — IDs of the movies the user liked
- `MEDIA_TYPE` — `movie`, `series`, or `short_drama`
- `LANG_ID` — Filter by language (optional)
- `ENFORCE_LANGUAGE_MATCHING` — Whether to only recommend same-language content
- `APPLY_GENRE_SPLIT` — Enable genre diversity in top-k results

---

## 📁 Folder Structure

```
movie-recommender/
├── final_dataset.csv
├── language_tbl.csv
├── enhanced_train_model_mpnet.py
├── enhanced_test_improved_mpnet.py
├── *.pt (embedding files)
├── README.md
└── requirements.txt
```

---

## 📊 Output Example

```
🎯 User clicked: Hera Pheri | Genres: Comedy, Drama
🎬 Top 10 Movie Recommendations:
1. Chhichhore | हिन्दी | Score: 0.8563
2. OMG 2       | हिन्दी | Score: 0.8028
...
```

---

## 📌 Model Info

Using `all-mpnet-base-v2` from SentenceTransformers:
- Better semantic accuracy than MiniLM
- Supports contextual understanding of description, title, and genres

---

## ✨ Future Improvements

- Hybrid model (collaborative + content-based)
- Real-time personalization
- Weight optimization with feedback loop
- Web-based front-end integration

---

## 🧑‍💻 Author

Made  by Dev Kolsawala

Feel free to ⭐ the repo if you found it useful!
