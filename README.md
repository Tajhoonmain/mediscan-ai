# MediScan AI — Clinical Text Triage Engine

> CS444 Information Retrieval · Semester Project  
> Course Instructor: Dr. Zoya  

A health-text classification system that applies **Naive Bayes**, **Rocchio**, and **kNN** to real-world health datasets, compares their performance, and provides a live interactive demo.

---

## Project Structure

```
mediscan_ai/
├── data/
│   ├── raw/
│   │   ├── health_twitter/Health-Tweets/   ← put downloaded .txt files here
│   │   └── bag_of_words/                  ← put docword.*.txt and vocab.*.txt here
│   └── processed/                         ← auto-generated cleaned CSVs
│
├── src/
│   ├── preprocessing/
│   │   └── pipeline.py      ← cleaning, TF-IDF, train/test split
│   ├── classifiers/
│   │   ├── naive_bayes.py   ← Algorithm 1
│   │   ├── rocchio.py       ← Algorithm 2
│   │   └── knn.py           ← Algorithm 3
│   └── evaluation/
│       └── metrics.py       ← Accuracy, Precision, Recall, F1 + plots
│
├── app/
│   └── demo.py              ← Streamlit live demo UI
│
├── notebooks/
│   └── exploration.ipynb    ← EDA and per-dataset analysis (optional)
│
├── models/                  ← saved .pkl model files (auto-generated)
├── reports/
│   └── figures/             ← auto-generated confusion matrices and charts
│
├── tests/                   ← unit tests
├── main.py                  ← single entry point to train + evaluate everything
├── requirements.txt
└── README.md
```

---

## Algorithms

| # | Algorithm | Key Idea | Best For |
|---|-----------|----------|----------|
| 1 | **Naive Bayes** | P(class\|doc) via word-frequency priors | Fast baseline, sparse text |
| 2 | **Rocchio** | Centroid vectors per class, cosine similarity | Topic prototype matching |
| 3 | **kNN** | k nearest training documents by cosine distance | Fine-grained similarity |

---

## Datasets

| Dataset | Source | Classes | Size |
|---------|--------|---------|------|
| Health News in Twitter | UCI ML Repository | 16 health news sources | ~16,000 tweets |
| Bag of Words (KOS Blog) | UCI ML Repository | Document topics | ~3,430 documents |

Download links:
- https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter
- https://archive.ics.uci.edu/ml/datasets/Bag+of+Words

---

## Setup & Installation

### 1. Clone / unzip the project

```bash
cd mediscan_ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (auto-runs on first use, or run manually)

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
```

---

## Download the Datasets

### Health News in Twitter

1. Go to https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter
2. Download the ZIP file
3. Extract so the `.txt` files are at:
   ```
   data/raw/health_twitter/Health-Tweets/bbchealth.txt
   data/raw/health_twitter/Health-Tweets/cnnhealth.txt
   ... (all 16 source files)
   ```

### Bag of Words

1. Go to https://archive.ics.uci.edu/ml/datasets/Bag+of+Words
2. Download `docword.kos.txt.gz` and `vocab.kos.txt.gz`
3. Extract to:
   ```
   data/raw/bag_of_words/docword.kos.txt
   data/raw/bag_of_words/vocab.kos.txt
   ```

---

## Running the Project

### Train and evaluate all algorithms

```bash
python main.py
```

This will:
- Load and clean both datasets
- Build TF-IDF features
- Train Naive Bayes, Rocchio, and kNN
- Print Accuracy / Precision / Recall / F1 for each
- Save confusion matrix plots to `reports/figures/`
- Save trained models to `models/`

### Launch the live demo

```bash
streamlit run app/demo.py
```

Open http://localhost:8501 in your browser. Type any health sentence and see all three classifiers vote in real time.

---

## Evaluation Metrics

All results are reported with **weighted averages** to account for class imbalance:

| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) |

---

## Preprocessing Pipeline

1. Lowercase all text
2. Remove URLs, @mentions, digits, punctuation
3. Keep hashtag text (e.g. `#diabetes` → `diabetes`)
4. Remove English stopwords + domain stopwords (`health`, `new`, etc.)
5. WordNet lemmatization
6. TF-IDF vectorization (max 10,000 features, unigrams + bigrams, sublinear TF)
7. Stratified 80/20 train/test split

---

## Group Members

| Name | Student ID | Role |
|------|-----------|------|
| [Member 1] | [ID] | Preprocessing + Naive Bayes |
| [Member 2] | [ID] | Rocchio + Evaluation |
| [Member 3 — Group Leader] | [ID] | kNN + Demo UI + Report |

---

## References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Health News in Twitter Dataset: UCI Machine Learning Repository
4. Bag of Words Dataset: UCI Machine Learning Repository
