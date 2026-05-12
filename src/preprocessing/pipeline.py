"""
MediScan AI — Preprocessing Pipeline
Cleans raw health text (tweets, bag-of-words) for IR classification.
"""

import re
import string
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data on first run
for pkg in ["stopwords", "punkt", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Health-domain stopwords to add (very common but content-free in medical text)
HEALTH_STOP = {"health", "say", "said", "new", "via", "http", "https", "rt"}


def clean_tweet(text: str) -> str:
    """Clean a single tweet or short health text string."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # keep hashtag text
    text = re.sub(r"\d+", "", text)                       # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and t not in HEALTH_STOP and len(t) > 2
    ]
    return " ".join(tokens)


def clean_bow_text(text: str) -> str:
    """Light cleaning for Bag-of-Words dataset (already tokenized)."""
    text = str(text).lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def load_health_twitter(filepath: str) -> pd.DataFrame:
    """
    Load the Health News in Twitter dataset.
    Expected format: TSV with columns [id, source, tweet_text]
    Source label is used as the class.
    Download from: https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter
    """
    df = pd.read_csv(filepath, sep="\t", header=None, names=["id", "label", "text"],
                     on_bad_lines="skip")
    df.dropna(subset=["text", "label"], inplace=True)
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.strip() != ""]
    print(f"[Twitter] Loaded {len(df)} samples, {df['label'].nunique()} classes")
    return df


def load_bag_of_words(filepath: str, vocab_path: str = None) -> pd.DataFrame:
    """
    Load the UCI Bag of Words dataset (docword format).
    filepath     : path to docword.*.txt
    vocab_path   : path to vocab.*.txt  (optional but recommended)

    Returns a DataFrame with [doc_id, text_repr, label]
    The label is derived from the source name (nips/kos/enron/etc.)
    """
    # Parse the docword sparse format: docID wordID count
    records = {}
    vocab = {}

    if vocab_path:
        with open(vocab_path) as f:
            vocab = {i + 1: w.strip() for i, w in enumerate(f)}

    with open(filepath) as f:
        n_docs = int(f.readline())
        n_words = int(f.readline())
        _ = f.readline()  # NNZ line
        print(f"[BoW] {n_docs} docs, {n_words} vocab size")
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            doc_id, word_id, count = int(parts[0]), int(parts[1]), int(parts[2])
            word = vocab.get(word_id, f"w{word_id}")
            records.setdefault(doc_id, []).extend([word] * count)

    rows = [{"doc_id": did, "text": " ".join(words)} for did, words in records.items()]
    df = pd.DataFrame(rows)
    df["clean_text"] = df["text"].apply(clean_bow_text)

    # Derive label from filename (e.g. docword.kos.txt -> label = "kos")
    import os
    source = os.path.basename(filepath).replace("docword.", "").replace(".txt", "")
    df["label"] = source
    print(f"[BoW] Loaded {len(df)} documents")
    return df


def build_features(df: pd.DataFrame,
                   text_col: str = "clean_text",
                   max_features: int = 10000,
                   vectorizer=None):
    """
    TF-IDF vectorization.
    Returns: (X_sparse, vectorizer)
    Pass a fitted vectorizer to transform without re-fitting (for test sets).
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,          # log(1+tf) — helps with bursty medical terms
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),         # unigrams + bigrams
        )
        X = vectorizer.fit_transform(df[text_col])
    else:
        X = vectorizer.transform(df[text_col])
    return X, vectorizer


def split_data(df: pd.DataFrame, X, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, le
