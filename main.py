"""
MediScan AI — Main Runner
Trains and evaluates all 3 classifiers on both datasets.
Run: python main.py
"""

import os
import joblib
import numpy as np
from pathlib import Path

from src.preprocessing.pipeline import (
    load_health_twitter, load_bag_of_words,
    build_features, split_data,
)
from src.classifiers.naive_bayes import MediScanNaiveBayes
from src.classifiers.rocchio import MediScanRocchio
from src.classifiers.knn import MediScankNN
from src.evaluation.metrics import (
    compute_metrics, compare_models, full_report,
    plot_confusion_matrix, plot_metric_comparison, plot_knn_k_tuning,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw")
OUTPUT_DIR = Path("reports/figures")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASETS = {
    "Health Twitter": {
        "loader": "twitter",
        "path": DATA_DIR / "health_twitter" / "Health-Tweets",   # folder with .txt files
    },
    "Bag of Words (KOS)": {
        "loader": "bow",
        "path": DATA_DIR / "bag_of_words" / "docword.kos.txt",
        "vocab": DATA_DIR / "bag_of_words" / "vocab.kos.txt",
    },
}


def load_twitter_all(folder: Path):
    """Concatenate all source .txt files in the Health-Tweets folder."""
    import pandas as pd
    frames = []
    for f in sorted(folder.glob("*.txt")):
        source = f.stem  # e.g. 'bbchealth', 'cnnhealth' ...
        df_src = pd.read_csv(f, sep="|", header=None,
                              names=["id", "date", "text"], on_bad_lines="skip", encoding="latin-1")
        df_src["label"] = source
        frames.append(df_src)
    df = pd.concat(frames, ignore_index=True)
    df.dropna(subset=["text"], inplace=True)

    from src.preprocessing.pipeline import clean_tweet
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.strip() != ""]
    print(f"[Twitter] Total: {len(df)} tweets, {df['label'].nunique()} sources")
    return df


def run_dataset(name: str, df, max_features: int = 10000):
    """Train & evaluate all 3 classifiers on one dataset. Returns results dict."""
    print(f"\n{'#'*60}")
    print(f"  Dataset: {name}")
    print(f"{'#'*60}")

    # Build TF-IDF features & split
    X, vectorizer = build_features(df, max_features=max_features)
    X_train, X_test, y_train, y_test, le = split_data(df, X)

    # ── Train ──────────────────────────────────────────────────────────────────
    nb  = MediScanNaiveBayes(alpha=1.0, use_complement=True).fit(X_train, y_train)
    roc = MediScanRocchio(alpha=1.0, beta=0.25).fit(X_train, y_train)

    # Tune kNN k on a small validation split
    val_size = min(0.15, 3000 / X_train.shape[0])
    from sklearn.model_selection import train_test_split
    Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=val_size,
                                       stratify=y_train, random_state=0)
    knn_temp = MediScankNN(k=5).fit(Xt, yt)
    best_k, k_results = knn_temp.tune_k(Xt, yt, Xv, yv, k_range=range(1, 12, 2))
    knn = MediScankNN(k=best_k).fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred_nb  = nb.predict(X_test)
    y_pred_roc = roc.predict(X_test)
    y_pred_knn = knn.predict(X_test)

    results = {
        "Naive Bayes": compute_metrics(y_test, y_pred_nb),
        "Rocchio":     compute_metrics(y_test, y_pred_roc),
        f"kNN (k={best_k})": compute_metrics(y_test, y_pred_knn),
    }

    compare_models(results, dataset_name=name)
    full_report(y_test, y_pred_nb,  le, "Naive Bayes")
    full_report(y_test, y_pred_roc, le, "Rocchio")
    full_report(y_test, y_pred_knn, le, f"kNN (k={best_k})")

    # ── Plots ──────────────────────────────────────────────────────────────────
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison(results, name,
                           save_path=str(OUTPUT_DIR / f"{safe_name}_comparison.png"))
    plot_confusion_matrix(y_test, y_pred_nb,  le, "Naive Bayes",
                          save_path=str(OUTPUT_DIR / f"{safe_name}_cm_nb.png"))
    plot_confusion_matrix(y_test, y_pred_roc, le, "Rocchio",
                          save_path=str(OUTPUT_DIR / f"{safe_name}_cm_rocchio.png"))
    plot_confusion_matrix(y_test, y_pred_knn, le, f"kNN k={best_k}",
                          save_path=str(OUTPUT_DIR / f"{safe_name}_cm_knn.png"))
    plot_knn_k_tuning(k_results,
                      save_path=str(OUTPUT_DIR / f"{safe_name}_knn_k_tune.png"))

    # ── Save models ────────────────────────────────────────────────────────────
    joblib.dump({"model": nb,  "vectorizer": vectorizer, "le": le},
                MODEL_DIR / f"{safe_name}_nb.pkl")
    joblib.dump({"model": roc, "vectorizer": vectorizer, "le": le},
                MODEL_DIR / f"{safe_name}_rocchio.pkl")
    joblib.dump({"model": knn, "vectorizer": vectorizer, "le": le},
                MODEL_DIR / f"{safe_name}_knn.pkl")
    print(f"[Main] Models saved to {MODEL_DIR}/")

    return results


def main():
    all_results = {}

    # ── Dataset 1: Health Twitter ──────────────────────────────────────────────
    twitter_path = DATASETS["Health Twitter"]["path"]
    if twitter_path.exists():
        df_twitter = load_twitter_all(twitter_path)
        all_results["Health Twitter"] = run_dataset("Health Twitter", df_twitter)
    else:
        print(f"[SKIP] Twitter data not found at {twitter_path}. "
              "Download from https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter")

    # ── Dataset 2: Bag of Words ────────────────────────────────────────────────
    bow_path   = DATASETS["Bag of Words (KOS)"]["path"]
    vocab_path = DATASETS["Bag of Words (KOS)"]["vocab"]
    if bow_path.exists():
        df_bow = load_bag_of_words(str(bow_path), str(vocab_path))
        # BoW single-source: create synthetic binary labels (top-half vs bottom-half word count)
        # Replace with multi-source merge if you download multiple BoW datasets
        df_bow["label"] = (df_bow["clean_text"].str.split().str.len() > 50).map(
            {True: "long_doc", False: "short_doc"}
        )
        all_results["Bag of Words"] = run_dataset("Bag of Words", df_bow)
    else:
        print(f"[SKIP] BoW data not found at {bow_path}. "
              "Download from https://archive.ics.uci.edu/ml/datasets/Bag+of+Words")

    print("\n[Done] All experiments complete.")
    return all_results


if __name__ == "__main__":
    main()
