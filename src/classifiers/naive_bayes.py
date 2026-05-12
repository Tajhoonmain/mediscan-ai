"""
MediScan AI — Algorithm 1: Naive Bayes Classifier
Multinomial Naive Bayes with Laplace smoothing for health text classification.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class MediScanNaiveBayes:
    """
    Wrapper around sklearn's MultinomialNB.

    Why Multinomial NB for health text?
    - Word-count features are non-negative integers (or TF weights) -> perfect fit
    - Laplace smoothing handles unseen medical jargon gracefully
    - ComplementNB variant works better on imbalanced health topic datasets

    Formula (log-probability):
        log P(c|d) = log P(c) + Σ_i [count(w_i, d) * log P(w_i|c)]
    """

    def __init__(self, alpha: float = 1.0, use_complement: bool = True):
        """
        alpha         : Laplace smoothing parameter (1.0 = standard Laplace)
        use_complement: ComplementNB is often better for imbalanced classes
        """
        self.alpha = alpha
        self.use_complement = use_complement
        nb_cls = ComplementNB if use_complement else MultinomialNB
        self.model = nb_cls(alpha=alpha)
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """Train on TF-IDF sparse matrix."""
        # MultinomialNB requires non-negative features; TF-IDF with sublinear_tf is fine
        # but we clip any possible negatives just in case
        X_train = X_train.copy()
        X_train.data = np.clip(X_train.data, 0, None)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"[NaiveBayes] Trained on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features, {len(np.unique(y_train))} classes")
        return self

    def predict(self, X_test):
        """Return predicted class labels."""
        X_test = X_test.copy()
        X_test.data = np.clip(X_test.data, 0, None)
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Return class probabilities for each sample."""
        X_test = X_test.copy()
        X_test.data = np.clip(X_test.data, 0, None)
        return self.model.predict_proba(X_test)

    def predict_single(self, text: str, vectorizer) -> dict:
        """
        Classify a single raw text string.
        Returns dict with top-3 class probabilities (useful for demo UI).
        """
        X = vectorizer.transform([text])
        X.data = np.clip(X.data, 0, None)
        probs = self.model.predict_proba(X)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        return {int(i): float(probs[i]) for i in top3_idx}
