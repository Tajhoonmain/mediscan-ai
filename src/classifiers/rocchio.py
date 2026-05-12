"""
MediScan AI — Algorithm 2: Rocchio Classification
Centroid-based classifier using TF-IDF vectors and cosine similarity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


class MediScanRocchio:
    """
    Rocchio Classifier — builds a prototype (centroid) vector per class,
    then assigns a new document to the class whose centroid it is closest to
    (by cosine similarity).

    Rocchio update formula:
        centroid_c = (alpha / |D_c|) * Σ_{d in D_c} tf-idf(d)
                   - (beta  / |D_¬c|) * Σ_{d not in D_c} tf-idf(d)

    For pure classification (no negative feedback), beta = 0:
        centroid_c = mean( tf-idf(d) for d in class c )

    This is the standard Rocchio classification variant.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        alpha : weight for positive class documents  (default 1.0)
        beta  : weight for negative class documents  (default 0.0 = no negative feedback)
                Set beta > 0 (e.g. 0.25) to push centroids away from other-class docs.
        """
        self.alpha = alpha
        self.beta = beta
        self.centroids_ = None      # shape: (n_classes, n_features)
        self.classes_ = None
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Build one centroid per class.
        X_train : sparse TF-IDF matrix (n_samples, n_features)
        y_train : integer class labels
        """
        self.classes_ = np.unique(y_train)
        n_features = X_train.shape[1]
        n_classes = len(self.classes_)
        centroids = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            mask_pos = (y_train == c)
            mask_neg = ~mask_pos

            # Positive centroid: mean of all documents in class c
            pos_docs = X_train[mask_pos]
            pos_centroid = np.asarray(pos_docs.mean(axis=0)).flatten()

            # Negative centroid: mean of all documents NOT in class c
            neg_centroid = np.zeros(n_features)
            if self.beta > 0 and mask_neg.sum() > 0:
                neg_docs = X_train[mask_neg]
                neg_centroid = np.asarray(neg_docs.mean(axis=0)).flatten()

            centroids[i] = self.alpha * pos_centroid - self.beta * neg_centroid

        # L2-normalise centroids so cosine_similarity is just a dot product
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.centroids_ = centroids / norms
        self.is_fitted = True
        print(f"[Rocchio] Built {n_classes} centroids, {n_features} dimensions "
              f"(alpha={self.alpha}, beta={self.beta})")
        return self

    def predict(self, X_test):
        """Assign each doc to the nearest centroid by cosine similarity."""
        sims = cosine_similarity(X_test, self.centroids_)   # (n_test, n_classes)
        nearest = np.argmax(sims, axis=1)
        return self.classes_[nearest]

    def decision_scores(self, X_test) -> np.ndarray:
        """Return raw cosine similarity scores for all classes (for demo UI)."""
        return cosine_similarity(X_test, self.centroids_)

    def predict_single(self, text: str, vectorizer, label_encoder=None) -> dict:
        """
        Classify a single raw text string.
        Returns dict: {class_label: cosine_similarity_score} for all classes.
        """
        X = vectorizer.transform([text])
        scores = self.decision_scores(X)[0]
        result = {}
        for i, c in enumerate(self.classes_):
            key = label_encoder.inverse_transform([c])[0] if label_encoder else int(c)
            result[key] = float(scores[i])
        return dict(sorted(result.items(), key=lambda x: -x[1]))
