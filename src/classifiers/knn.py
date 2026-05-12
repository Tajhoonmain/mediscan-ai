"""
MediScan AI — Algorithm 3: k-Nearest Neighbors Classification
Cosine-similarity based kNN for health text using TF-IDF vectors.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


class MediScankNN:
    """
    k-Nearest Neighbors Classifier using cosine distance.

    For text classification, cosine similarity is much better than Euclidean
    distance because it normalises document length — a long tweet about heart
    disease and a short one are treated the same way directionally.

    Decision rule (majority vote):
        y_hat = argmax_c Σ_{i in kNN(x)} I(y_i == c)

    Weighted variant (optional):
        Weight each neighbor's vote by its cosine similarity score — closer
        neighbors have more influence. Better for ambiguous health topics.
    """

    def __init__(self, k: int = 5, weights: str = "distance"):
        """
        k       : number of nearest neighbors (try 3, 5, 7 — report best)
        weights : 'uniform' (equal vote) or 'distance' (weighted by cosine sim)
        """
        self.k = k
        self.weights = weights
        # sklearn uses metric='cosine' which is 1 - cosine_similarity
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            weights=weights,
            algorithm="brute",      # brute-force required for cosine on sparse matrices
            n_jobs=-1,
        )
        self.X_train_store = None   # stored for neighbour inspection in demo
        self.y_train_store = None
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """Fit the kNN model (stores training data for lookup)."""
        self.model.fit(X_train, y_train)
        self.X_train_store = X_train
        self.y_train_store = np.array(y_train)
        self.is_fitted = True
        print(f"[kNN] Fitted with k={self.k}, weights='{self.weights}', "
              f"{X_train.shape[0]} training samples")
        return self

    def predict(self, X_test):
        """Return predicted class labels."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Return class probability estimates."""
        return self.model.predict_proba(X_test)

    def get_neighbors(self, X_query, vectorizer=None, label_encoder=None) -> list[dict]:
        """
        Return the k nearest neighbors for a query — used in the demo UI
        to show *why* a document was classified a certain way.

        Returns a list of dicts:
            [{"index": int, "similarity": float, "label": str}, ...]
        """
        distances, indices = self.model.kneighbors(X_query, n_neighbors=self.k)
        neighbors = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1.0 - dist   # convert cosine distance back to similarity
            label = self.y_train_store[idx]
            if label_encoder:
                label = label_encoder.inverse_transform([label])[0]
            neighbors.append({
                "train_index": int(idx),
                "cosine_similarity": round(float(similarity), 4),
                "label": label,
            })
        return neighbors

    def predict_single(self, text: str, vectorizer, label_encoder=None) -> dict:
        """
        Classify a single raw text string.
        Returns dict with prediction, confidence, and top-k neighbors.
        """
        X = vectorizer.transform([text])
        pred_label = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        neighbors = self.get_neighbors(X, vectorizer, label_encoder)

        if label_encoder:
            pred_label = label_encoder.inverse_transform([pred_label])[0]

        return {
            "prediction": pred_label,
            "confidence": round(float(max(proba)), 4),
            "neighbors": neighbors,
        }

    def tune_k(self, X_train, y_train, X_val, y_val, k_range=range(1, 16, 2)):
        """
        Try multiple k values on a validation set.
        Returns the best k and a dict of {k: accuracy} for plotting.
        """
        from sklearn.metrics import accuracy_score
        results = {}
        best_k, best_acc = self.k, 0.0
        for k in k_range:
            temp = KNeighborsClassifier(n_neighbors=k, metric="cosine",
                                        algorithm="brute", weights=self.weights)
            temp.fit(X_train, y_train)
            acc = accuracy_score(y_val, temp.predict(X_val))
            results[k] = round(acc, 4)
            if acc > best_acc:
                best_acc, best_k = acc, k
        print(f"[kNN] Best k={best_k} with accuracy={best_acc:.4f}")
        return best_k, results
