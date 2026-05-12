"""
MediScan AI — Unit Tests
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer

from src.classifiers.naive_bayes import MediScanNaiveBayes
from src.classifiers.rocchio import MediScanRocchio
from src.classifiers.knn import MediScankNN
from src.evaluation.metrics import compute_metrics


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "heart disease symptoms treatment",
    "flu vaccine cdc recommendation",
    "mental health anxiety depression therapy",
    "cancer tumor chemotherapy survival",
    "diabetes insulin blood sugar control",
    "covid pandemic vaccine immunity",
]
SAMPLE_LABELS = [0, 1, 2, 0, 1, 2]


@pytest.fixture
def tfidf_data():
    vec = TfidfVectorizer()
    X = vec.fit_transform(SAMPLE_TEXTS)
    y = np.array(SAMPLE_LABELS)
    return X, y, vec


# ── Naive Bayes tests ──────────────────────────────────────────────────────────

def test_naive_bayes_fit_predict(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScanNaiveBayes(alpha=1.0, use_complement=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1, 2})


def test_naive_bayes_proba_sums_to_one(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScanNaiveBayes().fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_naive_bayes_single(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScanNaiveBayes().fit(X, y)
    result = model.predict_single("cancer treatment", vec)
    assert isinstance(result, dict)
    assert len(result) == 3


# ── Rocchio tests ──────────────────────────────────────────────────────────────

def test_rocchio_fit_predict(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScanRocchio(alpha=1.0, beta=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_rocchio_centroids_shape(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScanRocchio().fit(X, y)
    assert model.centroids_.shape == (3, X.shape[1])


def test_rocchio_centroid_normalised(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScanRocchio().fit(X, y)
    norms = np.linalg.norm(model.centroids_, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_rocchio_beta_effect(tfidf_data):
    X, y, vec = tfidf_data
    m0 = MediScanRocchio(beta=0.0).fit(X, y)
    m1 = MediScanRocchio(beta=0.5).fit(X, y)
    # centroids should differ when beta != 0
    assert not np.allclose(m0.centroids_, m1.centroids_)


# ── kNN tests ──────────────────────────────────────────────────────────────────

def test_knn_fit_predict(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScankNN(k=3).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_knn_get_neighbors_count(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScankNN(k=3).fit(X, y)
    neighbors = model.get_neighbors(X[:1])
    assert len(neighbors) == 3


def test_knn_single_returns_dict(tfidf_data):
    X, y, vec = tfidf_data
    model = MediScankNN(k=3).fit(X, y)
    result = model.predict_single("flu symptoms fever", vec)
    assert "prediction" in result
    assert "confidence" in result
    assert "neighbors" in result


# ── Evaluation tests ───────────────────────────────────────────────────────────

def test_compute_metrics_perfect():
    y = [0, 1, 2, 0, 1]
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_compute_metrics_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert all(k in metrics for k in ["accuracy", "precision", "recall", "f1_score"])


def test_metrics_range():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 2]
    metrics = compute_metrics(y_true, y_pred)
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
