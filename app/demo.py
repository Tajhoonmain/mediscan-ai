"""
MediScan AI — Live Demo
Run: streamlit run app/demo.py
"""

import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to sys.path so joblib can unpickle 'src' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="MediScan AI", page_icon="🩺", layout="wide")

MODEL_DIR = Path("models")

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models(dataset_prefix: str):
    """Load all three trained models for the selected dataset."""
    try:
        nb_data  = joblib.load(MODEL_DIR / f"{dataset_prefix}_nb.pkl")
        roc_data = joblib.load(MODEL_DIR / f"{dataset_prefix}_rocchio.pkl")
        knn_data = joblib.load(MODEL_DIR / f"{dataset_prefix}_knn.pkl")
        return nb_data, roc_data, knn_data
    except FileNotFoundError:
        return None, None, None


def run_inference(text: str, nb_data, roc_data, knn_data):
    """Run all three classifiers on a single text input."""
    from src.preprocessing.pipeline import clean_tweet
    cleaned = clean_tweet(text)

    vectorizer = nb_data["vectorizer"]
    le         = nb_data["le"]
    X = vectorizer.transform([cleaned])

    # Naive Bayes
    nb_model = nb_data["model"]
    nb_proba  = nb_model.predict_proba(X)[0]
    nb_pred   = le.inverse_transform([np.argmax(nb_proba)])[0]
    nb_conf   = float(max(nb_proba))
    nb_top3   = sorted(
        [(le.inverse_transform([i])[0], float(p)) for i, p in enumerate(nb_proba)],
        key=lambda x: -x[1]
    )[:3]

    # Rocchio
    roc_model  = roc_data["model"]
    roc_scores = roc_model.decision_scores(X)[0]
    roc_pred   = le.inverse_transform(roc_model.predict(X))[0]
    roc_conf   = float(max(roc_scores))
    roc_top3   = sorted(
        [(le.inverse_transform([c])[0], float(s))
         for c, s in zip(roc_model.classes_, roc_scores)],
        key=lambda x: -x[1]
    )[:3]

    # kNN
    knn_model   = knn_data["model"]
    knn_pred    = le.inverse_transform(knn_model.predict(X))[0]
    knn_result  = knn_model.predict_single(text, vectorizer, le)

    return {
        "cleaned": cleaned,
        "naive_bayes": {"prediction": nb_pred, "confidence": nb_conf, "top3": nb_top3},
        "rocchio":     {"prediction": roc_pred, "similarity": roc_conf, "top3": roc_top3},
        "knn":         knn_result,
    }


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🩺 MediScan AI — Clinical Text Triage Demo")
st.caption("CS444 Semester Project · Naive Bayes · Rocchio · kNN")

# Sidebar
with st.sidebar:
    st.header("Settings")
    dataset_choice = st.selectbox(
        "Dataset / Model",
        ["Health_Twitter", "Bag_of_Words"],
        help="Switch between models trained on different datasets"
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This demo classifies health text using three IR algorithms trained on real datasets.")
    st.markdown("Each classifier votes independently. Compare their confidence scores below.")

# Load models
nb_data, roc_data, knn_data = load_models(dataset_choice)

if nb_data is None:
    st.warning(
        f"No trained models found for **{dataset_choice}**. "
        "Run `python main.py` first to train the models."
    )
    st.stop()

# Input
st.subheader("Enter health text to classify")
example_texts = [
    "New study links sleep deprivation to increased risk of heart disease",
    "CDC warns about rising flu cases across the midwest this winter",
    "Mental health apps help teenagers manage anxiety and depression symptoms",
    "Breakthrough cancer treatment shows 80% success rate in clinical trials",
]
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Type or paste a health tweet / headline:", height=100,
                               placeholder="e.g. New vaccine shows promising results against drug-resistant TB...")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Try an example:**")
    for ex in example_texts:
        if st.button(ex[:50] + "...", key=ex):
            user_input = ex

run_btn = st.button("🔍 Classify", type="primary", disabled=not bool(user_input))

if run_btn and user_input:
    with st.spinner("Running classifiers..."):
        results = run_inference(user_input, nb_data, roc_data, knn_data)

    st.markdown("---")
    st.subheader("Results")
    st.caption(f"Cleaned input: `{results['cleaned']}`")

    # Three columns — one per algorithm
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Naive Bayes")
        nb = results["naive_bayes"]
        st.metric("Prediction", nb["prediction"])
        st.metric("Confidence", f"{nb['confidence']:.1%}")
        st.markdown("**Top 3 classes:**")
        for label, prob in nb["top3"]:
            st.progress(prob, text=f"{label}: {prob:.3f}")

    with c2:
        st.markdown("#### Rocchio")
        roc = results["rocchio"]
        st.metric("Prediction", roc["prediction"])
        st.metric("Centroid similarity", f"{roc['similarity']:.4f}")
        st.markdown("**Top 3 classes:**")
        max_sim = max(s for _, s in roc["top3"])
        for label, sim in roc["top3"]:
            st.progress(sim / max(max_sim, 1e-6), text=f"{label}: {sim:.4f}")

    with c3:
        st.markdown(f"#### kNN (k neighbors)")
        knn = results["knn"]
        st.metric("Prediction", knn["prediction"])
        st.metric("Confidence", f"{knn['confidence']:.1%}")
        st.markdown("**Nearest neighbors:**")
        for nb_item in knn.get("neighbors", []):
            st.write(
                f"- **{nb_item['label']}** (sim: {nb_item['cosine_similarity']:.4f})"
            )

    # Verdict
    st.markdown("---")
    predictions = [
        results["naive_bayes"]["prediction"],
        results["rocchio"]["prediction"],
        results["knn"]["prediction"],
    ]
    from collections import Counter
    vote_counts = Counter(predictions)
    majority, votes = vote_counts.most_common(1)[0]
    agreement = "✅ All classifiers agree" if votes == 3 else f"⚠️ Majority vote ({votes}/3)"
    st.info(f"**Final verdict: {majority}** — {agreement}")
