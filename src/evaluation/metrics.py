"""
MediScan AI — Evaluation Module
Computes Accuracy, Precision, Recall, F1-score and plots confusion matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from pathlib import Path


def compute_metrics(y_true, y_pred, average: str = "weighted") -> dict:
    """
    Compute the four required IR metrics.

    average : 'weighted' (accounts for class imbalance — recommended for report)
              'macro'    (treats all classes equally)
              'micro'    (global counts)
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
    }


def full_report(y_true, y_pred, label_encoder=None, model_name: str = "") -> str:
    """Print sklearn's full per-class classification report."""
    target_names = None
    if label_encoder is not None:
        target_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    header = f"\n{'='*50}\n  {model_name} — Classification Report\n{'='*50}\n"
    print(header + report)
    return report


def compare_models(results: dict, dataset_name: str = "") -> pd.DataFrame:
    """
    results: { "Naive Bayes": metrics_dict, "Rocchio": ..., "kNN": ... }
    Returns a nicely formatted DataFrame for the report table.
    """
    rows = []
    for model, metrics in results.items():
        rows.append({
            "Model": model,
            "Accuracy":  f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall":    f"{metrics['recall']:.4f}",
            "F1-Score":  f"{metrics['f1_score']:.4f}",
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*50}\n  Results — {dataset_name}\n{'='*50}")
    print(df.to_string(index=False))
    return df


def plot_confusion_matrix(y_true, y_pred, label_encoder=None,
                          model_name: str = "", save_path: str = None):
    """Plot and optionally save a confusion matrix heatmap."""
    labels = label_encoder.classes_ if label_encoder else None
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(6, len(np.unique(y_true))), 
                                    max(5, len(np.unique(y_true)) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Saved confusion matrix -> {save_path}")
    plt.show()
    return fig


def plot_metric_comparison(results: dict, dataset_name: str = "", save_path: str = None):
    """
    Bar chart comparing Accuracy / Precision / Recall / F1 across all three models.
    Perfect for pasting into the project report.
    """
    models = list(results.keys())
    metrics_list = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(models))
    width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Algorithm Comparison — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Saved comparison chart -> {save_path}")
    plt.show()
    return fig


def plot_knn_k_tuning(k_results: dict, save_path: str = None):
    """Line plot of kNN accuracy vs k values."""
    ks = list(k_results.keys())
    accs = list(k_results.values())
    best_k = ks[np.argmax(accs)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, accs, marker="o", color="#4C72B0", linewidth=2)
    ax.axvline(best_k, color="#C44E52", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    ax.set_xlabel("k (Number of Neighbors)", fontsize=11)
    ax.set_ylabel("Validation Accuracy", fontsize=11)
    ax.set_title("kNN Accuracy vs. k", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
