"""
model.py — Train, evaluate, and persist the Sentiment Analyzer.

Pipeline: TF-IDF vectorizer → Logistic Regression
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend for script usage
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
)

from data import get_splits, load_dataset

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
POS_COL = "#4CAF82"   # green
NEG_COL = "#E05C5C"   # red
BG_COL  = "#F8F8F8"


# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5_000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1_000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
def train(save: bool = True) -> tuple:
    """Fit the pipeline and return (pipeline, metrics_dict)."""
    X_train, X_test, y_train, y_test = get_splits()

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=["Negative", "Positive"],
                                    output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    metrics = dict(
        accuracy=acc, report=report, cm=cm,
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        X_test=X_test, y_test=y_test, y_pred=y_pred,
    )

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipe, f)
        print(f"Model saved → {MODEL_PATH}")

    print(f"\nTest accuracy : {acc:.4f}  |  ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Negative", "Positive"]))
    return pipe, metrics


# ─────────────────────────────────────────────────────────────────────────────
def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train() first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
def predict(texts: list[str], model: Pipeline | None = None) -> pd.DataFrame:
    """Predict sentiment for a list of texts."""
    if model is None:
        model = load_model()
    labels  = model.predict(texts)
    probas  = model.predict_proba(texts)
    results = []
    for txt, lbl, prob in zip(texts, labels, probas):
        results.append({
            "text"        : txt,
            "sentiment"   : "Positive" if lbl == 1 else "Negative",
            "confidence"  : f"{max(prob):.2%}",
            "pos_score"   : round(prob[1], 4),
            "neg_score"   : round(prob[0], 4),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "figure.facecolor" : BG_COL,
        "axes.facecolor"   : BG_COL,
        "font.family"      : "DejaVu Sans",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    })


def plot_confusion_matrix(cm: np.ndarray, save: bool = True):
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=1, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    if save:
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    return fig


def plot_roc_curve(fpr, tpr, roc_auc: float, save: bool = True):
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=POS_COL, lw=2.5,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curve.png")
    if save:
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    return fig


def plot_top_features(model: Pipeline, n: int = 20, save: bool = True):
    """Bar chart of most influential words for each class."""
    _style()
    vect    = model.named_steps["tfidf"]
    clf     = model.named_steps["clf"]
    feature_names = np.array(vect.get_feature_names_out())
    coef    = clf.coef_[0]

    top_pos_idx = np.argsort(coef)[-n:][::-1]
    top_neg_idx = np.argsort(coef)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, idx, color, title in [
        (axes[0], top_pos_idx, POS_COL, "Top Positive Words"),
        (axes[1], top_neg_idx, NEG_COL, "Top Negative Words"),
    ]:
        words  = feature_names[idx]
        values = np.abs(coef[idx])
        bars = ax.barh(words[::-1], values[::-1], color=color, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Coefficient magnitude", fontsize=10)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)

    fig.suptitle("Most Influential Features (TF-IDF + LR)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "top_features.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


def plot_score_distribution(model: Pipeline, save: bool = True):
    """Histogram of positive-class probability scores on the full dataset."""
    _style()
    df = load_dataset()
    probas = model.predict_proba(df["text"])[:, 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, col, name in [(1, POS_COL, "Positive"), (0, NEG_COL, "Negative")]:
        mask = df["label"] == lbl
        ax.hist(probas[mask], bins=20, color=col, alpha=0.7, label=name,
                edgecolor="white", linewidth=0.5)

    ax.axvline(0.5, color="#555", linestyle="--", linewidth=1.2,
               label="Decision boundary")
    ax.set_xlabel("Predicted positive-class probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Score Distribution by True Label", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "score_distribution.png")
    if save:
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, metrics = train(save=True)
    plot_confusion_matrix(metrics["cm"])
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"])
    plot_top_features(model)
    plot_score_distribution(model)

    print("\n── Quick demo predictions ──")
    samples = [
        "I absolutely loved this, it was fantastic!",
        "Terrible quality, broke after one day.",
        "It was okay, nothing special.",
        "Best experience of my life, highly recommend!",
        "Very disappointing and a waste of money.",
    ]
    print(predict(samples, model).to_string(index=False))
