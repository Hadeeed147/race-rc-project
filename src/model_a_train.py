"""
Model A -- Answer Verification (Supervised + Ensemble)

Trains Logistic Regression, SVM, and Naive Bayes on TF-IDF features
for the binary task: is this option the correct answer? (1/0)

Also builds a soft-vote ensemble of all three models.

Metrics reported:
  - Accuracy (per-option)
  - Macro F1 (per-option)
  - Exact Match (EM) -- fraction of questions where the highest-scoring
    option matches the gold label

Usage:
    python src/model_a_train.py          # train all models
    python src/model_a_train.py --force  # retrain even if pkl files exist
"""

import os
import sys
import time
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── paths ───────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "models") # where X_train.pkl is currently saved
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models", "model_a", "traditional")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

OPTIONS = ["A", "B", "C", "D"]


# ── helpers ─────────────────────────────────────────────────────────
def load_data(split="train"):
    """Load pre-built TF-IDF matrix, labels, and question IDs."""
    X = joblib.load(os.path.join(DATA_DIR, f"X_{split}.pkl"))
    y = joblib.load(os.path.join(DATA_DIR, f"y_{split}.pkl"))
    qids = joblib.load(os.path.join(DATA_DIR, f"qids_{split}.pkl"))
    return X, y, qids


def exact_match(y_true, y_prob, qids):
    """
    Exact Match: for each question (group of 4 options), check if the
    option with highest predicted probability matches the gold label.

    y_prob : array of shape (n_samples,) -- probability of class 1
    qids   : list of question identifiers (same qid for 4 consecutive rows)
    """
    correct = 0
    total = 0

    # Group by question (every 4 consecutive rows)
    for i in range(0, len(y_true), 4):
        group_true = y_true[i : i + 4]
        group_prob = y_prob[i : i + 4]

        gold_idx = np.argmax(group_true)    # which option is correct
        pred_idx = np.argmax(group_prob)    # which option model thinks is correct

        if gold_idx == pred_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_model(name, model, X_val, y_val, qids_val):
    """Evaluate a model and print metrics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Binary predictions
    y_pred = model.predict(X_val)

    # Probabilities for Exact Match
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_val)
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    em = exact_match(y_val, y_prob, qids_val)
    cm = confusion_matrix(y_val, y_pred)

    print(f"  Option Accuracy : {acc:.4f}")
    print(f"  Macro F1        : {f1_macro:.4f}")
    print(f"  Exact Match (EM): {em:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"    FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")
    print()
    print(classification_report(y_val, y_pred, target_names=["wrong", "correct"]))

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "exact_match": round(em, 4),
        "confusion_matrix": cm.tolist(),
    }


# ── model training ──────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train, force=False):
    path = os.path.join(MODEL_SAVE_DIR, "lr_classifier.pkl")
    if not force and os.path.exists(path):
        print("[LR] Loading existing model ...")
        return joblib.load(path)

    print("[LR] Training Logistic Regression ...")
    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",   # Switched to liblinear for much faster training
        C=1.0,
        class_weight="balanced", # handle 75/25 class imbalance
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"[LR] Done in {time.time()-t0:.1f}s")
    joblib.dump(model, path)
    return model


def train_svm(X_train, y_train, force=False):
    path = os.path.join(MODEL_SAVE_DIR, "svm_classifier.pkl")
    if not force and os.path.exists(path):
        print("[SVM] Loading existing model ...")
        return joblib.load(path)

    print("[SVM] Training LinearSVC ...")
    t0 = time.time()
    # LinearSVC is much faster than SVC for large datasets
    # We use class_weight="balanced" and rely on decision_function for ensemble
    model = LinearSVC(
        max_iter=2000,
        C=1.0,
        class_weight="balanced", # handle 75/25 class imbalance
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"[SVM] Done in {time.time()-t0:.1f}s")
    joblib.dump(model, path)
    return model


def train_naive_bayes(X_train, y_train, force=False):
    path = os.path.join(MODEL_SAVE_DIR, "nb_classifier.pkl")
    if not force and os.path.exists(path):
        print("[NB] Loading existing model ...")
        return joblib.load(path)

    print("[NB] Training Multinomial Naive Bayes ...")
    t0 = time.time()
    # fit_prior=False forces uniform priors (50/50), preventing it from always predicting 0
    model = MultinomialNB(alpha=0.1, fit_prior=False)
    model.fit(X_train, y_train)
    print(f"[NB] Done in {time.time()-t0:.1f}s")
    joblib.dump(model, path)
    return model


# ── ensemble ────────────────────────────────────────────────────────
class SoftVoteEnsemble:
    """Soft-vote ensemble that averages predict_proba from multiple models."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probas = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                probas.append(model.predict_proba(X))
            else:
                # Fallback for models without predict_proba
                dec = model.decision_function(X)
                from scipy.special import expit
                p1 = expit(dec)
                probas.append(np.column_stack([1 - p1, p1]))
        avg = np.mean(probas, axis=0)
        return avg

    def predict(self, X):
        avg = self.predict_proba(X)
        return (avg[:, 1] >= 0.5).astype(int)


# ── main ────────────────────────────────────────────────────────────
def main():
    force = "--force" in sys.argv

    # Verify preprocessing has been run
    if not os.path.exists(os.path.join(DATA_DIR, "X_train.pkl")):
        print("ERROR: Feature matrices not found. Run preprocessing first:")
        print("  python src/preprocessing.py")
        sys.exit(1)

    # Load data
    print("[data] Loading TF-IDF features ...")
    X_train, y_train, qids_train = load_data("train")
    X_val, y_val, qids_val = load_data("val")
    print(f"[data] Train: {X_train.shape},  Val: {X_val.shape}")

    # Train models
    lr_model = train_logistic_regression(X_train, y_train, force)
    svm_model = train_svm(X_train, y_train, force)
    # nb_model = train_naive_bayes(X_train, y_train, force)

    # Build ensemble
    print("\n[ENS] Building soft-vote ensemble (LR + SVM) ...")
    ensemble = SoftVoteEnsemble([lr_model, svm_model])
    ensemble_path = os.path.join(MODEL_SAVE_DIR, "ensemble.pkl")
    joblib.dump(ensemble, ensemble_path)

    # Evaluate all models
    results = []
    results.append(evaluate_model("Logistic Regression", lr_model, X_val, y_val, qids_val))
    results.append(evaluate_model("SVM (Linear)", svm_model, X_val, y_val, qids_val))
    # results.append(evaluate_model("Naive Bayes", nb_model, X_val, y_val, qids_val))
    results.append(evaluate_model("Ensemble (Soft Vote)", ensemble, X_val, y_val, qids_val))

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "model_a_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[results] Saved to {results_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'EM':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['model']:<25} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['exact_match']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
