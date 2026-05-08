"""
Session 2 — Model A supervised training.

Trains three classifiers on the option-level TF-IDF matrix:
  * LogisticRegression(class_weight='balanced')
  * LinearSVC(class_weight='balanced')  wrapped in CalibratedClassifierCV
    (so it exposes predict_proba for the Session 3 soft-vote ensemble)
  * ComplementNB                        (handles 1:3 imbalance better than MultinomialNB)

For each model, prints on the val set:
  - Accuracy, Macro F1
  - classification_report (per-class precision/recall/f1)
  - Confusion matrix
  - Question-level Exact Match (option with max prob == gold answer)
Saves models to models/.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def load_split(split):
    X = sp.load_npz(os.path.join(DATA_DIR, f"X_{split}.npz"))
    y = np.load(os.path.join(DATA_DIR, f"y_{split}.npy"))
    qid = np.load(os.path.join(DATA_DIR, f"qid_{split}.npy"), allow_pickle=True)
    opt = np.load(os.path.join(DATA_DIR, f"opt_{split}.npy"), allow_pickle=True)
    return X, y, qid, opt


def question_level_exact_match(probs_pos, qids, opts, y):
    """For each question, pick option with max P(correct); compare to gold (where y==1)."""
    df = pd.DataFrame({"qid": qids, "opt": opts, "p": probs_pos, "y": y})
    pred = df.loc[df.groupby("qid")["p"].idxmax(), ["qid", "opt"]]
    pred = pred.rename(columns={"opt": "pred"})
    gold = df.loc[df["y"] == 1, ["qid", "opt"]].rename(columns={"opt": "gold"})
    merged = pred.merge(gold, on="qid", how="inner")
    return (merged["pred"] == merged["gold"]).mean(), len(merged)


def evaluate(name, model, X_val, y_val, qid_val, opt_val):
    print(f"\n----- {name} -----")
    t0 = time.time()
    y_pred = model.predict(X_val)
    probs_pos = model.predict_proba(X_val)[:, 1]
    print(f"  predict: {time.time() - t0:.1f}s")

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    em, n_q = question_level_exact_match(probs_pos, qid_val, opt_val, y_val)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Exact Match (per question, n={n_q}): {em:.4f}")
    print("  classification_report:")
    print(classification_report(y_val, y_pred, digits=4,
                                target_names=["wrong (0)", "correct (1)"]))
    cm = confusion_matrix(y_val, y_pred)
    print(f"  confusion_matrix [[TN FP][FN TP]]:\n{cm}")
    return {"name": name, "accuracy": acc, "macro_f1": macro_f1,
            "exact_match": em, "confusion_matrix": cm.tolist()}


def main():
    print("=" * 60)
    print("Session 2 — Model A supervised training")
    print("=" * 60)

    print("\nLoading splits ...")
    X_train, y_train, _, _ = load_split("train")
    X_val,   y_val, qid_val, opt_val = load_split("val")
    print(f"  X_train: {X_train.shape}, y_train sum={int(y_train.sum())}")
    print(f"  X_val  : {X_val.shape}")

    results = []

    # -- Logistic Regression --
    print("\nTraining LogisticRegression(class_weight='balanced') ...")
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            solver="liblinear", random_state=42)
    lr.fit(X_train, y_train)
    print(f"  fit: {time.time() - t0:.1f}s")
    joblib.dump(lr, os.path.join(MODELS_DIR, "lr_classifier.pkl"))
    results.append(evaluate("LogisticRegression", lr, X_val, y_val, qid_val, opt_val))

    # -- LinearSVC + Calibration --
    print("\nTraining CalibratedClassifierCV(LinearSVC(class_weight='balanced')) ...")
    t0 = time.time()
    base_svc = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
    svc = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
    svc.fit(X_train, y_train)
    print(f"  fit: {time.time() - t0:.1f}s")
    joblib.dump(svc, os.path.join(MODELS_DIR, "svm_classifier.pkl"))
    results.append(evaluate("LinearSVC (calibrated)", svc, X_val, y_val, qid_val, opt_val))

    # -- Complement Naive Bayes --
    print("\nTraining ComplementNB ...")
    t0 = time.time()
    nb = ComplementNB()
    nb.fit(X_train, y_train)
    print(f"  fit: {time.time() - t0:.1f}s")
    joblib.dump(nb, os.path.join(MODELS_DIR, "nb_classifier.pkl"))
    results.append(evaluate("ComplementNB", nb, X_val, y_val, qid_val, opt_val))

    # -- Summary --
    print("\n" + "=" * 60)
    print("Summary (val set)")
    print("=" * 60)
    print(f"{'Model':<28} {'Acc':>7} {'Macro F1':>10} {'Exact Match':>13}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<28} {r['accuracy']:>7.4f} {r['macro_f1']:>10.4f} {r['exact_match']:>13.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
