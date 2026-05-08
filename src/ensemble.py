"""
Session 3 — Soft-vote ensemble.

Averages P(y=1) from LR + LinearSVC(calibrated) + ComplementNB and selects
each question's predicted option via per-question argmax across A/B/C/D.

NEVER uses a 0.5 threshold (calibrated SVC's probas are squashed below 0.5).
The argmax-across-options strategy is the same one the backend's /predict
will use.

Saves models/ensemble.pkl as {"models": [lr, svc, nb], "weights": [1/3]*3}.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
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


def question_argmax_em(probs_pos, qids, opts, y):
    df = pd.DataFrame({"qid": qids, "opt": opts, "p": probs_pos, "y": y})
    pred = df.loc[df.groupby("qid")["p"].idxmax(), ["qid", "opt"]].rename(columns={"opt": "pred"})
    gold = df.loc[df["y"] == 1, ["qid", "opt"]].rename(columns={"opt": "gold"})
    merged = pred.merge(gold, on="qid", how="inner")
    return (merged["pred"] == merged["gold"]).mean(), len(merged)


def per_option_metrics(y_true, probs_pos, qids):
    """Per-option accuracy/F1 by deriving y_pred from per-question argmax."""
    df = pd.DataFrame({"qid": qids, "p": probs_pos, "y": y_true})
    df["y_pred"] = 0
    idx_max = df.groupby("qid")["p"].idxmax()
    df.loc[idx_max, "y_pred"] = 1
    return df["y_pred"].values


def evaluate(name, probs_pos, y_val, qid_val, opt_val):
    print(f"\n----- {name} -----")
    em, n_q = question_argmax_em(probs_pos, qid_val, opt_val, y_val)
    y_pred = per_option_metrics(y_val, probs_pos, qid_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Exact Match (per question, n={n_q}): {em:.4f}")
    print("  classification_report:")
    print(classification_report(y_val, y_pred, digits=4,
                                target_names=["wrong (0)", "correct (1)"]))
    cm = confusion_matrix(y_val, y_pred)
    print(f"  confusion_matrix [[TN FP][FN TP]]:\n{cm}")
    return {"name": name, "accuracy": float(acc), "macro_f1": float(macro_f1),
            "exact_match": float(em), "confusion_matrix": cm.tolist()}


def main():
    print("=" * 60)
    print("Session 3 — Soft-vote ensemble")
    print("=" * 60)
    print("Loading models ...")
    lr = joblib.load(os.path.join(MODELS_DIR, "lr_classifier.pkl"))
    svc = joblib.load(os.path.join(MODELS_DIR, "svm_classifier.pkl"))
    nb = joblib.load(os.path.join(MODELS_DIR, "nb_classifier.pkl"))

    print("Loading val split ...")
    X_val, y_val, qid_val, opt_val = load_split("val")
    print(f"  X_val: {X_val.shape}")

    t0 = time.time()
    p_lr = lr.predict_proba(X_val)[:, 1]
    p_svc = svc.predict_proba(X_val)[:, 1]
    p_nb = nb.predict_proba(X_val)[:, 1]
    print(f"  predict_proba (3 models): {time.time() - t0:.1f}s")
    p_ens = (p_lr + p_svc + p_nb) / 3.0

    results = []
    results.append(evaluate("Logistic Regression", p_lr, y_val, qid_val, opt_val))
    results.append(evaluate("LinearSVC (calibrated)", p_svc, y_val, qid_val, opt_val))
    results.append(evaluate("ComplementNB", p_nb, y_val, qid_val, opt_val))
    results.append(evaluate("Ensemble (avg)", p_ens, y_val, qid_val, opt_val))

    # Save ensemble bundle
    bundle = {"models": [lr, svc, nb], "weights": [1/3, 1/3, 1/3]}
    out_path = os.path.join(MODELS_DIR, "ensemble.pkl")
    joblib.dump(bundle, out_path)
    print(f"\nSaved ensemble bundle -> {out_path}")

    # Save metrics JSON for analytics endpoint
    metrics_path = os.path.join(MODELS_DIR, "model_a_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved per-model metrics -> {metrics_path}")

    print("\n" + "=" * 60)
    print("Comparison (val set, per-question argmax across A/B/C/D)")
    print("=" * 60)
    print(f"{'Model':<28} {'Acc':>7} {'Macro F1':>10} {'Exact Match':>13}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<28} {r['accuracy']:>7.4f} {r['macro_f1']:>10.4f} {r['exact_match']:>13.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
