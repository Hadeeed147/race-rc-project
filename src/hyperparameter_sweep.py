"""
Session 5 — Hyperparameter sweep on the train split (CV) + ensemble-weight
search on val. Test set is NOT touched here.

Searches:
  LR (LogisticRegression, balanced, liblinear)        C in {0.1, 0.5, 1.0, 2.0, 5.0}
  LinearSVC (calibrated, balanced) inner C            C in {0.1, 0.5, 1.0, 2.0}
  ComplementNB                                        alpha in {0.1, 0.5, 1.0, 2.0, 5.0}
  Ensemble weights {equal, lr-heavy, nb-heavy, em-weighted}

CV: 3-fold StratifiedKFold on the TRAIN split. Scoring: macro-F1.
Refits the best estimator per family on the full TRAIN split, evaluates on
VAL, then tries the four weight schemes and keeps the best one by val EM.

Saves:
  models/hyperparameter_sweep.json
  models/lr_classifier.pkl   (overwrite, tuned)
  models/svm_classifier.pkl  (overwrite, tuned)
  models/nb_classifier.pkl   (overwrite, tuned)
  models/ensemble.pkl        (overwrite, tuned weights)

Note: GridSearchCV on 281k rows × 20k features is feasible but slow. We
subsample TRAIN to SWEEP_N rows for the CV, then refit the best on the
full TRAIN. This is standard practice for sweeping wide hyperparam grids.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

SWEEP_N = 80_000          # subsample size for the CV. Refit best on full TRAIN.
RAND = 42


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
    return float((merged["pred"] == merged["gold"]).mean())


def per_option_y_pred(probs_pos, qids):
    df = pd.DataFrame({"qid": qids, "p": probs_pos})
    df["y_pred"] = 0
    df.loc[df.groupby("qid")["p"].idxmax(), "y_pred"] = 1
    return df["y_pred"].values


def eval_on_val(name, probs_pos, y_val, qid_val, opt_val):
    em = question_argmax_em(probs_pos, qid_val, opt_val, y_val)
    yp = per_option_y_pred(probs_pos, qid_val)
    acc = accuracy_score(y_val, yp)
    f1 = f1_score(y_val, yp, average="macro")
    print(f"  {name:<32} Acc={acc:.4f}  F1={f1:.4f}  EM={em:.4f}")
    return {"accuracy": float(acc), "macro_f1": float(f1), "exact_match": em}


def main():
    print("=" * 60)
    print("Session 5 — Hyperparameter sweep")
    print("=" * 60)

    print("Loading splits ...")
    X_tr, y_tr, _, _ = load_split("train")
    X_va, y_va, qid_va, opt_va = load_split("val")
    print(f"  TRAIN: {X_tr.shape},  VAL: {X_va.shape}")

    rng = np.random.RandomState(RAND)
    if SWEEP_N and X_tr.shape[0] > SWEEP_N:
        idx = rng.choice(X_tr.shape[0], SWEEP_N, replace=False)
        X_sw, y_sw = X_tr[idx], y_tr[idx]
    else:
        X_sw, y_sw = X_tr, y_tr
    print(f"  CV sweep sample: {X_sw.shape}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RAND)
    results = {}

    # -------- LR --------
    print("\n--- LR sweep ---")
    t0 = time.time()
    lr_grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000,
                           solver="liblinear", random_state=RAND),
        param_grid={"C": [0.1, 0.5, 1.0, 2.0, 5.0]},
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=False, verbose=0,
    )
    lr_grid.fit(X_sw, y_sw)
    print(f"  best C: {lr_grid.best_params_}, cv f1_macro={lr_grid.best_score_:.4f}  ({time.time()-t0:.0f}s)")
    best_lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                                 solver="liblinear", random_state=RAND,
                                 C=lr_grid.best_params_["C"])
    t0 = time.time()
    best_lr.fit(X_tr, y_tr)
    print(f"  refit on full TRAIN: {time.time()-t0:.0f}s")
    p_lr = best_lr.predict_proba(X_va)[:, 1]
    results["lr"] = {**eval_on_val("LR (tuned)", p_lr, y_va, qid_va, opt_va),
                     "best_params": lr_grid.best_params_,
                     "cv_macro_f1": float(lr_grid.best_score_)}
    joblib.dump(best_lr, os.path.join(MODELS_DIR, "lr_classifier.pkl"))

    # -------- ComplementNB --------
    print("\n--- NB sweep ---")
    t0 = time.time()
    nb_grid = GridSearchCV(
        ComplementNB(),
        param_grid={"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]},
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=False, verbose=0,
    )
    nb_grid.fit(X_sw, y_sw)
    print(f"  best alpha: {nb_grid.best_params_}, cv f1_macro={nb_grid.best_score_:.4f}  ({time.time()-t0:.0f}s)")
    best_nb = ComplementNB(alpha=nb_grid.best_params_["alpha"])
    best_nb.fit(X_tr, y_tr)
    p_nb = best_nb.predict_proba(X_va)[:, 1]
    results["nb"] = {**eval_on_val("NB (tuned)", p_nb, y_va, qid_va, opt_va),
                     "best_params": nb_grid.best_params_,
                     "cv_macro_f1": float(nb_grid.best_score_)}
    joblib.dump(best_nb, os.path.join(MODELS_DIR, "nb_classifier.pkl"))

    # -------- LinearSVC (calibrated) --------
    # GridSearchCV on a calibrated wrapper triples the inner cv cost. We
    # search over inner C manually and pick the best by val EM (cheap).
    print("\n--- SVC sweep (manual, calibrated) ---")
    svc_results = []
    for C in (0.1, 0.5, 1.0, 2.0):
        t0 = time.time()
        base = LinearSVC(C=C, class_weight="balanced", max_iter=2000, random_state=RAND)
        cal = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        cal.fit(X_sw, y_sw)
        ps = cal.predict_proba(X_va)[:, 1]
        em = question_argmax_em(ps, qid_va, opt_va, y_va)
        f1 = f1_score(y_va, per_option_y_pred(ps, qid_va), average="macro")
        print(f"  C={C:>4}  val Macro-F1={f1:.4f}  EM={em:.4f}   ({time.time()-t0:.0f}s)")
        svc_results.append({"C": C, "macro_f1": float(f1), "exact_match": em})
    best_C = max(svc_results, key=lambda r: r["exact_match"])["C"]
    print(f"  best C by val EM: {best_C}")
    base = LinearSVC(C=best_C, class_weight="balanced", max_iter=2000, random_state=RAND)
    best_svc = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    t0 = time.time()
    best_svc.fit(X_tr, y_tr)
    print(f"  refit on full TRAIN: {time.time()-t0:.0f}s")
    p_svc = best_svc.predict_proba(X_va)[:, 1]
    results["svc"] = {**eval_on_val("SVC (tuned)", p_svc, y_va, qid_va, opt_va),
                      "best_params": {"C": best_C},
                      "sweep": svc_results}
    joblib.dump(best_svc, os.path.join(MODELS_DIR, "svm_classifier.pkl"))

    # -------- Ensemble weight search --------
    print("\n--- Ensemble weight search (val) ---")
    schemes = {
        "equal":       (1/3, 1/3, 1/3),
        "lr_heavy":    (0.50, 0.25, 0.25),
        "nb_heavy":    (0.25, 0.25, 0.50),
        "svc_heavy":   (0.25, 0.50, 0.25),
        "em_weighted": None,                  # filled below
    }
    em_lr  = results["lr"]["exact_match"]
    em_svc = results["svc"]["exact_match"]
    em_nb  = results["nb"]["exact_match"]
    s = em_lr + em_svc + em_nb
    schemes["em_weighted"] = (em_lr / s, em_svc / s, em_nb / s)

    best_scheme = None
    best_em = -1.0
    weight_table = {}
    for name, (w_lr, w_svc, w_nb) in schemes.items():
        ps = w_lr * p_lr + w_svc * p_svc + w_nb * p_nb
        em = question_argmax_em(ps, qid_va, opt_va, y_va)
        yp = per_option_y_pred(ps, qid_va)
        f1 = f1_score(y_va, yp, average="macro")
        weight_table[name] = {"weights": [round(w_lr, 4), round(w_svc, 4), round(w_nb, 4)],
                              "macro_f1": float(f1), "exact_match": em}
        print(f"  {name:<13}  w={[round(w_lr,3), round(w_svc,3), round(w_nb,3)]}   F1={f1:.4f}  EM={em:.4f}")
        if em > best_em:
            best_em = em; best_scheme = name

    chosen = schemes[best_scheme]
    print(f"\n  best scheme: {best_scheme}  EM={best_em:.4f}")
    bundle = {"models": [best_lr, best_svc, best_nb], "weights": list(chosen)}
    joblib.dump(bundle, os.path.join(MODELS_DIR, "ensemble.pkl"))
    print(f"  saved tuned ensemble.pkl")

    # Default-vs-tuned summary (re-load default-trained baseline numbers from S3)
    print("\n" + "=" * 60)
    print("Default (S3) vs Tuned (S5) — val")
    print("=" * 60)
    try:
        with open(os.path.join(MODELS_DIR, "model_a_metrics.json")) as f:
            old = json.load(f)
    except Exception:
        old = []
    old_map = {m["name"].split("(")[0].strip(): m for m in old}

    print(f"{'Component':<24} {'Default Acc':>11} {'Tuned Acc':>11} {'Default F1':>11} {'Tuned F1':>11} {'Default EM':>11} {'Tuned EM':>11}")
    for short, key in [("Logistic Regression", "lr"),
                       ("LinearSVC", "svc"),
                       ("ComplementNB", "nb"),
                       ("Ensemble", "ensemble")]:
        if key == "ensemble":
            tuned = {"accuracy": None, "macro_f1": float(f1), "exact_match": float(best_em)}
            tuned["accuracy"] = float(accuracy_score(
                y_va, per_option_y_pred(
                    chosen[0]*p_lr + chosen[1]*p_svc + chosen[2]*p_nb, qid_va)))
        else:
            tuned = results[key]
        old_m = old_map.get(short, {})
        print(f"{short:<24} {old_m.get('accuracy',0):>11.4f} {tuned['accuracy']:>11.4f} "
              f"{old_m.get('macro_f1',0):>11.4f} {tuned['macro_f1']:>11.4f} "
              f"{old_m.get('exact_match',0):>11.4f} {tuned['exact_match']:>11.4f}")

    # Save sweep summary
    out = {
        "sweep_n_train": int(X_sw.shape[0]),
        "results": results,
        "weight_table": weight_table,
        "best_scheme": best_scheme,
        "tuned_ensemble_em": float(best_em),
    }
    with open(os.path.join(MODELS_DIR, "hyperparameter_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> models/hyperparameter_sweep.json")


if __name__ == "__main__":
    main()
