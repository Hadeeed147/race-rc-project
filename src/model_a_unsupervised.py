"""
Session 2 — Model A unsupervised + semi-supervised.

K-Means:
  MiniBatchKMeans(n_clusters=4) on a 20k sample of the option-level TF-IDF
  matrix. Reports Silhouette (5k subsample, since silhouette is O(n^2))
  and Cluster Purity vs the y labels. Saves to models/kmeans.pkl.

Label Propagation:
  LabelPropagation requires dense input -> reduce TF-IDF to 100 dims with
  TruncatedSVD. Sample N rows, mark 10% as labeled (rest = -1), fit with
  kernel='knn', n_neighbors=7. Compare F1 vs a fully-supervised LR trained
  on the same labeled subset. Saves SVD + label-prop model.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, f1_score, classification_report
from sklearn.semi_supervised import LabelPropagation

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

KMEANS_SAMPLE = 20_000
SILHOUETTE_SAMPLE = 5_000
LP_SAMPLE = 5_000          # LabelPropagation memory ~ O(n^2)
LP_LABELED_FRAC = 0.10


def load(split):
    X = sp.load_npz(os.path.join(DATA_DIR, f"X_{split}.npz"))
    y = np.load(os.path.join(DATA_DIR, f"y_{split}.npy"))
    return X, y


def cluster_purity(clusters, labels):
    """Sum of majority-class counts per cluster, divided by total."""
    ct = pd.crosstab(pd.Series(clusters, name="c"), pd.Series(labels, name="y"))
    return ct.max(axis=1).sum() / ct.values.sum()


def main():
    rng = np.random.RandomState(42)

    print("=" * 60)
    print("Session 2 — Model A unsupervised + semi-supervised")
    print("=" * 60)

    print("\nLoading splits ...")
    X_train, y_train = load("train")
    X_val,   y_val   = load("val")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    # =========================================================
    # K-Means
    # =========================================================
    print("\n----- K-Means (MiniBatch) -----")
    n = X_train.shape[0]
    idx = rng.choice(n, size=min(KMEANS_SAMPLE, n), replace=False)
    X_km = X_train[idx]
    y_km = y_train[idx]

    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=512,
                         n_init=10, max_iter=200)
    cluster_labels = km.fit_predict(X_km)
    print(f"  fit: {time.time() - t0:.1f}s on {X_km.shape[0]} samples")

    # Silhouette on a smaller subsample (silhouette is O(n^2))
    s_idx = rng.choice(X_km.shape[0], size=min(SILHOUETTE_SAMPLE, X_km.shape[0]),
                       replace=False)
    t0 = time.time()
    sil = silhouette_score(X_km[s_idx], cluster_labels[s_idx], metric="cosine")
    print(f"  Silhouette (cosine, n={len(s_idx)}): {sil:.4f}  [{time.time()-t0:.1f}s]")

    pur = cluster_purity(cluster_labels, y_km)
    print(f"  Cluster Purity vs y: {pur:.4f}")
    print(f"  Cluster sizes: {np.bincount(cluster_labels)}")

    joblib.dump(km, os.path.join(MODELS_DIR, "kmeans.pkl"))
    print(f"  saved -> models/kmeans.pkl")

    # =========================================================
    # Label Propagation (semi-supervised)
    # =========================================================
    print("\n----- Label Propagation (semi-supervised) -----")
    lp_idx = rng.choice(n, size=min(LP_SAMPLE, n), replace=False)
    X_lp_sparse = X_train[lp_idx]
    y_lp_full   = y_train[lp_idx].copy()

    print(f"  Reducing to 100 dims with TruncatedSVD ...")
    t0 = time.time()
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_lp = svd.fit_transform(X_lp_sparse)
    X_val_dense = svd.transform(X_val)
    print(f"  SVD: {time.time() - t0:.1f}s, explained var sum: "
          f"{svd.explained_variance_ratio_.sum():.3f}")

    n_lp = X_lp.shape[0]
    n_labeled = int(LP_LABELED_FRAC * n_lp)
    perm = rng.permutation(n_lp)
    labeled_idx = perm[:n_labeled]
    y_lp = np.full(n_lp, -1, dtype=np.int64)
    y_lp[labeled_idx] = y_lp_full[labeled_idx]
    print(f"  Sample={n_lp}, labeled={n_labeled} (10%), unlabeled={n_lp - n_labeled}")

    print("  Fitting LabelPropagation(kernel='knn', n_neighbors=7) ...")
    t0 = time.time()
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=200)
    lp.fit(X_lp, y_lp)
    print(f"  fit: {time.time() - t0:.1f}s")

    pred_lp_val = lp.predict(X_val_dense)
    f1_lp = f1_score(y_val, pred_lp_val, average="macro")
    print(f"  Label Prop Macro F1 on val: {f1_lp:.4f}")
    print(classification_report(y_val, pred_lp_val, digits=4,
                                target_names=["wrong (0)", "correct (1)"]))

    # Fully-supervised LR on the same labeled subset (also dense / SVD features)
    print("  Baseline: LR(class_weight='balanced') on the same labeled subset ...")
    lr_sup = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_sup.fit(X_lp[labeled_idx], y_lp_full[labeled_idx])
    pred_sup_val = lr_sup.predict(X_val_dense)
    f1_sup = f1_score(y_val, pred_sup_val, average="macro")
    print(f"  Supervised LR Macro F1 on val (same {n_labeled} labels, dense): {f1_sup:.4f}")

    joblib.dump({"label_prop": lp, "svd": svd},
                os.path.join(MODELS_DIR, "label_prop.pkl"))
    print("  saved -> models/label_prop.pkl  (bundle of label-prop model + SVD)")

    print("\n" + "=" * 60)
    print(f"K-Means     : Silhouette={sil:.4f}  Purity={pur:.4f}")
    print(f"LabelProp   : Macro F1={f1_lp:.4f}   vs supervised LR (same labels)={f1_sup:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
