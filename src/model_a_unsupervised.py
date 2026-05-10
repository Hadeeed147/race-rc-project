"""
Session 6 — Unsupervised & Semi-Supervised Learning (Model A).
Requirement: Section 4.2.2 (20 Marks).
Implement:
1. K-Means clustering to discover latent question patterns (Silhouette + Purity).
2. Label Propagation (Semi-supervised) to improve performance with limited labels.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, f1_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def load_split(split):
    X = sp.load_npz(os.path.join(DATA_DIR, f"X_{split}.npz"))
    y = np.load(os.path.join(DATA_DIR, f"y_{split}.npy"))
    return X, y

def run_unsupervised_kmeans(n_clusters=4, samples=2000):
    print("\n" + "="*50)
    print("1. UNSUPERVISED LEARNING: K-Means Clustering")
    print("="*50)
    
    X_val, y_val = load_split("val")
    # Take a subset for clustering speed/visualization
    n = min(samples, X_val.shape[0])
    X = X_val[:n]
    y = y_val[:n]
    
    print(f"Running K-Means (k={n_clusters}) on {n} samples ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    sil = silhouette_score(X, labels)
    print(f"  Silhouette Score: {sil:.4f}")
    
    # Calculate Purity (how well clusters align with Correct/Wrong labels)
    def purity_score(y_true, y_pred):
        contingency_matrix = pd.crosstab(y_true, y_pred)
        return np.sum(contingency_matrix.max(axis=0)) / np.sum(contingency_matrix.values)
    
    purity = purity_score(y, labels)
    print(f"  Cluster Purity (vs Correct/Wrong labels): {purity:.4f}")
    
    # Discovery: Most frequent words per cluster
    # (Since we use the global TF-IDF, we can't easily get cluster-specific keywords
    # without the vectorizer. We'll just show the scores for now.)
    
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    return sil, purity

def run_semi_supervised_label_prop(n_labeled=500, total_samples=3000, kmeans_sil=0.0, kmeans_purity=0.0):
    print("\n" + "="*50)
    print("2. SEMI-SUPERVISED LEARNING: Label Propagation")
    print("="*50)
    
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    
    # Take a small subset for training
    n_tot = min(total_samples, X_train.shape[0])
    X = X_train[:n_tot].toarray() # LabelProp requires dense matrix usually
    y = y_train[:n_tot].copy()
    
    # Hide labels: set most to -1
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y)) > (n_labeled / n_tot)
    y_partial = y.copy()
    y_partial[random_unlabeled_points] = -1
    
    print(f"Total samples: {n_tot}, Labeled: {n_labeled}, Unlabeled: {n_tot - n_labeled}")
    
    # 1. Baseline: Logistic Regression on small labeled set only
    X_small = X[~random_unlabeled_points]
    y_small = y[~random_unlabeled_points]
    
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_small, y_small)
    y_pred_lr = lr.predict(X_val[:1000].toarray())
    f1_lr = f1_score(y_val[:1000], y_pred_lr, average='macro')
    
    # 2. Semi-supervised: Label Propagation
    print("Running Label Propagation ...")
    lp = LabelPropagation(kernel='knn', n_neighbors=7)
    lp.fit(X, y_partial)
    
    y_pred_lp = lp.predict(X_val[:1000].toarray())
    f1_lp = f1_score(y_val[:1000], y_pred_lp, average='macro')
    
    print(f"\nResults (Macro F1 on Val set):")
    print(f"  Supervised (LR, {n_labeled} labels): {f1_lr:.4f}")
    print(f"  Semi-Supervised (LabelProp):      {f1_lp:.4f}")
    
    improvement = (f1_lp - f1_lr) / f1_lr * 100 if f1_lr > 0 else 0
    print(f"  Relative Improvement: {improvement:+.1f}%")
    
    # Save results
    res = {
        "kmeans": {"silhouette": kmeans_sil, "purity": kmeans_purity},
        "label_prop": {"f1_baseline": f1_lr, "f1_lp": f1_lp, "improvement": improvement}
    }
    with open(os.path.join(MODELS_DIR, "unsupervised_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    
    return f1_lr, f1_lp

if __name__ == "__main__":
    sil, pur = run_unsupervised_kmeans()
    run_semi_supervised_label_prop(kmeans_sil=sil, kmeans_purity=pur)
