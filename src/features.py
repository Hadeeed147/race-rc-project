"""
Session 2 — TF-IDF feature pipeline.

Loads the three split CSVs, reshapes each to option-level (4 rows per
question), fits TfidfVectorizer on TRAIN only, transforms val + test,
and persists the vectorizer and feature matrices to disk.

Outputs:
  models/tfidf_vectorizer.pkl
  data/X_{train,val,test}.npz   (sparse TF-IDF matrices)
  data/y_{train,val,test}.npy   (option-level binary labels: 1 = correct)
  data/qid_{train,val,test}.npy (question id per option row)
  data/opt_{train,val,test}.npy (option label A/B/C/D per row)
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OPTION_LABELS = ["A", "B", "C", "D"]


def reshape_to_option_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per question -> four rows (one per option)."""
    long = df.melt(
        id_vars=["id", "article", "question", "answer"],
        value_vars=OPTION_LABELS,
        var_name="option_label",
        value_name="option_text",
    )
    long["y"] = (long["answer"] == long["option_label"]).astype(np.int8)
    long = long.drop(columns=["answer"])
    # Stable order: by id, then option label (A,B,C,D)
    long["__ord"] = long["option_label"].map({l: i for i, l in enumerate(OPTION_LABELS)})
    long = long.sort_values(["id", "__ord"]).drop(columns="__ord").reset_index(drop=True)
    return long


def build_combined(df_long: pd.DataFrame) -> np.ndarray:
    a = df_long["article"].astype(str).fillna("")
    q = df_long["question"].astype(str).fillna("")
    o = df_long["option_text"].astype(str).fillna("")
    return (a + " " + q + " " + o).values


def main():
    print("=" * 60)
    print("Session 2 — Feature pipeline")
    print("=" * 60)

    paths = {
        "train": os.path.join(DATA_DIR, "train_split.csv"),
        "val":   os.path.join(DATA_DIR, "val_split.csv"),
        "test":  os.path.join(DATA_DIR, "test_split.csv"),
    }

    longs = {}
    for split, path in paths.items():
        t0 = time.time()
        df = pd.read_csv(path)
        df_long = reshape_to_option_level(df)
        longs[split] = df_long
        print(f"[{split}] {len(df):>6} questions -> {len(df_long):>7} option rows "
              f"({time.time() - t0:.1f}s)")

    # -- Fit TF-IDF on TRAIN only --
    print("\nFitting TfidfVectorizer on TRAIN ...")
    train_text = build_combined(longs["train"])
    vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    t0 = time.time()
    X_train = vec.fit_transform(train_text)
    print(f"  vocab: {len(vec.vocabulary_):>6}, X_train: {X_train.shape}, "
          f"nnz: {X_train.nnz:,} ({time.time() - t0:.1f}s)")

    print("Transforming val/test ...")
    X_val  = vec.transform(build_combined(longs["val"]))
    X_test = vec.transform(build_combined(longs["test"]))
    print(f"  X_val: {X_val.shape}, X_test: {X_test.shape}")

    # -- Persist artifacts --
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vec, vec_path)
    print(f"\nSaved vectorizer -> {vec_path}")

    for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        sp.save_npz(os.path.join(DATA_DIR, f"X_{split}.npz"), X)
        np.save(os.path.join(DATA_DIR, f"y_{split}.npy"),
                longs[split]["y"].values.astype(np.int8))
        np.save(os.path.join(DATA_DIR, f"qid_{split}.npy"),
                longs[split]["id"].values)
        np.save(os.path.join(DATA_DIR, f"opt_{split}.npy"),
                longs[split]["option_label"].values)
        print(f"  saved {split} matrices ({X.shape}) + labels")

    # Sanity check on label distribution
    y_train = longs["train"]["y"].values
    print(f"\nTRAIN class balance: 1={int(y_train.sum())}, 0={int((1 - y_train).sum())} "
          f"(ratio {y_train.mean():.3f})")
    print("Done.")


if __name__ == "__main__":
    main()
