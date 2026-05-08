"""
RACE preprocessing — Session 1 + Session 2.

Phase 1: Loads data/raw/train.csv, performs an 80/10/10 split (random_state=42),
         and saves data/processed/train_split.csv, val_split.csv, test_split.csv.

Phase 2: Builds TF-IDF feature matrices for Model A answer verification.
         - Explodes each question into 4 rows (one per option A/B/C/D)
         - combined text = article + " " + question + " " + option
         - Labels: 1 if option is correct answer, 0 otherwise
         - Fits TfidfVectorizer on train only, transforms val/test
         - Saves vectorizer + feature matrices + labels with joblib
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

# ── paths ───────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

INPUT_CSV = os.path.join(RAW_DIR, "train.csv")
TRAIN_OUT = os.path.join(PROC_DIR, "train_split.csv")
VAL_OUT = os.path.join(PROC_DIR, "val_split.csv")
TEST_OUT = os.path.join(PROC_DIR, "test_split.csv")

OPTIONS = ["A", "B", "C", "D"]


# ── Phase 1: Split raw CSV ─────────────────────────────────────────
def split_data(force=False):
    """Load raw train.csv and do 80/10/10 split. Skips if files exist."""
    if not force and all(os.path.exists(p) for p in [TRAIN_OUT, VAL_OUT, TEST_OUT]):
        print("[split] Splits already exist — skipping.  (use --force to redo)")
        return

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found.")
        print("Place the RACE train.csv in data/raw/ and re-run.")
        sys.exit(1)

    os.makedirs(PROC_DIR, exist_ok=True)

    print(f"[split] Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['A', 'B', 'C', 'D'])
    print(f"[split] Total rows: {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"[split]   train : {len(train_df):>6} rows")
    print(f"[split]   val   : {len(val_df):>6} rows")
    print(f"[split]   test  : {len(test_df):>6} rows")


# ── Phase 2: Explode + TF-IDF ──────────────────────────────────────
def get_overlap_features(text1, text2):
    """Calculates normalized word overlap between two strings."""
    set1 = set(str(text1).lower().split())
    set2 = set(str(text2).lower().split())
    if not set1: return 0.0
    return float(len(set1.intersection(set2)) / len(set1))


def explode_options(df):
    """
    Convert each question row into 4 rows (one per option A/B/C/D).

    Returns:
        texts    : list[str]  — combined "article question option" strings
        labels   : np.ndarray — 1 if option is correct, 0 otherwise
        qids     : list       — question index (for Exact-Match grouping)
        overlaps : np.ndarray — 2D array of overlap features
    """
    texts = []
    labels = []
    qids = []
    overlaps = []

    for idx, row in df.iterrows():
        article = str(row["article"])
        question = str(row["question"])
        correct = str(row["answer"]).strip().upper()

        for opt in OPTIONS:
            option_text = str(row[opt])
            combined = f"{article} {question} {option_text}"
            texts.append(combined)
            labels.append(1 if opt == correct else 0)
            qids.append(idx)
            
            overlap_art = get_overlap_features(option_text, article)
            overlap_q = get_overlap_features(option_text, question)
            opt_len = len(str(option_text).split())
            overlaps.append([overlap_art, overlap_q, opt_len])

    return texts, np.array(labels, dtype=np.int8), qids, np.array(overlaps, dtype=np.float32)


def build_features(force=False):
    """Fit TF-IDF on train, transform train/val/test, save everything."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    if not force and os.path.exists(vectorizer_path):
        print("[tfidf] Feature files already exist — skipping.  (use --force to redo)")
        return

    # ── load splits ─────────────────────────────────────────────────
    print("[tfidf] Loading splits ...")
    train_df = pd.read_csv(TRAIN_OUT)
    val_df = pd.read_csv(VAL_OUT)
    test_df = pd.read_csv(TEST_OUT)

    # ── explode into 4 rows per question ────────────────────────────
    print("[tfidf] Exploding options (4 rows per question) and extracting overlap ...")
    train_texts, train_labels, train_qids, train_overlaps = explode_options(train_df)
    val_texts, val_labels, val_qids, val_overlaps = explode_options(val_df)
    test_texts, test_labels, test_qids, test_overlaps = explode_options(test_df)

    print(f"[tfidf]   train : {len(train_texts):>7} samples  ({len(train_df)} questions × 4)")
    print(f"[tfidf]   val   : {len(val_texts):>7} samples  ({len(val_df)} questions × 4)")
    print(f"[tfidf]   test  : {len(test_texts):>7} samples  ({len(test_df)} questions × 4)")
    print(f"[tfidf]   label balance (train): pos={train_labels.sum()}, neg={(1-train_labels).sum()}")

    # ── fit TF-IDF on TRAIN ONLY ────────────────────────────────────
    print("[tfidf] Fitting TfidfVectorizer on train (this may take a minute) ...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=50000,    # Reverted to 50,000 to keep crucial option words
        sublinear_tf=True,     # log(1 + tf) — standard for text classification
        ngram_range=(1, 2),    # unigrams + bigrams for richer features
        min_df=3,              # ignore very rare terms
        max_df=0.95,           # ignore terms in >95% of docs
        dtype=np.float32,      # save memory
    )
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_train = hstack([X_train_tfidf, train_overlaps])
    print(f"[tfidf]   fit_transform done in {time.time()-t0:.1f}s  ->  shape {X_train.shape}")

    # ── transform val and test (NO refit!) ──────────────────────────
    print("[tfidf] Transforming val ...")
    X_val_tfidf = vectorizer.transform(val_texts)
    X_val = hstack([X_val_tfidf, val_overlaps])

    print("[tfidf] Transforming test ...")
    X_test_tfidf = vectorizer.transform(test_texts)
    X_test = hstack([X_test_tfidf, test_overlaps])

    # ── save everything ─────────────────────────────────────────────
    print("[tfidf] Saving to models/ ...")
    joblib.dump(vectorizer, vectorizer_path)

    joblib.dump(X_train, os.path.join(MODEL_DIR, "X_train.pkl"))
    joblib.dump(train_labels, os.path.join(MODEL_DIR, "y_train.pkl"))
    joblib.dump(train_qids, os.path.join(MODEL_DIR, "qids_train.pkl"))

    joblib.dump(X_val, os.path.join(MODEL_DIR, "X_val.pkl"))
    joblib.dump(val_labels, os.path.join(MODEL_DIR, "y_val.pkl"))
    joblib.dump(val_qids, os.path.join(MODEL_DIR, "qids_val.pkl"))

    joblib.dump(X_test, os.path.join(MODEL_DIR, "X_test.pkl"))
    joblib.dump(test_labels, os.path.join(MODEL_DIR, "y_test.pkl"))
    joblib.dump(test_qids, os.path.join(MODEL_DIR, "qids_test.pkl"))

    print("[tfidf] All done!")
    print(f"[tfidf]   Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"[tfidf]   X_train         : {X_train.shape}")
    print(f"[tfidf]   X_val           : {X_val.shape}")
    print(f"[tfidf]   X_test          : {X_test.shape}")


# ── main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force = "--force" in sys.argv
    split_data(force=force)
    build_features(force=force)
