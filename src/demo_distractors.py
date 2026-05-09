"""Session 6 — quick verification: prints 5 before/after distractor sets
on real val articles. Does NOT retrain or modify any artefact."""

import os
import sys
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model_b import extract_distractors, _single_token_fallback   # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")


def main():
    vec = joblib.load(os.path.join(MODELS, "tfidf_vectorizer.pkl"))
    val = pd.read_csv(os.path.join(DATA, "val_split.csv")).sample(5, random_state=21)

    for k, (_, r) in enumerate(val.iterrows(), 1):
        gold = str(r[r["answer"]])
        old = _single_token_fallback(r["article"], gold, vec, k=3)
        new = extract_distractors(r["article"], gold, vec, top_k=3)
        print("=" * 60)
        print(f"Sample {k} (id={r['id']})")
        print(f"  gold answer ({r['answer']}): {gold[:80]}")
        print(f"  before (single-token):  {old}")
        print(f"  after  (noun-phrases):  {new}")


if __name__ == "__main__":
    main()
