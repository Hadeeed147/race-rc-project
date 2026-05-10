"""
Session 5 — Standalone BLEU / ROUGE / METEOR evaluation script.

Loads test_split.csv, runs the improved distractor pipeline,
and computes corpus-level BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR.

Compares against the old metrics in models/model_b_metrics.json if available.
Saves new results to models/eval_metrics_v2.json.

Usage:
    python src/evaluate_metrics.py [--samples N] [--split val|test]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib

# Make src/ importable
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from model_b import extract_distractors, tokenize, split_sentences  # noqa: E402

ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def main():
    parser = argparse.ArgumentParser(description="Evaluate distractor quality")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples to evaluate (default: 500)")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Which split to evaluate on (default: test)")
    args = parser.parse_args()

    # ---- Imports ----
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        import nltk
    except ImportError as e:
        print(f"ERROR: nltk import failed: {e}")
        print("Install with: pip install nltk")
        sys.exit(1)

    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        print(f"ERROR: rouge_score import failed: {e}")
        print("Install with: pip install rouge-score")
        sys.exit(1)

    # Ensure NLTK data
    for pkg in ("wordnet", "omw-1.4", "punkt_tab"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg in ("wordnet", "omw-1.4")
                           else f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    print("=" * 70)
    print(f"  Distractor Evaluation — {args.split}_split, {args.samples} samples")
    print("=" * 70)

    # ---- Load models ----
    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("Loaded TF-IDF vectorizer.")

    # ---- Load data ----
    csv_path = os.path.join(DATA_DIR, f"{args.split}_split.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    n = min(args.samples, len(df))
    df = df.sample(n, random_state=42).reset_index(drop=True)
    print(f"Evaluating on {n} samples from {args.split}_split.csv\n")

    # ---- Evaluate ----
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    bleu_refs, bleu_hyps = [], []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    meteor_scores = []
    n_skipped = 0

    t0 = time.time()
    for idx, row in df.iterrows():
        article = row["article"]
        correct = str(row[row["answer"]])
        question = row["question"]
        gold_distractors = [str(row[L]) for L in ["A", "B", "C", "D"]
                            if L != row["answer"]]

        gens = extract_distractors(article, correct, vec, top_k=3,
                                   question=question)
        if not gens:
            n_skipped += 1
            continue

        # ---- ROUGE ----
        for g in gens:
            best_r1 = best_r2 = best_rL = 0.0
            for ref in gold_distractors:
                sc = rouge.score(ref, g)
                best_r1 = max(best_r1, sc["rouge1"].fmeasure)
                best_r2 = max(best_r2, sc["rouge2"].fmeasure)
                best_rL = max(best_rL, sc["rougeL"].fmeasure)
            rouge1_scores.append(best_r1)
            rouge2_scores.append(best_r2)
            rougeL_scores.append(best_rL)

        # ---- METEOR ----
        for g in gens:
            try:
                hyp_tok = tokenize(g)
                ref_toks = [tokenize(d) for d in gold_distractors]
                if hyp_tok and all(ref_toks):
                    meteor_scores.append(float(meteor_score(ref_toks, hyp_tok)))
            except Exception:
                pass

        # ---- Corpus BLEU ----
        for g in gens:
            bleu_refs.append([tokenize(d) for d in gold_distractors])
            bleu_hyps.append(tokenize(g))

        # Progress
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  ... processed {idx + 1}/{n} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    bleu_val = float(corpus_bleu(bleu_refs, bleu_hyps,
                                  smoothing_function=smooth)) if bleu_hyps else 0.0

    # ---- Results ----
    new_metrics = {
        "split": args.split,
        "n_evaluated": int(n - n_skipped),
        "n_skipped": int(n_skipped),
        "bleu": bleu_val,
        "rouge1_f": float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        "rouge2_f": float(np.mean(rouge2_scores)) if rouge2_scores else 0.0,
        "rougeL_f": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        "meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
        "eval_time_sec": round(elapsed, 1),
    }

    # ---- Print results ----
    print(f"\n{'=' * 70}")
    print(f"  RESULTS ({n - n_skipped} questions, {n_skipped} skipped, {elapsed:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<12} {'New (v2)':>10}")
    print(f"  {'-' * 24}")
    for k in ["bleu", "rouge1_f", "rouge2_f", "rougeL_f", "meteor"]:
        print(f"  {k:<12} {new_metrics[k]:>10.4f}")

    # ---- Compare with old metrics ----
    old_path = os.path.join(MODELS_DIR, "model_b_metrics.json")
    if os.path.exists(old_path):
        with open(old_path) as f:
            old = json.load(f)
        old_d = old.get("distractors", {})
        print(f"\n  {'=' * 50}")
        print(f"  COMPARISON (v1 → v2)")
        print(f"  {'=' * 50}")
        print(f"  {'Metric':<12} {'Old (v1)':>10} {'New (v2)':>10} {'Change':>10}")
        print(f"  {'-' * 44}")
        for k_new, k_old in [("bleu", "bleu"), ("rouge1_f", "rouge1_f"),
                              ("rouge2_f", "rouge2_f"), ("rougeL_f", "rougeL_f"),
                              ("meteor", "meteor")]:
            old_val = old_d.get(k_old, 0.0)
            new_val = new_metrics[k_new]
            delta = new_val - old_val
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {k_new:<12} {old_val:>10.4f} {new_val:>10.4f} {arrow}{abs(delta):>8.4f}")

    # ---- Save ----
    out_path = os.path.join(MODELS_DIR, "eval_metrics_v2.json")
    with open(out_path, "w") as f:
        json.dump(new_metrics, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
