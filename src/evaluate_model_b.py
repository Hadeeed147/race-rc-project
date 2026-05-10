"""
Session 3 — Formal Model B evaluation.

Distractors (200 val samples):
  Generate 3 distractors with model_b.extract_distractors and compare
  against the 3 gold non-answer options. Reports:
    * BLEU                   (corpus-level, multi-reference)
    * ROUGE-1, ROUGE-2, ROUGE-L (averaged)
    * METEOR                 (averaged, per pair)
    * Token-overlap Precision / Recall / F1 (averaged)

Hints:
  Generate 3 hint sentences with model_b.generate_hints. Heuristic "gold"
  per question: the article sentence with the highest token overlap with
  the correct answer. Reports:
    * Precision @ 1, Precision @ 3
    * R^2 of the LR hint-scorer's predicted probabilities vs the heuristic
      relevance score (token overlap with answer) on the val sentences.

Saves a backend-friendly JSON to models/model_b_metrics.json.
"""

import os
import sys
import json
import time
import math
import joblib
import numpy as np
import pandas as pd

# Make src/ importable so we can re-use model_b helpers
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from model_b import (  # noqa: E402
    extract_distractors,
    generate_hints,
    split_sentences,
    tokenize,
    _sentence_features,
)

ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

N_SAMPLES = 200


# --------------- Metric helpers ---------------
def token_set(s):
    return set(tokenize(s or ""))


def prf(generated, gold):
    """Token-overlap Precision/Recall/F1 over a list of generated vs gold strings."""
    g_tokens = set()
    for g in generated:
        g_tokens |= token_set(g)
    a_tokens = set()
    for a in gold:
        a_tokens |= token_set(a)
    if not g_tokens or not a_tokens:
        return 0.0, 0.0, 0.0
    tp = len(g_tokens & a_tokens)
    p = tp / max(1, len(g_tokens))
    r = tp / max(1, len(a_tokens))
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def main():
    # Lazy imports of optional packages (so import failures are clear)
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        import nltk
    except Exception as e:
        print(f"ERROR: nltk import failed: {e}")
        sys.exit(1)
    try:
        from rouge_score import rouge_scorer
    except Exception as e:
        print(f"ERROR: rouge_score import failed: {e}")
        sys.exit(1)

    # NLTK data needed for METEOR
    for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg in ("wordnet", "omw-1.4")
                           else f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import cosine_similarity

    print("=" * 60)
    print("Session 3 — Model B formal evaluation")
    print("=" * 60)

    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    scorer = joblib.load(os.path.join(MODELS_DIR, "hint_scorer.pkl"))
    print("Loaded vectorizer + hint scorer.")

    val = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv")) \
            .sample(N_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"Evaluating on {len(val)} val samples.\n")

    # --------- Distractors ---------
    print("--- Distractors ---")
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    bleu_refs, bleu_hyps = [], []
    rouge1, rouge2, rougeL = [], [], []
    meteors = []
    p_list, r_list, f_list = [], [], []
    n_skipped = 0

    t0 = time.time()
    for _, row in val.iterrows():
        article = row["article"]
        correct = str(row[row["answer"]])
        gold_distractors = [str(row[L]) for L in ["A", "B", "C", "D"] if L != row["answer"]]
        gens = extract_distractors(article, correct, vec, top_k=3)
        if not gens:
            n_skipped += 1
            continue

        # Token-overlap PRF over the bags
        p, r, f = prf(gens, gold_distractors)
        p_list.append(p); r_list.append(r); f_list.append(f)

        # ROUGE: pair each generated with each gold, take the best per generated, then average
        for g in gens:
            best_r1 = best_r2 = best_rL = 0.0
            for ref in gold_distractors:
                sc = rouge.score(ref, g)
                best_r1 = max(best_r1, sc["rouge1"].fmeasure)
                best_r2 = max(best_r2, sc["rouge2"].fmeasure)
                best_rL = max(best_rL, sc["rougeL"].fmeasure)
            rouge1.append(best_r1); rouge2.append(best_r2); rougeL.append(best_rL)

        # METEOR: tokenize and score each generated vs all gold refs (best ref)
        for g in gens:
            try:
                hyp_tok = tokenize(g)
                ref_toks = [tokenize(d) for d in gold_distractors]
                if not hyp_tok or not all(ref_toks):
                    continue
                meteors.append(float(meteor_score(ref_toks, hyp_tok)))
            except Exception:
                pass

        # Corpus BLEU lists (multi-reference at the question level)
        for g in gens:
            bleu_refs.append([tokenize(d) for d in gold_distractors])
            bleu_hyps.append(tokenize(g))

    bleu_score_val = float(corpus_bleu(bleu_refs, bleu_hyps, smoothing_function=smooth))
    print(f"  evaluated {len(val) - n_skipped} questions ({n_skipped} skipped). "
          f"({time.time() - t0:.1f}s)")
    distractor_metrics = {
        "n_questions": int(len(val) - n_skipped),
        "bleu":        bleu_score_val,
        "rouge1_f":    float(np.mean(rouge1)) if rouge1 else 0.0,
        "rouge2_f":    float(np.mean(rouge2)) if rouge2 else 0.0,
        "rougeL_f":    float(np.mean(rougeL)) if rougeL else 0.0,
        "meteor":      float(np.mean(meteors)) if meteors else 0.0,
        "precision":   float(np.mean(p_list)) if p_list else 0.0,
        "recall":      float(np.mean(r_list)) if r_list else 0.0,
        "f1":          float(np.mean(f_list)) if f_list else 0.0,
    }
    for k, v in distractor_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<10}: {v:.4f}")
        else:
            print(f"  {k:<10}: {v}")

    # --------- Hints ---------
    print("\n--- Hints ---")
    p1_list, p3_list = [], []
    r2_pred, r2_true = [], []
    n_hint_skipped = 0
    for _, row in val.iterrows():
        article = row["article"]
        question = row["question"]
        correct = str(row[row["answer"]])
        sents = split_sentences(article)
        if len(sents) < 3:
            n_hint_skipped += 1
            continue

        # Heuristic gold: sentence with highest token overlap with correct answer
        ans_tokens = token_set(correct)
        if not ans_tokens:
            n_hint_skipped += 1
            continue
        overlaps = []
        for s in sents:
            st = token_set(s)
            ov = (len(st & ans_tokens) / max(1, len(ans_tokens))) if ans_tokens else 0.0
            overlaps.append(ov)
        gold_idx = int(np.argmax(overlaps))
        gold_sent = sents[gold_idx]

        # Generate 3 hints
        hints = generate_hints(article, question, correct, vec, scorer)

        # Precision@1, Precision@3 (does the gold sentence appear in top 1/3?)
        p1 = 1.0 if hints and hints[-1].strip() == gold_sent.strip() else 0.0
        # The "specific" hint is hints[2]; treat it as the top-1 hint
        p3 = 1.0 if any(h.strip() == gold_sent.strip() for h in hints) else 0.0
        p1_list.append(p1); p3_list.append(p3)

        # R^2 collection: predicted probas vs heuristic gold scores (overlaps)
        feats = _sentence_features(sents, question, correct, vec)
        pred = scorer.predict_proba(feats)[:, 1]
        r2_pred.extend(pred.tolist())
        r2_true.extend(overlaps)

    if r2_pred:
        # r2_score(y_true, y_pred)
        r2 = float(r2_score(r2_true, r2_pred))
    else:
        r2 = 0.0
    hint_metrics = {
        "n_questions": int(len(val) - n_hint_skipped),
        "precision_at_1": float(np.mean(p1_list)) if p1_list else 0.0,
        "precision_at_3": float(np.mean(p3_list)) if p3_list else 0.0,
        "r2_scorer":     r2,
    }
    for k, v in hint_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v}")

    # ---- Save JSON ----
    metrics = {
        "distractors": distractor_metrics,
        "hints":       hint_metrics,
        "n_samples":   N_SAMPLES,
    }
    out = os.path.join(MODELS_DIR, "model_b_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
