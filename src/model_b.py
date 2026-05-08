"""
Session 2 — Model B: distractor generation + hint generation.

Distractor pipeline:
  candidate extractor: frequent content words/bigrams from the article
  scorer: TF-IDF cosine similarity vs correct_answer
  diversity penalty: skip candidates too similar to already-selected
  output: top 3 non-answer distractors

Hint pipeline:
  split article into sentences
  per-sentence features: cosine_sim(sent, question), keyword_overlap,
    sentence_position, sentence_length
  train a LogisticRegression hint scorer on a heuristic gold label
    (top 20% of sentences by cosine to question = positive)
  rank sentences -> return three at relevance percentiles
    ~70% (vague), ~85% (moderate), ~95% (specific)

Saves models/hint_scorer.pkl. Prints sample outputs on 5 val rows.
"""

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

STOPWORDS = set(ENGLISH_STOP_WORDS)


# ---------- Helpers ----------
TOK_RE = re.compile(r"[A-Za-z]{3,}")
SENT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text):
    return [t for t in TOK_RE.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text):
    if not isinstance(text, str):
        return []
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


# ---------- Distractor generation ----------
def extract_distractors(article, correct_answer, vectorizer, top_k=3,
                        max_candidates=60, diversity_threshold=0.7):
    if not isinstance(article, str) or not article.strip():
        return []
    tokens = tokenize(article)
    if len(tokens) < 5:
        return []
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    cand_pool = [w for w, _ in Counter(tokens).most_common(50)]
    cand_pool += [b for b, _ in Counter(bigrams).most_common(20)]
    ans_low = (correct_answer or "").strip().lower()
    cand_pool = [c for c in cand_pool if c and c.lower() != ans_low and ans_low not in c.lower()]
    if not cand_pool:
        return []
    cand_pool = cand_pool[:max_candidates]

    cand_vecs = vectorizer.transform(cand_pool)
    ans_vec = vectorizer.transform([correct_answer or ""])
    sims = cosine_similarity(cand_vecs, ans_vec).ravel()

    # Rank by similarity descending; apply diversity penalty
    order = np.argsort(-sims)
    selected = []
    selected_vecs = []
    for j in order:
        if len(selected) >= top_k:
            break
        v = cand_vecs[j]
        if not selected_vecs:
            selected.append(cand_pool[j])
            selected_vecs.append(v)
            continue
        max_sim_to_picked = max(
            cosine_similarity(v, sv)[0, 0] for sv in selected_vecs
        )
        if max_sim_to_picked < diversity_threshold:
            selected.append(cand_pool[j])
            selected_vecs.append(v)
    return selected[:top_k]


# ---------- Hint generation ----------
def _sentence_features(sents, question, vectorizer):
    if not sents:
        return np.empty((0, 4))
    q_vec = vectorizer.transform([question or ""])
    sent_vecs = vectorizer.transform(sents)
    cosines = cosine_similarity(sent_vecs, q_vec).ravel()

    q_tokens = set(tokenize(question or ""))
    n = len(sents)
    feats = np.zeros((n, 4), dtype=np.float64)
    for i, s in enumerate(sents):
        s_tokens = set(tokenize(s))
        overlap = (len(s_tokens & q_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
        position = i / max(1, n - 1)
        length = len(s.split())
        feats[i] = [cosines[i], overlap, position, length]
    return feats


def train_hint_scorer(train_csv, vectorizer, n_questions=2000, random_state=42):
    df = pd.read_csv(train_csv).sample(n_questions, random_state=random_state)
    feats_all, y_all = [], []
    for _, r in df.iterrows():
        sents = split_sentences(r["article"])
        if len(sents) < 5:
            continue
        feats = _sentence_features(sents, r["question"], vectorizer)
        # heuristic gold: top 20% by cosine -> positive
        cosines = feats[:, 0]
        rank = np.argsort(-cosines)
        n_top = max(1, int(0.2 * len(sents)))
        positives = set(rank[:n_top].tolist())
        labels = np.array([1 if i in positives else 0 for i in range(len(sents))])
        feats_all.append(feats)
        y_all.append(labels)
    X = np.vstack(feats_all)
    y = np.concatenate(y_all)
    print(f"  hint-scorer training set: X={X.shape}, pos rate={y.mean():.3f}")
    scorer = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    scorer.fit(X, y)
    return scorer


def generate_hints(article, question, vectorizer, hint_scorer):
    sents = split_sentences(article)
    if len(sents) < 3:
        return (sents + ["", "", ""])[:3]
    feats = _sentence_features(sents, question, vectorizer)
    scores = hint_scorer.predict_proba(feats)[:, 1]
    order = np.argsort(-scores)  # 0 = most relevant
    n = len(order)

    # Pick at relevance percentiles: ~70% (vague), ~85% (moderate), ~95% (specific)
    targets = [int(round(n * 0.30)), int(round(n * 0.15)), int(round(n * 0.05))]
    targets = [max(0, min(t, n - 1)) for t in targets]
    picks, seen = [], set()
    for t in targets:
        # Walk outward from t until we find an unseen index
        for delta in range(n):
            for cand in (t + delta, t - delta):
                if 0 <= cand < n and cand not in seen:
                    seen.add(cand)
                    picks.append(cand)
                    break
            if len(picks) > len(targets) - len([p for p in targets if p == t]):
                break
        if len(picks) >= 3:
            break
    while len(picks) < 3:
        picks.append(picks[-1] if picks else 0)
    return [sents[order[p]] for p in picks[:3]]


# ---------- Driver ----------
def main():
    print("=" * 60)
    print("Session 2 — Model B (distractors + hints)")
    print("=" * 60)

    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    vec = joblib.load(vec_path)
    print(f"Loaded vectorizer ({vec_path})")

    print("\nTraining hint scorer ...")
    t0 = time.time()
    train_csv = os.path.join(DATA_DIR, "train_split.csv")
    scorer = train_hint_scorer(train_csv, vec, n_questions=2000)
    joblib.dump(scorer, os.path.join(MODELS_DIR, "hint_scorer.pkl"))
    print(f"  hint scorer fit + saved ({time.time() - t0:.1f}s)")

    # Demo: 5 val samples
    print("\nDemo on 5 val samples:")
    val = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv")).sample(5, random_state=7)
    for k, (_, r) in enumerate(val.iterrows(), 1):
        gold = r[r["answer"]]
        print("\n" + "-" * 60)
        print(f"Sample {k} (id={r['id']})")
        print(f"  Q: {r['question']}")
        print(f"  Gold answer ({r['answer']}): {gold}")
        d = extract_distractors(r["article"], str(gold), vec, top_k=3)
        print(f"  Distractors: {d}")
        h = generate_hints(r["article"], r["question"], vec, scorer)
        for i, hint in enumerate(h, 1):
            print(f"  Hint {i}: {hint[:140]}{'...' if len(hint) > 140 else ''}")

    print("\nDone.")


if __name__ == "__main__":
    main()
