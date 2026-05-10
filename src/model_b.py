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
from typing import List
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

STOPWORDS = set(ENGLISH_STOP_WORDS)


# ---------- Helpers ----------
TOK_RE = re.compile(r"[A-Za-z]{3,}")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z\-']*\b")
# 1-3 capitalized words OR 2-3 lowercase words. Used for noun-phrase distractors.
NP_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+\s+){0,2}[A-Z][a-z]+\b|"
    r"\b(?:[a-z]+\s+){1,2}[a-z]+\b"
)
# Abbreviations that should NOT trigger sentence splits
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "jr", "sr", "st", "vs", "etc",
           "vol", "dept", "est", "approx", "govt", "inc", "ltd", "co",
           "gen", "sgt", "cpl", "pvt", "rev", "hon", "pres"}
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ABBREV, key=len, reverse=True)) + r")\.\s",
    flags=re.IGNORECASE,
)
_LETTER_JUNK = {"yours", "sincerely", "dear", "regards", "cheers", "thanks"}


def tokenize(text):
    return [t for t in TOK_RE.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text):
    """Improved sentence splitter that protects abbreviations and filters junk."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Step 1: Protect abbreviations
    protected = text
    for m in reversed(list(_ABBREV_PATTERN.finditer(protected))):
        abbr = m.group(1)
        protected = protected[:m.start()] + abbr + "<DOT> " + protected[m.end():]
    # Step 2: Protect initials like "S.H.E." or "U.S."
    protected = re.sub(r'\b([A-Z])\.([A-Z])\.', r'\1<DOT>\2<DOT>', protected)
    # Step 3: Split on sentence-ending punctuation followed by space+uppercase
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', protected)
    # Step 4: Restore placeholders and filter
    out = []
    for s in raw:
        s = s.replace("<DOT>", ".").strip()
        if not s:
            continue
        words = s.split()
        if len(words) < 4:
            continue
        # Skip letter sign-offs and greetings
        first_low = words[0].lower().rstrip(".,;:")
        if first_low in _LETTER_JUNK:
            continue
        out.append(s)
    return out


def extract_noun_phrases(article, max_words=3, min_words=2):
    """Pull 2–3 word noun phrases from the article via the NP_PATTERN regex.

    Filtering rules: must be `min_words..max_words` long; first and last
    tokens cannot be stopwords; phrase must be unique (case-insensitive).
    """
    if not isinstance(article, str) or not article.strip():
        return []
    seen, out = set(), []
    for m in NP_PATTERN.finditer(article):
        phrase = m.group(0).strip()
        words = phrase.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if words[0].lower() in STOPWORDS or words[-1].lower() in STOPWORDS:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _get_category(phrase):
    """Simple heuristic to categorize the answer/candidate."""
    if not phrase: return "other"
    low = phrase.lower()
    if re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", phrase):
        return "year"
    if re.search(r"\b\d+\b", phrase):
        return "number"
    # Proper noun: contains capitalized word not at start? 
    # Or just capitalized words in general.
    if any(w[0].isupper() for w in phrase.split()):
        return "proper"
    return "common"


def _single_token_fallback(article, correct_answer, vectorizer, k):
    """The Session-2 single-token / bigram extractor, used to backfill
    when noun-phrase candidates are too few. Same scoring + diversity logic."""
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
    cand_pool = cand_pool[:60]
    cand_vecs = vectorizer.transform(cand_pool)
    ans_vec = vectorizer.transform([correct_answer or ""])
    sims = cosine_similarity(cand_vecs, ans_vec).ravel()
    order = np.argsort(-sims)
    selected, selected_vecs = [], []
    for j in order:
        if len(selected) >= k:
            break
        v = cand_vecs[j]
        if not selected_vecs:
            selected.append(cand_pool[j]); selected_vecs.append(v); continue
        if max(cosine_similarity(v, sv)[0, 0] for sv in selected_vecs) < 0.7:
            selected.append(cand_pool[j]); selected_vecs.append(v)
    return selected


# ---------- Distractor generation ----------
def extract_distractors(article, correct_answer, vectorizer, top_k=3, ranker=None):
    """
    Improved distractors: category-aware + supervised ranking.
    """
    if not isinstance(article, str) or not article.strip():
        return []

    ans_low = (correct_answer or "").strip().lower()
    ans_cat = _get_category(correct_answer)
    
    # 1. Candidate pool
    np_pool = extract_noun_phrases(article)
    legacy_pool = _single_token_fallback(article, correct_answer, vectorizer, k=50)
    full_pool = list(set(np_pool + legacy_pool))
    
    # Filter substrings
    full_pool = [p for p in full_pool if p and p.lower() != ans_low and p.lower() not in ans_low and ans_low not in p.lower()]
    
    if not full_pool:
        return []

    # 2. Category matching: prioritize candidates in the same category
    candidates = []
    for p in full_pool:
        cat = _get_category(p)
        score = 1.0 if cat == ans_cat else 0.5
        candidates.append((p, score))

    # 2b. Form matching: determine answer form and filter
    ans_words = (correct_answer or "").split()
    ans_word_count = len(ans_words)
    ans_is_capitalized = any(w[0].isupper() for w in ans_words if w) if ans_words else False

    # 3. Supervised Ranking (if ranker is provided)
    if ranker:
        try:
            ohe_vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b").fit([correct_answer or ""])
            cand_vecs = ohe_vec.transform(full_pool)
            ans_vec = ohe_vec.transform([correct_answer or ""])
            sims = cosine_similarity(cand_vecs, ans_vec).ravel()
        except Exception:
            sims = np.zeros(len(full_pool))

        freqs = np.array([article.lower().count(p.lower()) for p in full_pool])
        chars = np.array([len(set(p.lower()) & set(ans_low)) / max(1, len(set(ans_low))) for p in full_pool])

        X = np.column_stack([sims, freqs, chars])
        scores = ranker.predict_proba(X)[:, 1]
        final_scores = scores * np.array([c[1] for c in candidates])
        order = np.argsort(-final_scores)
    else:
        cand_vecs = vectorizer.transform(full_pool)
        ans_vec = vectorizer.transform([correct_answer or ""])
        sims = cosine_similarity(cand_vecs, ans_vec).ravel()
        final_scores = sims * np.array([c[1] for c in candidates])
        order = np.argsort(-final_scores)

    selected = []
    for idx in order:
        if len(selected) >= top_k:
            break
        cand = full_pool[idx]
        # Form consistency: prefer same word-count range as answer
        cand_wc = len(cand.split())
        if ans_word_count == 1 and cand_wc > 2:
            continue
        if ans_word_count >= 3 and cand_wc < 2:
            continue
        # Substring/overlap filter
        if any(cand.lower() in s.lower() or s.lower() in cand.lower() for s in selected):
            continue
        # Token overlap between distractors should be low
        cand_toks = set(tokenize(cand))
        too_similar = False
        for s in selected:
            s_toks = set(tokenize(s))
            if cand_toks and s_toks:
                overlap = len(cand_toks & s_toks) / max(1, min(len(cand_toks), len(s_toks)))
                if overlap > 0.5:
                    too_similar = True
                    break
        if too_similar:
            continue
        # Match capitalization to answer
        if ans_is_capitalized and cand[0].islower():
            cand = cand[0].upper() + cand[1:]
        selected.append(cand)

    return selected


# ---------- Hint generation ----------
def _sentence_features(sents, question, correct_answer, vectorizer):
    if not sents:
        return np.empty((0, 5))
    q_vec = vectorizer.transform([question or ""])
    a_vec = vectorizer.transform([correct_answer or ""])
    sent_vecs = vectorizer.transform(sents)
    
    q_cosines = cosine_similarity(sent_vecs, q_vec).ravel()
    a_cosines = cosine_similarity(sent_vecs, a_vec).ravel()

    q_tokens = set(tokenize(question or ""))
    a_tokens = set(tokenize(correct_answer or ""))
    n = len(sents)
    feats = np.zeros((n, 5), dtype=np.float64)
    for i, s in enumerate(sents):
        s_tokens = set(tokenize(s))
        q_overlap = (len(s_tokens & q_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
        a_overlap = (len(s_tokens & a_tokens) / max(1, len(a_tokens))) if a_tokens else 0.0
        position = i / max(1, n - 1)
        length = len(s.split())
        # Features: [q_sim, a_sim, q_overlap, a_overlap, length]
        feats[i] = [q_cosines[i], a_cosines[i], q_overlap, a_overlap, length]
    return feats


def train_hint_scorer(train_csv, vectorizer, n_questions=2000, random_state=42):
    df = pd.read_csv(train_csv).sample(n_questions, random_state=random_state)
    feats_all, y_all = [], []
    for _, r in df.iterrows():
        ans = str(r[r["answer"]])
        sents = split_sentences(r["article"])
        if len(sents) < 5:
            continue
        feats = _sentence_features(sents, r["question"], ans, vectorizer)
        # Gold label: Top sentence by ANSWER similarity + QUESTION similarity
        combined = feats[:, 0] + feats[:, 1]
        rank = np.argsort(-combined)
        n_top = max(1, int(0.2 * len(sents)))
        positives = set(rank[:n_top].tolist())
        labels = np.array([1 if i in positives else 0 for i in range(len(sents))])
        feats_all.append(feats)
        y_all.append(labels)
    X = np.vstack(feats_all)
    y = np.concatenate(y_all)
    scorer = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    scorer.fit(X, y)
    return scorer


def generate_hints(article, question, correct_answer, vectorizer, hint_scorer):
    """Generate 3 graduated hints: vague → moderate → specific.

    Improvement: filter out fragments, sign-offs, and duplicates.
    Use answer-awareness to pick hints at meaningful relevance levels.
    """
    sents = split_sentences(article)
    if len(sents) < 3:
        return (sents + ["", "", ""])[:3]

    # Filter sentences: must be real sentences, not fragments
    valid_sents = []
    valid_indices = []
    for i, s in enumerate(sents):
        words = s.split()
        if len(words) < 5:
            continue
        # Must end with sentence-ending punctuation (not a fragment)
        if not s.rstrip().endswith(('.', '!', '?', '"', "'")):
            # Allow if long enough (some passages lack trailing periods)
            if len(words) < 8:
                continue
        valid_sents.append(s)
        valid_indices.append(i)

    if len(valid_sents) < 3:
        # Fallback to all sentences if filtering removed too many
        valid_sents = sents
        valid_indices = list(range(len(sents)))

    feats = _sentence_features(valid_sents, question, correct_answer, vectorizer)
    scores = hint_scorer.predict_proba(feats)[:, 1]

    # Compute answer overlap for each sentence
    ans_tokens = set(tokenize(correct_answer or ""))
    q_tokens = set(tokenize(question or ""))
    ans_overlaps = []
    for s in valid_sents:
        s_toks = set(tokenize(s))
        ov = len(s_toks & ans_tokens) / max(1, len(ans_tokens)) if ans_tokens else 0.0
        ans_overlaps.append(ov)

    # Strategy: pick 3 hints at different relevance levels
    order = np.argsort(-scores)
    n = len(order)

    picks = []
    seen = set()

    # Hint 3 (SPECIFIC): highest-scored sentence that has answer overlap
    for idx in order:
        if ans_overlaps[idx] > 0 and idx not in seen:
            picks.append(idx)
            seen.add(idx)
            break
    # If no sentence has answer overlap, pick the top-scored one
    if not picks:
        picks.append(order[0])
        seen.add(order[0])

    # Hint 2 (MODERATE): sentence with high question overlap but NOT the specific hint
    for idx in order:
        if idx in seen:
            continue
        s_toks = set(tokenize(valid_sents[idx]))
        q_overlap = len(s_toks & q_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
        if q_overlap > 0.1 or scores[idx] > np.median(scores):
            picks.append(idx)
            seen.add(idx)
            break
    if len(picks) < 2 and n >= 2:
        # Fallback: pick from the middle of the ranking
        mid = order[n // 3]
        if mid not in seen:
            picks.append(mid)
            seen.add(mid)

    # Hint 1 (VAGUE): a general topic sentence — low answer overlap, moderate score
    for idx in order[n // 3:]:
        if idx in seen:
            continue
        if ans_overlaps[idx] < 0.15:
            picks.append(idx)
            seen.add(idx)
            break
    if len(picks) < 3 and n >= 3:
        # Fallback: pick the last unseen from ranking
        for idx in order:
            if idx not in seen:
                picks.append(idx)
                seen.add(idx)
                break

    # Ensure we have exactly 3
    while len(picks) < 3:
        picks.append(picks[-1] if picks else 0)

    # Return in order: vague, moderate, specific
    return [valid_sents[picks[2]] if len(picks) > 2 else "",
            valid_sents[picks[1]] if len(picks) > 1 else "",
            valid_sents[picks[0]]]


def train_distractor_ranker(train_csv, n_questions=1000):
    """
    Step 3 compliance: train a distractor ranker.
    We'll use a heuristic: candidates that are very similar to 
    the answer but NOT the answer are good distractors.
    """
    df = pd.read_csv(train_csv).sample(min(n_questions, 500), random_state=42)
    X_all, y_all = [], []
    
    for _, r in df.iterrows():
        ans = str(r[r["answer"]])
        article = r["article"]
        # Dummy candidates for training
        np_pool = extract_noun_phrases(article)
        if not np_pool: continue
        
        try:
            ohe = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b").fit([ans])
        except ValueError:
            continue
            
        for p in np_pool[:10]:
            if p.lower() == ans.lower(): continue
            try:
                sim = cosine_similarity(ohe.transform([p]), ohe.transform([ans]))[0,0]
            except Exception: sim = 0
            freq = article.lower().count(p.lower())
            char_match = len(set(p.lower()) & set(ans.lower())) / max(1, len(set(ans.lower())))
            X_all.append([sim, freq, char_match])
            # Heuristic label: if similarity is moderate, it's a good distractor
            y_all.append(1 if 0.2 < sim < 0.8 else 0)
            
    if not X_all: return None
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_all, y_all)
    return rf


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

    print("\nTraining distractor ranker (Section 5.3.1) ...")
    d_ranker = train_distractor_ranker(train_csv)
    if d_ranker:
        joblib.dump(d_ranker, os.path.join(MODELS_DIR, "distractor_ranker.pkl"))
        print("  distractor ranker saved.")

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
        h = generate_hints(r["article"], r["question"], str(gold), vec, scorer)
        for i, hint in enumerate(h, 1):
            print(f"  Hint {i}: {hint[:140]}{'...' if len(hint) > 140 else ''}")

    print("\nDone.")


if __name__ == "__main__":
    main()
