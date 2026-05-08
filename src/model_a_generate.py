"""
Session 3 — Template-based question generator + Random Forest ranker.

Pipeline:
  1) Sentence-split the article (regex on .!?).
  2) Score sentence informativeness via TF-IDF cosine similarity vs the
     rest of the article (high = sentence is on-topic).
  3) For each top-K sentence, apply Wh-templates:
        - capitalized non-initial token  -> "Who ...?"
        - in/at/on + capitalized          -> "Where ...?"
        - has digits / 4-digit year       -> "When ...?" or "How many ...?"
        - has because/so/therefore        -> "Why ...?"
        - default                          -> "What ...?"
  4) Train a RandomForest "informativeness ranker" on simple features
     (sentence length, position in article, has_named_entity_proxy,
     keyword_overlap_score, question_word_match) using a heuristic gold:
     top 30 % of generated questions by informativeness = positive.

Saves models/question_ranker.pkl. Prints 5 example article -> generated
questions.
"""

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

STOPWORDS = set(ENGLISH_STOP_WORDS)
TOK_RE = re.compile(r"[A-Za-z]{3,}")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z\-']*\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
NUM_RE = re.compile(r"\b\d+\b")

CAUSAL = {"because", "since", "therefore", "so", "thus", "hence"}
LOC_PREPS = {"in", "at", "on", "near", "from"}


def split_sentences(text):
    if not isinstance(text, str):
        return []
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def tokenize(text):
    return [t for t in TOK_RE.findall((text or "").lower()) if t not in STOPWORDS]


def has_capital_nonstart(words):
    """A capitalized token that isn't the first token (proxy for named entity)."""
    return any(w[0].isupper() for w in words[1:] if w and w[0].isalpha())


def find_proper_noun(words):
    """Return a capitalized non-initial token, or None."""
    for w in words[1:]:
        if w and w[0].isupper() and w.isalpha():
            return w
    return None


def detect_template(sentence):
    """Pick one of {who, where, when, how_many, why, what} based on cues."""
    words = WORD_RE.findall(sentence)
    if not words:
        return "what"
    low = [w.lower() for w in words]
    has_year = bool(YEAR_RE.search(sentence))
    has_num = bool(NUM_RE.search(sentence))
    if any(c in low for c in CAUSAL):
        return "why"
    if has_year:
        return "when"
    # location: in/at/on + Capitalized
    for i in range(len(words) - 1):
        if low[i] in LOC_PREPS and words[i + 1] and words[i + 1][0].isupper():
            return "where"
    if has_capital_nonstart(words):
        return "who"
    if has_num:
        return "how_many"
    return "what"


def realize_template(sentence, template):
    """Produce a question + an answer span (a phrase from the sentence)."""
    words = WORD_RE.findall(sentence)
    proper = find_proper_noun(words)
    year = YEAR_RE.search(sentence)
    num = NUM_RE.search(sentence)
    # Extract a "main noun phrase": longest run of content words (proxy)
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    main_np = " ".join(content[:4]) if content else sentence

    if template == "who" and proper:
        q = f"Who is mentioned in the passage in connection with {main_np.lower()}?"
        ans = proper
    elif template == "where" and proper:
        q = f"Where does the passage say {proper.lower()} is located or comes from?"
        ans = proper
    elif template == "when" and year:
        q = f"When did the events around {main_np.lower()} take place?"
        ans = year.group(0)
    elif template == "how_many" and num:
        q = f"How many are referenced in the passage about {main_np.lower()}?"
        ans = num.group(0)
    elif template == "why":
        q = f"Why does the passage say {main_np.lower()}?"
        # Take the clause after a causal cue, if any
        m = re.search(r"\b(because|since|therefore|so|thus|hence)\b\s+([^.!?]+)",
                      sentence, flags=re.IGNORECASE)
        ans = m.group(2).strip() if m else main_np
    else:  # what / fallback
        q = f"What does the passage say about {main_np.lower()}?"
        ans = main_np
    return q, ans


def informativeness_scores(sentences, vectorizer):
    """Cosine sim of each sentence against the concatenation of the rest."""
    if not sentences:
        return np.array([])
    vecs = vectorizer.transform(sentences)
    full = vectorizer.transform([" ".join(sentences)])
    sims = cosine_similarity(vecs, full).ravel()
    return sims


def sentence_features(sentences, vectorizer):
    """Per-sentence feature matrix used by the RF ranker."""
    if not sentences:
        return np.empty((0, 5))
    sims = informativeness_scores(sentences, vectorizer)
    n = len(sentences)
    feats = np.zeros((n, 5), dtype=np.float64)
    for i, s in enumerate(sentences):
        words = WORD_RE.findall(s)
        sent_tokens = set(tokenize(s))
        article_tokens = set(tokenize(" ".join(sentences)))
        kw_overlap = (len(sent_tokens & article_tokens) / max(1, len(sent_tokens))
                      if sent_tokens else 0.0)
        feats[i, 0] = len(words)
        feats[i, 1] = i / max(1, n - 1)
        feats[i, 2] = 1.0 if has_capital_nonstart(words) else 0.0
        feats[i, 3] = kw_overlap
        feats[i, 4] = sims[i]
    return feats


def generate_questions(article, vectorizer, ranker=None, top_k=3):
    sents = split_sentences(article)
    if len(sents) < 3:
        return []
    feats = sentence_features(sents, vectorizer)
    if ranker is not None:
        scores = ranker.predict_proba(feats)[:, 1]
    else:
        scores = feats[:, 4]  # fall back to informativeness
    order = np.argsort(-scores)
    out = []
    for idx in order[: max(top_k, 5)]:  # try a few extra in case templates fail
        sent = sents[idx]
        template = detect_template(sent)
        q, a = realize_template(sent, template)
        if q and a:
            out.append({"sentence": sent, "template": template,
                        "question": q, "answer": a, "score": float(scores[idx])})
        if len(out) >= top_k:
            break
    return out


def train_ranker(train_csv, vectorizer, n_questions=1000, random_state=42):
    print("Training RandomForest question_ranker ...")
    df = pd.read_csv(train_csv).sample(n_questions, random_state=random_state)
    feats_all, y_all = [], []
    for _, r in df.iterrows():
        sents = split_sentences(r["article"])
        if len(sents) < 5:
            continue
        feats = sentence_features(sents, vectorizer)
        # Heuristic gold: top 30 % by informativeness = positive
        rank = np.argsort(-feats[:, 4])
        n_top = max(1, int(0.3 * len(sents)))
        positives = set(rank[:n_top].tolist())
        labels = np.array([1 if i in positives else 0 for i in range(len(sents))])
        feats_all.append(feats)
        y_all.append(labels)
    X = np.vstack(feats_all)
    y = np.concatenate(y_all)
    print(f"  ranker training set: X={X.shape}, pos rate={y.mean():.3f}")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, n_jobs=-1, class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X, y)
    return rf


def main():
    print("=" * 60)
    print("Session 3 — Template-based question generation")
    print("=" * 60)
    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("Loaded vectorizer.")

    t0 = time.time()
    ranker = train_ranker(os.path.join(DATA_DIR, "train_split.csv"), vec,
                          n_questions=1000)
    joblib.dump(ranker, os.path.join(MODELS_DIR, "question_ranker.pkl"))
    print(f"Saved ranker -> models/question_ranker.pkl ({time.time() - t0:.1f}s)")

    # Demo on 5 val samples
    val = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv")).sample(5, random_state=11)
    print("\nDemo on 5 val samples:")
    for k, (_, r) in enumerate(val.iterrows(), 1):
        gens = generate_questions(r["article"], vec, ranker, top_k=3)
        print("\n" + "-" * 60)
        print(f"Sample {k} (id={r['id']})  gold Q: {r['question']}")
        for j, g in enumerate(gens, 1):
            print(f"  [{j}] ({g['template']}) {g['question']}")
            print(f"      answer: {g['answer'][:80]}{'...' if len(g['answer']) > 80 else ''}")
    print("\nDone.")


if __name__ == "__main__":
    main()
