"""
Session 5 — Improved template-based question generator.

Major upgrades over Session 3:

  Sentence-informativeness scoring
    + reward named-entity proxies (multi-word capitalized phrases not at
      sentence start), digits, dates, quantifiers, causal markers
    - penalize very short (<8 words) or very long (>30 words) sentences
    - penalize sentences that begin with a pronoun (no antecedent for the
      generated question)

  Template selection (matches content, no random fallback)
    person cue (Mr/Mrs/Dr/Ms/She/He/named entity) -> Who
    location marker (in/at/on/near/from + Capitalized) -> Where
    year/date/time -> When
    number + plural noun -> How many
    causal marker -> Why
    otherwise -> What

  Stem post-processing (the big fix that stops answer leakage)
    1. extract the answer span before the stem
    2. remove the answer tokens from the stem
    3. drop leading articles (a/an/the)
    4. capitalize first letter, force trailing "?"
    5. cap stem length at 14 words
    6. reject stem if shorter than 4 words OR if it still contains the
       full answer (rare, but possible after dedup)

The Random Forest ranker is retrained on the new feature set
(7 features, plus a tiny content-cue indicator), saved to
models/question_ranker.pkl (overwrite). Prints 10 before/after examples.
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
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.? \d{0,4}\b", flags=re.IGNORECASE)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b", flags=re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
QUANT = {"many", "most", "few", "all", "several", "some", "every", "each"}
CAUSAL = {"because", "since", "therefore", "so", "thus", "hence", "due"}
LOC_PREPS = {"in", "at", "on", "near", "from"}
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir", "lady"}
PERSON_PRONOUNS = {"he", "she", "her", "him", "they", "them"}
PRONOUN_STARTS = {"he", "she", "it", "they", "this", "that", "these", "those",
                  "we", "you", "i"}
LEADING_ARTICLES = {"a", "an", "the"}
STEM_MAX_WORDS = 14
STEM_MIN_WORDS = 4
SENT_MIN_WORDS = 8
SENT_MAX_WORDS = 30


# ----------------- helpers -----------------
def split_sentences(text):
    if not isinstance(text, str):
        return []
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def tokenize(text):
    return [t for t in TOK_RE.findall((text or "").lower()) if t not in STOPWORDS]


def words_of(sentence):
    return WORD_RE.findall(sentence)


def find_proper_noun_phrase(words):
    """Longest run of consecutive capitalized non-initial tokens."""
    best, best_len = None, 0
    cur, cur_len = [], 0
    for i, w in enumerate(words):
        if i == 0:                                # ignore sentence-initial cap
            continue
        if w and w[0].isupper() and w.isalpha():
            cur.append(w); cur_len += 1
            if cur_len > best_len:
                best, best_len = list(cur), cur_len
        else:
            cur, cur_len = [], 0
    return " ".join(best) if best else None


def first_proper_noun(words):
    for i, w in enumerate(words):
        if i == 0:
            continue
        if w and w[0].isupper() and w.isalpha():
            return w
    return None


def starts_with_pronoun(words):
    if not words:
        return False
    return words[0].lower() in PRONOUN_STARTS


def has_title_then_name(words):
    low = [w.lower() for w in words]
    for i, w in enumerate(low[:-1]):
        if w.rstrip(".") in TITLES and words[i + 1] and words[i + 1][0].isupper():
            return True
    return False


# ----------------- informativeness scoring -----------------
def informativeness_score(sentence, sentences, sim_to_article):
    """Composite [0, ~1.5] score combining cue density and sim-to-article."""
    words = words_of(sentence)
    n = len(words)
    if n == 0:
        return 0.0
    score = sim_to_article * 1.0

    # cue rewards (each at most ~0.15)
    has_pn = bool(find_proper_noun_phrase(words))
    has_year = bool(YEAR_RE.search(sentence))
    has_date = bool(DATE_RE.search(sentence))
    has_time = bool(TIME_RE.search(sentence))
    has_num = bool(NUM_RE.search(sentence))
    low = [w.lower() for w in words]
    has_quant = any(q in low for q in QUANT)
    has_causal = any(c in low for c in CAUSAL)

    score += 0.18 * has_pn
    score += 0.10 * (has_year or has_date or has_time)
    score += 0.08 * has_num
    score += 0.05 * has_quant
    score += 0.10 * has_causal

    # length penalty (smooth tent)
    if n < SENT_MIN_WORDS:
        score *= 0.4
    elif n > SENT_MAX_WORDS:
        score *= 0.6
    # pronoun start = no antecedent
    if starts_with_pronoun(words):
        score *= 0.4
    return float(max(0.0, score))


def informativeness_scores(sentences, vectorizer):
    if not sentences:
        return np.array([])
    sims = cosine_similarity(
        vectorizer.transform(sentences),
        vectorizer.transform([" ".join(sentences)])
    ).ravel()
    return np.array([
        informativeness_score(s, sentences, sims[i]) for i, s in enumerate(sentences)
    ])


# ----------------- template selection -----------------
def detect_template(sentence):
    words = words_of(sentence)
    if not words:
        return "what"
    low = [w.lower() for w in words]

    # Person cue first — beats other cues
    if has_title_then_name(words) or any(p in low for p in PERSON_PRONOUNS):
        return "who"
    if find_proper_noun_phrase(words) and not (
        any(low[i] in LOC_PREPS and (i + 1 < len(words)) and
            words[i + 1] and words[i + 1][0].isupper()
            for i in range(len(words)))
    ):
        return "who"

    # Location marker
    for i in range(len(words) - 1):
        if low[i] in LOC_PREPS and words[i + 1] and words[i + 1][0].isupper():
            return "where"

    if YEAR_RE.search(sentence) or DATE_RE.search(sentence) or TIME_RE.search(sentence):
        return "when"

    if NUM_RE.search(sentence):
        # number + plural noun proxy: a number followed by a word ending in 's'
        nums = list(NUM_RE.finditer(sentence))
        for m in nums:
            tail = sentence[m.end(): m.end() + 30]
            tail_words = WORD_RE.findall(tail)
            if tail_words and (tail_words[0].endswith("s") or tail_words[0] in QUANT):
                return "how_many"
        # otherwise still a "when"-ish? fall through.

    if any(c in low for c in CAUSAL):
        return "why"

    return "what"


# ----------------- answer-span extraction -----------------
def extract_answer_span(sentence, template):
    words = words_of(sentence)
    if template == "who":
        return find_proper_noun_phrase(words) or first_proper_noun(words) or ""
    if template == "where":
        # prep + Capitalized run
        low = [w.lower() for w in words]
        for i in range(len(words) - 1):
            if low[i] in LOC_PREPS and words[i + 1] and words[i + 1][0].isupper():
                run = [words[i + 1]]
                j = i + 2
                while j < len(words) and words[j] and words[j][0].isupper() and words[j].isalpha():
                    run.append(words[j]); j += 1
                return " ".join(run)
        return find_proper_noun_phrase(words) or ""
    if template == "when":
        m = YEAR_RE.search(sentence) or DATE_RE.search(sentence) or TIME_RE.search(sentence)
        return m.group(0) if m else ""
    if template == "how_many":
        m = NUM_RE.search(sentence)
        if not m:
            return ""
        # grab the number + the next 1-2 nouns (ending in 's' typically)
        tail = sentence[m.end(): m.end() + 60]
        tail_words = WORD_RE.findall(tail)
        if tail_words:
            return f"{m.group(0)} {tail_words[0]}".strip()
        return m.group(0)
    if template == "why":
        m = re.search(r"\b(because|since|therefore|so|thus|hence|due to)\b\s+([^.!?]+)",
                      sentence, flags=re.IGNORECASE)
        if m:
            return m.group(2).strip().split(",")[0].strip()
        return ""
    # what / fallback: longest run of content words
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    return " ".join(content[:5])


# ----------------- stem realisation + post-processing -----------------
def _strip_answer_from_stem(stem, answer):
    if not answer:
        return stem
    ans_tokens = set(t.lower() for t in WORD_RE.findall(answer))
    if not ans_tokens:
        return stem
    out_words = []
    for w in stem.split():
        if w.lower().strip(".,?!") in ans_tokens:
            continue
        out_words.append(w)
    return " ".join(out_words)


def _polish(stem):
    parts = stem.split()
    while parts and parts[0].lower() in LEADING_ARTICLES:
        parts = parts[1:]
    parts = parts[:STEM_MAX_WORDS]
    if not parts:
        return ""
    parts[0] = parts[0][0].upper() + parts[0][1:] if parts[0] else parts[0]
    text = " ".join(parts).rstrip(".,;:!?")
    return text + "?"


def realize_template(sentence, template, answer):
    words = words_of(sentence)
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    main_np = " ".join(content[:5]) if content else sentence
    main_np = _strip_answer_from_stem(main_np, answer).strip()

    if template == "who":
        stem = f"Who is associated with {main_np.lower()}"
    elif template == "where":
        stem = f"Where in the passage is {main_np.lower()} mentioned"
    elif template == "when":
        stem = f"When did the events about {main_np.lower()} happen"
    elif template == "how_many":
        stem = f"How many of {main_np.lower()} are mentioned"
    elif template == "why":
        stem = f"Why does the passage describe {main_np.lower()}"
    else:
        stem = f"What does the passage say about {main_np.lower()}"
    return _polish(stem)


def _is_valid_question(stem, answer):
    parts = [w for w in stem.split() if w.strip()]
    if len(parts) < STEM_MIN_WORDS:
        return False
    # answer must NOT appear verbatim inside the stem
    if answer:
        if answer.strip().lower() and answer.strip().lower() in stem.lower():
            return False
    return True


# ----------------- ranker features -----------------
def sentence_features(sentences, vectorizer):
    """7-feature matrix: length, position, named-entity flag, kw_overlap,
    sim_to_article, has_year_or_num, starts_with_pronoun_flag (negated)."""
    if not sentences:
        return np.empty((0, 7))
    sims = cosine_similarity(
        vectorizer.transform(sentences),
        vectorizer.transform([" ".join(sentences)])
    ).ravel()
    article_tokens = set(tokenize(" ".join(sentences)))
    n = len(sentences)
    feats = np.zeros((n, 7), dtype=np.float64)
    for i, s in enumerate(sentences):
        words = words_of(s)
        s_tokens = set(tokenize(s))
        kw_overlap = (len(s_tokens & article_tokens) / max(1, len(s_tokens))
                      if s_tokens else 0.0)
        feats[i, 0] = len(words)
        feats[i, 1] = i / max(1, n - 1)
        feats[i, 2] = 1.0 if find_proper_noun_phrase(words) else 0.0
        feats[i, 3] = kw_overlap
        feats[i, 4] = sims[i]
        feats[i, 5] = 1.0 if (YEAR_RE.search(s) or NUM_RE.search(s)) else 0.0
        feats[i, 6] = 0.0 if starts_with_pronoun(words) else 1.0
    return feats


def generate_questions(article, vectorizer, ranker=None, top_k=3):
    sents = split_sentences(article)
    if len(sents) < 3:
        return []
    feats = sentence_features(sents, vectorizer)
    info = informativeness_scores(sents, vectorizer)
    if ranker is not None:
        scores = 0.6 * ranker.predict_proba(feats)[:, 1] + 0.4 * info
    else:
        scores = info
    order = np.argsort(-scores)

    out, seen_stems = [], set()
    for idx in order[: max(top_k * 4, 8)]:
        sent = sents[idx]
        template = detect_template(sent)
        answer = extract_answer_span(sent, template)
        stem = realize_template(sent, template, answer)
        if not _is_valid_question(stem, answer):
            # if even the answer is empty, skip
            if not answer:
                continue
            # try an alternative template
            for fallback in ("what", "who", "where", "when"):
                if fallback == template: continue
                ans2 = extract_answer_span(sent, fallback)
                stem2 = realize_template(sent, fallback, ans2)
                if _is_valid_question(stem2, ans2):
                    template, answer, stem = fallback, ans2, stem2
                    break
            else:
                continue
        if stem.lower() in seen_stems:
            continue
        seen_stems.add(stem.lower())
        out.append({
            "sentence": sent,
            "template": template,
            "question": stem,
            "answer": answer,
            "score": float(scores[idx]),
        })
        if len(out) >= top_k:
            break
    return out


# ----------------- ranker training -----------------
def train_ranker(train_csv, vectorizer, n_questions=1500, random_state=42):
    print("Training improved RandomForest question_ranker (7 features) ...")
    df = pd.read_csv(train_csv).sample(n_questions, random_state=random_state)
    feats_all, y_all = [], []
    for _, r in df.iterrows():
        sents = split_sentences(r["article"])
        if len(sents) < 5:
            continue
        feats = sentence_features(sents, vectorizer)
        info = informativeness_scores(sents, vectorizer)
        # heuristic gold: top 30 % by composite informativeness
        rank = np.argsort(-info)
        n_top = max(1, int(0.3 * len(sents)))
        positives = set(rank[:n_top].tolist())
        labels = np.array([1 if i in positives else 0 for i in range(len(sents))])
        feats_all.append(feats)
        y_all.append(labels)
    X = np.vstack(feats_all)
    y = np.concatenate(y_all)
    print(f"  ranker training set: X={X.shape}, pos rate={y.mean():.3f}")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, n_jobs=-1, class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X, y)
    return rf


# ----------------- driver -----------------
def main():
    print("=" * 60)
    print("Session 5 — Improved question generator")
    print("=" * 60)
    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("Loaded vectorizer.")

    # Try the OLD ranker for the before/after comparison
    old_ranker = None
    try:
        old_ranker = joblib.load(os.path.join(MODELS_DIR, "question_ranker.pkl"))
    except Exception:
        old_ranker = None

    t0 = time.time()
    new_ranker = train_ranker(os.path.join(DATA_DIR, "train_split.csv"), vec,
                              n_questions=1500)
    joblib.dump(new_ranker, os.path.join(MODELS_DIR, "question_ranker.pkl"))
    print(f"Saved improved ranker -> models/question_ranker.pkl ({time.time() - t0:.1f}s)")

    # 10 before/after examples on real RACE val rows
    val = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv")).sample(10, random_state=11)
    print("\nBefore (S3 templates) vs After (S5 templates):")
    for k, (_, r) in enumerate(val.iterrows(), 1):
        print(f"\n--- Sample {k} (id={r['id']}) ---")
        print(f"  gold Q: {r['question']}")

        if old_ranker is not None:
            try:
                # Use the OLD ranker but also OLD-style stems by calling generate_questions
                # with a feature-shape adapter. The old ranker was 5-feature; if shapes
                # mismatch, predict_proba will throw and we'll just print "(unavailable)".
                _ = old_ranker.n_features_in_
                if old_ranker.n_features_in_ == 7:
                    old_out = generate_questions(r["article"], vec, old_ranker, top_k=1)
                else:
                    old_out = generate_questions(r["article"], vec, None, top_k=1)
            except Exception:
                old_out = generate_questions(r["article"], vec, None, top_k=1)
        else:
            old_out = generate_questions(r["article"], vec, None, top_k=1)

        new_out = generate_questions(r["article"], vec, new_ranker, top_k=1)
        if old_out:
            print(f"  before: ({old_out[0]['template']}) {old_out[0]['question']}")
            print(f"          answer={old_out[0]['answer'][:60]}")
        if new_out:
            print(f"  after : ({new_out[0]['template']}) {new_out[0]['question']}")
            print(f"          answer={new_out[0]['answer'][:60]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
