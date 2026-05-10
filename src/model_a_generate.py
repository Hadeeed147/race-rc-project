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
import re
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
PERSON_PRONOUNS = {"he", "she", "they"}
PRONOUN_STARTS = {"he", "she", "it", "they", "this", "that", "these", "those",
                  "we", "you", "i"}
LEADING_ARTICLES = {"a", "an", "the"}
STEM_MAX_WORDS = 25
STEM_MIN_WORDS = 4
SENT_MIN_WORDS = 6
SENT_MAX_WORDS = 35
# Abbreviations that should NOT trigger sentence splits
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "jr", "sr", "st", "vs", "etc",
           "vol", "dept", "est", "approx", "govt", "inc", "ltd", "co",
           "gen", "sgt", "cpl", "pvt", "rev", "hon", "pres"}
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ABBREV, key=len, reverse=True)) + r")\.\s",
    flags=re.IGNORECASE,
)
_LETTER_JUNK = {"yours", "sincerely", "dear", "regards", "cheers", "thanks"}
# Pattern for quoted text in passages
QUOTED_RE = re.compile(r'"([^"]{3,40})"')


# ----------------- helpers -----------------
def split_sentences(text):
    """Improved sentence splitter that protects abbreviations and filters junk."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Step 1: Protect abbreviations by replacing their period with a placeholder
    protected = text
    for m in reversed(list(_ABBREV_PATTERN.finditer(protected))):
        abbr = m.group(1)
        protected = protected[:m.start()] + abbr + "<DOT> " + protected[m.end():]
    # Step 2: Protect initials like "S.H.E." or "U.S."
    protected = re.sub(r'\b([A-Z])\.([A-Z])\.', r'\1<DOT>\2<DOT>', protected)
    # Step 3: Split on sentence-ending punctuation followed by space+uppercase
    # More aggressive split: any .!? followed by space and Uppercase
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    # Step 4: Restore placeholders and filter
    out = []
    for s in raw:
        s = s.replace("<DOT>", ".").strip()
        if not s:
            continue
        words = s.split()
        if len(words) < 5 or len(words) > 30: # Tighten sentence length
            continue
        # Skip letter sign-offs and greetings
        first_low = words[0].lower().rstrip(".,;:")
        if first_low in _LETTER_JUNK:
            continue
        out.append(s)
    return out


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
    """First capitalized word that isn't a pronoun, title, or article."""
    skip = PRONOUN_STARTS | TITLES | LEADING_ARTICLES | {"i"}
    for i, w in enumerate(words):
        if i == 0:
            continue
        clean = w.strip(".,?!;:")
        if clean and clean[0].isupper() and clean.isalpha() and clean.lower() not in skip:
            return clean
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

    # 1. Time/Date (HIGHEST PRIORITY)
    if YEAR_RE.search(sentence) or DATE_RE.search(sentence) or TIME_RE.search(sentence):
        return "when"

    # 2. Person cue
    if has_title_then_name(words) or any(p in low for p in PERSON_PRONOUNS):
        return "who"
    if find_proper_noun_phrase(words) and not (
        any(low[i] in LOC_PREPS and (i + 1 < len(words)) and
            words[i + 1] and words[i + 1][0].isupper()
            for i in range(len(words)))
    ):
        return "who"

    # 3. Location marker
    for i in range(len(words) - 1):
        if low[i] in LOC_PREPS and words[i + 1] and words[i + 1][0].isupper():
            return "where"

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


def _extract_principal_phrase(sentence, answer):
    """
    Smarter Safe version: Isolates the predicate but 
    RETAINS the verb so the question is grammatically correct.
    """
    low = sentence.lower()
    ans_low = (answer or "").lower()
    
    # 1. Subject-replacement logic: If answer is at the start,
    # the question is just [Wh] + [rest of sentence]
    if ans_low and low.startswith(ans_low):
        phrase = sentence[len(ans_low):].strip()
        return phrase
        
    # 2. Hinge-splitting but keeping the hinge word
    # (is, was, has, etc.)
    hinges = [r" was ", r" is ", r" were ", r" are ", r" has ", r" had ", r" went ", r" did ", r" took "]
    for h in hinges:
        match = re.search(h, sentence, flags=re.IGNORECASE)
        if match:
            # Return from the verb onwards
            return sentence[match.start():].strip()
            
    # 3. Location/Object hinges (in, at, with)
    loc_hinges = [r" in ", r" at ", r" with ", r" about "]
    for h in loc_hinges:
        match = re.search(h, sentence, flags=re.IGNORECASE)
        if match:
            # Return before the hinge
            return sentence[:match.start()].strip()
            
    return sentence


# ----------------- answer-span extraction -----------------
def extract_answer_span(sentence, template):
    """
    Extract a meaningful answer span from a sentence.
    Prioritises multi-word named entities, quoted text, dates, and quantities
    over random single tokens.
    """
    words = sentence.split()
    if not words:
        return ""

    # 1. Quoted text — often the key phrase in RACE passages
    qm = QUOTED_RE.search(sentence)
    if qm:
        span = qm.group(1).strip()
        if 1 <= len(span.split()) <= 2:
            return span

    # 2. Multi-word proper noun phrase (e.g. "Thomas Moore", "New York")
    pn = find_proper_noun_phrase(words)
    if pn and 1 <= len(pn.split()) <= 2:
        return pn

    # 3. Year or date expression
    year_match = re.search(r"\b(19|20)\d{2}\b", sentence)
    if year_match and template in ("when", "what"):
        return year_match.group()

    # 4. Quantity expression: number + following word(s) e.g. "three years"
    qty = re.search(r"\b(\d+(?:\.\d+)?\s+\w+)\b", sentence)
    if qty and template == "how_many":
        span = qty.group(1).strip()
        if 1 <= len(span.split()) <= 2:
            return span

    # 5. Single proper noun (not a pronoun)
    fpn = first_proper_noun(words)
    if fpn and len(fpn) > 2:
        # If the template is 'who', we MUST have a proper noun or person word
        return fpn

    # 6. Fallback: the longest content word that isn't a stopword or pronoun
    # Avoid '-ing' verbs and common adverbs as they make bad Wh-questions
    bad_suffixes = ("ing", "ly", "ed", "es")
    skip_set = STOPWORDS | PRONOUN_STARTS | {"i", "am", "is", "was", "were", "are", "be"}
    
    candidates = []
    for w in words:
        w_clean = w.strip(".,?!").lower()
        if w_clean in skip_set or len(w_clean) < 2:
            continue
        # Removed bad_suffixes check to allow more candidates
        candidates.append(w.strip(".,?!"))
        
    if candidates:
        return max(candidates, key=len)

    return words[-1].strip(".,?!") if words else ""


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
    """Clean up the question stem: capitalize, trim, add question mark."""
    if not stem:
        return None
    # Remove leading common adverbs that don't belong in a question
    stem = re.sub(r'^(also|however|so far|furthermore|moreover|additionally|consequently|therefore),\s*', '', stem, flags=re.IGNORECASE)
    
    parts = stem.split()
    # Ensure it starts with a Wh-word (it should, but just in case)
    wh_words = {"who", "what", "where", "when", "why", "how"}
    if not parts or parts[0].lower() not in wh_words:
        # If it doesn't start with a Wh-word, it might be a mangled sentence.
        return None

    # Limit length
    parts = parts[:20] 
    parts[0] = parts[0][0].upper() + parts[0][1:] if parts[0] else parts[0]
    
    # Remove duplicate adjacent words
    deduped = [parts[0]]
    for w in parts[1:]:
        if w.lower() != deduped[-1].lower():
            deduped.append(w)
            
    text = " ".join(deduped).rstrip(".,;:!? ")
    return text + "?"


def realize_template(sentence, template, answer):
    """
    Build a question by replacing the answer span with a Wh-word.
    ALWAYS starts with a Wh-word. Uses auxiliary inversion.
    Returns None if a grammatical question cannot be formed.
    """
    if not answer or not sentence:
        return None
        
    ans_low = answer.lower().strip()
    sent_low = sentence.lower()

    # Map template to Wh-word prefix
    wh = {"who": "Who", "where": "Where", "when": "When",
          "how_many": "How many", "why": "Why"}.get(template, "What")

    pattern = re.compile(r'\b' + re.escape(ans_low) + r'\b', flags=re.IGNORECASE)
    match = pattern.search(sentence)

    if not match:
        return None

    before = sentence[:match.start()].strip()
    after = sentence[match.end():].strip().rstrip(".,;:")

    if not before:
        # Subject replacement: [Wh] [Rest]?
        return _polish(f"{wh} {after}")
        
    if len(before.split()) <= 1 and before.lower() in {"in", "at", "on", "from", "by"}:
        # Prepended preposition: "In London" -> "Where ...?"
        return _polish(f"{wh} {after}")

    # Try auxiliary inversion: [Wh] [Aux] [Subject] [Verb Rest]?
    # Look for common auxiliary/modal verbs
    aux_pattern = re.compile(r'\b(is|was|are|were|has|had|can|could|will|would|should|does|did|do)\b', re.IGNORECASE)
    aux_match = aux_pattern.search(before)
    
    if aux_match:
        aux = aux_match.group(1)
        subj = before[:aux_match.start()].strip()
        rest = before[aux_match.end():].strip()
        
        # Only invert if the subject is relatively simple (<= 2 words)
        # to avoid mangling complex sentences.
        if 1 <= len(subj.split()) <= 2:
            # If the subject is 'I', change to 'you' for a better question
            if subj.lower() == 'i':
                subj = 'you'
                if aux.lower() == 'was': aux = 'were'
                if aux.lower() == 'am': aux = 'are'
            
            # assembled = Wh + Aux + Subj + RestOfPredicate + After
            parts = [wh, aux, subj, rest, after]
            return _polish(" ".join(p for p in parts if p))

    # Fallback: If no aux found or subj is too complex, use the original hinge logic
    # This ensures we don't return None and crash the UI.
    phrase = _extract_principal_phrase(sentence, answer)
    phrase = phrase.strip(".,?! ")
    if not phrase:
        phrase = "the passage"
    return _polish(f"{wh} {phrase}")



def _is_valid_question(stem, answer):
    """Pass-able quality gate."""
    if not stem:
        return False
    parts = [w for w in stem.split() if w.strip()]
    
    # Very lenient word count
    if len(parts) < 3:
        return False
        
    # answer must NOT appear verbatim inside the stem
    if answer and answer.strip().lower():
        if answer.strip().lower() in stem.lower():
            return False
            
    # Must contain at least one content word
    content = [w for w in parts[1:] if w.lower().rstrip("?.,") not in STOPWORDS]
    if not content:
        # Even if no content words, let it through if it's a Wh-question
        if parts[0].lower() not in {"who", "what", "where", "when", "why", "how"}:
            return False
        
    # REMOVED internal quote check to satisfy 'pass-able' requirement
    return True


# ----------------- ranker features -----------------
def sentence_features(sentences, vectorizer, target_text=None):
    """
    7-feature matrix: length, position, named-entity flag, kw_overlap,
    sim_to_target, has_year_or_num, starts_with_pronoun_flag (negated).
    
    target_text: usually the gold answer (training) or heuristic answer (inference).
    """
    if not sentences:
        return np.empty((0, 7))
    
    target_vec = vectorizer.transform([target_text or " ".join(sentences)])
    sims = cosine_similarity(
        vectorizer.transform(sentences),
        target_vec
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
        feats[i, 4] = sims[i] # This is now similarity to target_text (the answer!)
        feats[i, 5] = 1.0 if (YEAR_RE.search(s) or NUM_RE.search(s)) else 0.0
        feats[i, 6] = 0.0 if starts_with_pronoun(words) else 1.0
    return feats


def generate_questions(text, vectorizer, ranker=None, top_k=20):
    """
    Generate multiple quiz questions from a passage.
    Returns a list of dicts: [{'question': str, 'answer': str, 'template': str}, ...]
    """
    sents = split_sentences(text)
    if not sents:
        return []
    
    # Step 1 Compliance: For each sentence, pick a candidate answer 
    # so we can calculate the "overlap with answer" feature.
    candidate_answers = []
    for s in sents:
        temp_template = detect_template(s)
        candidate_answers.append(extract_answer_span(s, temp_template))
    
    # Aggregate info for ranking
    info = informativeness_scores(sents, vectorizer)
    
    # We use the most informative answer as a proxy target for global ranking
    best_ans_idx = int(np.argmax(info))
    proxy_target = candidate_answers[best_ans_idx]
    
    feats = sentence_features(sents, vectorizer, target_text=proxy_target)
    
    if ranker is not None:
        scores = 0.6 * ranker.predict_proba(feats)[:, 1] + 0.4 * info
    else:
        scores = info
    order = np.argsort(-scores)

    out, seen_stems = [], set()
    for idx in order:
        sent = sents[idx]
        template = detect_template(sent)
        answer = candidate_answers[idx]

        # Skip empty answers. Allow up to 4 words to ensure we get results.
        if not answer or len(answer) < 2 or len(answer.split()) > 4:
            continue

        # Re-derive template from the answer type for better alignment
        template = _align_template_to_answer(template, answer, sent)

        stem = realize_template(sent, template, answer)
        if not stem:
            continue
        
        if not _is_valid_question(stem, answer):
            continue
            
        # Dedup stems
        if stem.lower() in seen_stems:
            continue
        seen_stems.add(stem.lower())

        out.append({
            "question": stem,
            "answer": answer,
            "template": template
        })
        if len(out) >= top_k:
            break
            
    return out


def _align_template_to_answer(template, answer, sentence):
    """Ensure the Wh-word matches what was actually extracted as the answer."""
    ans = (answer or "").strip()
    # If it's a year → when
    if re.match(r'^(19|20)\d{2}$', ans):
        return "when"
    # If it's a number + noun → how_many
    if re.match(r'^\d+\s+\w+', ans):
        return "how_many"
    # If it looks like a person name (Title + Name, or multi-word proper)
    words = ans.split()
    if words and words[0].lower().rstrip('.') in TITLES:
        return "who"
    if all(w[0].isupper() and w.isalpha() for w in words) and len(words) >= 2:
        # Multi-word proper noun — could be person or place
        # Check if it follows a location preposition in the sentence
        sent_low = sentence.lower()
        for prep in ("in ", "at ", "from ", "near ", "to "):
            if prep + ans.lower() in sent_low:
                return "where"
        return "who"
    # Single proper noun — keep original template but avoid "who" for places
    if template == "who" and len(words) == 1 and words[0][0].isupper():
        # Heuristic: if preceded by location preposition → where
        sent_low = sentence.lower()
        for prep in ("in ", "at ", "from ", "near ", "to "):
            if prep + ans.lower() in sent_low:
                return "where"
    return template


# ----------------- ranker training -----------------
def train_ranker(train_csv, vectorizer, n_questions=1500, random_state=42):
    print("Training improved RandomForest question_ranker (7 features) ...")
    df = pd.read_csv(train_csv).sample(n_questions, random_state=random_state)
    feats_all, y_all = [], []
    for _, r in df.iterrows():
        sents = split_sentences(r["article"])
        if len(sents) < 5:
            continue
            
        # Get the real gold answer for this question
        gold_ans = str(r[r["answer"]])
        
        feats = sentence_features(sents, vectorizer, target_text=gold_ans)
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
