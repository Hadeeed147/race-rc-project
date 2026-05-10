"""Quick quality check — see what the system actually produces."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import joblib
import pandas as pd
from model_a_generate import generate_questions
from model_b import extract_distractors, generate_hints, split_sentences

vec = joblib.load('models/tfidf_vectorizer.pkl')
ranker = joblib.load('models/question_ranker.pkl')
scorer = joblib.load('models/hint_scorer.pkl')

val = pd.read_csv('data/val_split.csv').sample(5, random_state=99)
for k, (_, r) in enumerate(val.iterrows(), 1):
    print(f"\n{'='*70}")
    print(f"SAMPLE {k}  (id={r['id']})")
    print(f"{'='*70}")
    gold_q = r['question']
    gold_ans_label = r['answer']
    gold_ans_text = str(r[gold_ans_label])
    print(f"GOLD Q: {gold_q}")
    print(f"GOLD A: ({gold_ans_label}) {gold_ans_text}")
    print(f"GOLD OPTIONS: A={r['A'][:50]}  B={r['B'][:50]}  C={r['C'][:50]}  D={r['D'][:50]}")
    print()

    qs = generate_questions(r['article'], vec, ranker, top_k=1)
    if qs:
        q = qs[0]
        print(f"GEN Q:  {q['question']}")
        print(f"GEN A:  {q['answer']}")
        print(f"TEMPLATE: {q['template']}")
        print()

        ds = extract_distractors(r['article'], q['answer'], vec, top_k=3)
        print(f"DISTRACTORS: {ds}")
        print()

        hints = generate_hints(r['article'], q['question'], q['answer'], vec, scorer)
        for i, h in enumerate(hints, 1):
            print(f"HINT {i}: {h[:150]}")
    else:
        print("NO QUESTION GENERATED")
    print()
