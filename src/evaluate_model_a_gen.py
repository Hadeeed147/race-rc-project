"""
Session 6 — Evaluation of Model A Question Generation.
Requirement: Section 4.2.2 (Generation Metrics: BLEU, ROUGE, METEOR).
Compares generated questions against the ground-truth RACE questions.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from model_a_generate import generate_questions, tokenize  # noqa: E402

ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def evaluate_question_generation(samples=200):
    print(f"Evaluating Model A Question Generation ({samples} samples) ...")
    
    # Ensure NLTK data
    for pkg in ("wordnet", "omw-1.4", "punkt_tab"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg in ("wordnet", "omw-1.4") else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    ranker = joblib.load(os.path.join(MODELS_DIR, "question_ranker.pkl"))
    
    df = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv")).sample(samples, random_state=42)
    
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    refs, hyps = [], []
    rouge_l_scores, meteor_scores = [], []
    n_skipped = 0
    
    for idx, row in df.iterrows():
        article = row["article"]
        gold_q = row["question"]
        
        # Generate questions
        candidates = generate_questions(article, vec, ranker, top_k=1)
        if not candidates:
            n_skipped += 1
            continue
            
        gen_q = candidates[0]["question"]
        
        # BLEU
        refs.append([tokenize(gold_q)])
        hyps.append(tokenize(gen_q))
        
        # ROUGE
        sc = rouge.score(gold_q, gen_q)
        rouge_l_scores.append(sc["rougeL"].fmeasure)
        
        # METEOR
        try:
            meteor_scores.append(float(meteor_score([tokenize(gold_q)], tokenize(gen_q))))
        except Exception:
            pass
            
    bleu = float(corpus_bleu(refs, hyps, smoothing_function=smooth)) if hyps else 0.0
    
    results = {
        "n_samples": samples,
        "n_generated": samples - n_skipped,
        "bleu": bleu,
        "rouge_l": np.mean(rouge_l_scores) if rouge_l_scores else 0.0,
        "meteor": np.mean(meteor_scores) if meteor_scores else 0.0
    }
    
    # Save
    out_path = os.path.join(MODELS_DIR, "model_a_gen_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}")
        
    return results

if __name__ == "__main__":
    evaluate_question_generation()
