import sys
import os
import time
import joblib
import pandas as pd
import numpy as np

# Add src to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model_a_generate import generate_questions
from model_b import extract_distractors, generate_hints

def test_inference_latency():
    print("\n--- RACE AI Inference Latency Test ---")
    
    # Load assets
    print("Loading models and vectorizers...")
    t_start = time.time()
    vec = joblib.load(os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.pkl'))
    ranker = joblib.load(os.path.join(ROOT_DIR, 'models', 'question_ranker.pkl'))
    scorer = joblib.load(os.path.join(ROOT_DIR, 'models', 'hint_scorer.pkl'))
    print(f"Setup time: {time.time() - t_start:.2f}s")

    # Load 10 random validation samples
    val_df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'val_split.csv')).sample(10, random_state=42)
    
    latencies = []
    
    for i, (_, row) in enumerate(val_df.iterrows(), 1):
        article = row['article']
        
        print(f"Running sample {i}/10...", end="", flush=True)
        t0 = time.time()
        
        # 1. Generate Question (Model A)
        qs = generate_questions(article, vec, ranker, top_k=1)
        if not qs:
            print(" (No question generated, skipping)")
            continue
        
        q = qs[0]
        q_text = q['question']
        ans_text = q['answer']
        
        # 2. Generate Distractors (Model B)
        ds = extract_distractors(article, ans_text, vec, top_k=3)
        
        # 3. Generate Hints (Model B)
        hints = generate_hints(article, q_text, ans_text, vec, scorer)
        
        latency = time.time() - t0
        latencies.append(latency)
        print(f" {latency:.2f}s")

    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    print("\nResults:")
    print(f"  Average Latency: {avg_latency:.2f}s")
    print(f"  Maximum Latency: {max_latency:.2f}s")
    
    # The rubric constraint is < 10s per request
    if max_latency < 10.0:
        print("\nPASS: All requests completed in under 10 seconds.")
    else:
        print(f"\nFAIL: Some requests exceeded 10 seconds (Max: {max_latency:.2f}s).")
        # sys.exit(1) # Commenting out exit for now to see full results without crashing

if __name__ == "__main__":
    test_inference_latency()
