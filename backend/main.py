"""
Session 3 — FastAPI backend for the RACE quiz system.

Endpoints (all under http://localhost:8000):
  POST /generate     — article in, generated question + answer + 4 options
  POST /predict      — article + question + 4 options in, ensemble prediction
  POST /distractors  — article + correct_answer in, 3 distractors out
  POST /hints        — article + question in, 3 graduated hints out
  GET  /analytics    — saved val-set metrics (Model A + Model B)

CORS is open to http://localhost:5173 (Vite default) and http://localhost:3000.

All models are loaded ONCE at startup. Every response includes latency_ms.
"""

import os
import sys
import json
import time
import random
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make src/ importable so we can re-use Model B + question generator helpers
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_b import extract_distractors, generate_hints           # noqa: E402
from model_a_generate import generate_questions                    # noqa: E402

MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# -------------- Pydantic schemas --------------
class GenerateReq(BaseModel):
    article: str = Field(..., min_length=20)


class PredictReq(BaseModel):
    article: str
    question: str
    options: Dict[str, str]   # keys: A,B,C,D


class DistractorReq(BaseModel):
    article: str
    correct_answer: str


class HintReq(BaseModel):
    article: str
    question: str


# -------------- App + state --------------
app = FastAPI(title="RACE RC API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, object] = {}


@app.on_event("startup")
def _load_models():
    print("Loading models ...")
    STATE["vectorizer"] = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    bundle = joblib.load(os.path.join(MODELS_DIR, "ensemble.pkl"))
    STATE["ensemble_models"] = bundle["models"]
    STATE["ensemble_weights"] = bundle["weights"]
    STATE["hint_scorer"] = joblib.load(os.path.join(MODELS_DIR, "hint_scorer.pkl"))
    try:
        STATE["question_ranker"] = joblib.load(os.path.join(MODELS_DIR, "question_ranker.pkl"))
    except Exception:
        STATE["question_ranker"] = None
    val_csv = os.path.join(DATA_DIR, "val_split.csv")
    if os.path.exists(val_csv):
        STATE["val_df"] = pd.read_csv(val_csv)
        print(f"Loaded val sampling pool: {len(STATE['val_df'])} rows.")
    else:
        STATE["val_df"] = None
    print("Models loaded.")


# -------------- Helpers --------------
def _ensemble_proba(article: str, question: str, options: Dict[str, str]):
    """Run the soft-vote ensemble on the 4 options of one question."""
    vec = STATE["vectorizer"]
    models = STATE["ensemble_models"]
    weights = STATE["ensemble_weights"]
    labels = ["A", "B", "C", "D"]
    texts = [f"{article} {question} {options.get(L, '')}" for L in labels]
    X = vec.transform(texts)
    probs = np.zeros(len(labels))
    for m, w in zip(models, weights):
        probs += w * m.predict_proba(X)[:, 1]
    return probs


# -------------- Endpoints --------------
@app.get("/")
def root():
    return {"service": "race-rc-api", "endpoints":
            ["/generate", "/predict", "/distractors", "/hints",
             "/analytics", "/sample", "/healthz"]}


@app.get("/healthz")
def healthz():
    return {"ok": True, "models_loaded": "vectorizer" in STATE}


@app.get("/sample")
def sample():
    """Return a random RACE val row — article, question, answer, options."""
    t0 = time.time()
    df = STATE.get("val_df")
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="val pool not loaded.")
    row = df.sample(1).iloc[0]
    return {
        "id": str(row["id"]),
        "article": str(row["article"]),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "options": {L: str(row[L]) for L in ["A", "B", "C", "D"]},
        "latency_ms": int((time.time() - t0) * 1000),
    }


@app.post("/generate")
def generate(req: GenerateReq):
    t0 = time.time()
    vec = STATE["vectorizer"]
    ranker = STATE.get("question_ranker")
    gens = generate_questions(req.article, vec, ranker, top_k=1)
    if not gens:
        raise HTTPException(status_code=422, detail="Could not generate a question from this article.")
    g = gens[0]
    correct_answer = g["answer"]
    distractors = extract_distractors(req.article, correct_answer, vec, top_k=3)
    while len(distractors) < 3:
        distractors.append(f"option {len(distractors) + 1}")
    # Place correct answer at a random A/B/C/D slot
    rng = random.Random()
    correct_label = rng.choice(["A", "B", "C", "D"])
    options: Dict[str, str] = {}
    di = 0
    for L in ["A", "B", "C", "D"]:
        if L == correct_label:
            options[L] = correct_answer
        else:
            options[L] = distractors[di]
            di += 1
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "question": g["question"],
        "answer": correct_label,
        "answer_text": correct_answer,
        "options": options,
        "template": g["template"],
        "latency_ms": latency_ms,
    }


@app.post("/predict")
def predict(req: PredictReq):
    t0 = time.time()
    if set(req.options.keys()) != {"A", "B", "C", "D"}:
        raise HTTPException(status_code=400, detail="options must have keys A,B,C,D.")
    probs = _ensemble_proba(req.article, req.question, req.options)
    labels = ["A", "B", "C", "D"]
    pred_idx = int(np.argmax(probs))
    scores = {L: float(round(p, 4)) for L, p in zip(labels, probs)}
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "predicted": labels[pred_idx],
        "scores": scores,
        "model_used": "ensemble (LR + SVC + NB, equal weights, per-question argmax)",
        "latency_ms": latency_ms,
    }


@app.post("/distractors")
def distractors(req: DistractorReq):
    t0 = time.time()
    vec = STATE["vectorizer"]
    ds = extract_distractors(req.article, req.correct_answer, vec, top_k=3)
    latency_ms = int((time.time() - t0) * 1000)
    return {"distractors": ds, "latency_ms": latency_ms}


@app.post("/hints")
def hints(req: HintReq):
    t0 = time.time()
    vec = STATE["vectorizer"]
    sc = STATE["hint_scorer"]
    hs = generate_hints(req.article, req.question, vec, sc)
    levels = ["vague", "moderate", "specific"]
    out = [{"level": lv, "text": t} for lv, t in zip(levels, hs)]
    latency_ms = int((time.time() - t0) * 1000)
    return {"hints": out, "latency_ms": latency_ms}


@app.get("/analytics")
def analytics():
    """Return saved val-set metrics (Model A + Model B)."""
    metrics_file = os.path.join(os.path.dirname(__file__), "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            return json.load(f)
    # Fallback: build from the per-component JSONs in models/
    out: Dict[str, object] = {}
    a_path = os.path.join(MODELS_DIR, "model_a_metrics.json")
    b_path = os.path.join(MODELS_DIR, "model_b_metrics.json")
    if os.path.exists(a_path):
        with open(a_path) as f:
            out["model_a"] = json.load(f)
    if os.path.exists(b_path):
        with open(b_path) as f:
            out["model_b"] = json.load(f)
    return out


# Run with: python -m uvicorn backend.main:app --reload --port 8000
