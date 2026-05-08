# 🧠 RACE AI Project — Master Plan

### Intelligent Reading Comprehension & Quiz Generation System
>
> **Team:** 2 people | **Timeline:** 5 sessions | **UI:** React + FastAPI | **ML:** Classical only (scikit-learn) | **Features:** TF-IDF | **Compute:** Local (RTX 3050 6GB)

---

## 🔍 What This Project Builds

An **AI-powered quiz system** trained on the RACE dataset (English reading passages + MCQs from Chinese school exams).

The system does 5 things:

1. Takes a reading passage as input
2. **Generates** a meaningful question from the passage *(Model A — generation task)*
3. **Verifies** whether a chosen option is correct *(Model A — verification task)*
4. **Generates** 3 plausible distractors + 3 graduated hints *(Model B)*
5. Exposes everything through a React UI with an analytics dashboard

---

## 🎯 Grading Rubric

| Component | Marks | Notes |
|-----------|------:|-------|
| EDA & Preprocessing | 10 | Visualizations, clean pipeline |
| Model A — Traditional ML (supervised) | 15 | ≥ 2 models + comparison table |
| **Model A — Unsupervised / Semi-Supervised** | **20** | **K-Means + Label Propagation** |
| Model A — Ensemble | 5 | Soft-vote / stacking |
| Model B — Distractor Generation | 15 | BLEU/ROUGE/METEOR + Confusion Matrix |
| Model B — Hint Generation | 10 | Graduated hints + R² for scorer |
| User Interface | 15 | All 4 screens, smooth UX, error handling |
| Final Report | 5 | Deferred — handled later |
| Code Quality | 5 | Readable, documented, clean commits |
| **TOTAL** | **100** | |

> 🔥 **The unsupervised piece (20 marks) is the largest single component. K-Means is required, not optional.**

---

## 📦 The Dataset

**Source:** <https://www.kaggle.com/datasets/ankitdhiman7/race-dataset>

Use **only `train.csv`** and split it 80/10/10:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42)
# Expected ~70k / 8.7k / 8.7k
```

**Columns:** `id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`

---

## 🧪 Feature Representation — TF-IDF

TF-IDF (`TfidfVectorizer` from sklearn) is the chosen feature representation. Cosine similarity computed on top of TF-IDF vectors for distractor ranking and hint scoring.

**Golden rule:** `fit_transform()` on **train only**, `transform()` on val/test. Save fitted vectorizer with `joblib`.

---

## ⚖️ Class Imbalance Handling (Critical — Per Instructor Clarification)

Option-level training produces a **1:3 class imbalance** (1 correct option, 3 wrong per question). Without handling, models will trivially predict "wrong" for everything and report ~75% accuracy — meaningless.

**Required mitigations:**

- Use `class_weight='balanced'` for Logistic Regression and SVM
- Use `ComplementNB` (better for imbalanced data) instead of plain `MultinomialNB`
- Always report **per-class** Precision, Recall, F1 — not just overall Accuracy
- Always show **Confusion Matrix** to verify the model isn't collapsing to one class
- Optional: `SMOTE` from `imblearn` for oversampling if class weights aren't enough

---

## 🏗️ System Architecture

```
[train.csv]
     │
     ▼
[preprocessing.py]   ← clean, TF-IDF, 80/10/10 split
     │
  ┌──┴──┐
  ▼     ▼
[Model A]                          [Model B]
- Verification: LR / SVM / NB      - Distractor pipeline (TF-IDF + cosine)
- Unsupervised: K-Means            - Hint scorer (extractive + ML ranker)
- Semi-supervised: Label Prop.
- Soft-vote ensemble
- Generation: Wh-templates + ranker
  │                                    │
  └──────────┬─────────────────────────┘
             ▼
     [FastAPI Backend]
     POST /generate       ← Model A generation
     POST /predict        ← Model A verification
     POST /distractors    ← Model B
     POST /hints          ← Model B
     GET  /analytics      ← saved metrics
             │
             ▼
      [React Frontend — 4 pages]
```

---

## 💻 Compute Setup

**Hardware:** RTX 3050 6GB is sufficient for the entire project. **No Colab/Kaggle needed.**

- All classical ML training runs on CPU (sklearn) — minutes per model
- Only the BERT/T5 **inference baseline** (Session 5) uses the GPU — fits in 6GB easily
- Use `batch_size=8` for BERT/T5 inference; drop to 4 if OOM

```bash
# GPU setup (only needed for Session 5 baseline comparison)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers
```

```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # NVIDIA GeForce RTX 3050
```

---

## 👥 Team Split

### 👤 Person 1 — Model A + Backend

| Task | Output |
|------|--------|
| EDA | `notebooks/EDA.ipynb` |
| Preprocessing + TF-IDF + cosine sim | `src/preprocessing.py` |
| Model A supervised: LR + SVM + NB (with `class_weight='balanced'`) | `src/model_a_train.py` |
| **Model A unsupervised: K-Means** (Silhouette + Purity) | `src/model_a_unsupervised.py` |
| **Model A semi-supervised: Label Propagation** | `src/model_a_unsupervised.py` |
| Soft-vote ensemble | `src/model_a_train.py` |
| Model A generation: Wh-templates + SVM/RF ranker | `src/model_a_generate.py` |
| Evaluation: full metric suite (see below) | `src/evaluate.py` |
| BERT/T5 baseline inference on test set | `src/baselines.py` |
| FastAPI endpoints + CORS | `backend/main.py` |

### 👤 Person 2 — Model B + React UI

| Task | Output |
|------|--------|
| Distractor pipeline (frequency + TF-IDF cosine + diversity penalty) | `src/model_b.py` |
| Hint generator (sentence scoring + ML ranker) | `src/model_b.py` |
| Evaluation: BLEU/ROUGE/METEOR + R² for hint scorer | `src/evaluate.py` |
| React + Vite setup | `frontend/` |
| 4 pages (Input / Quiz / Hints / Analytics) | `frontend/src/pages/*.jsx` |
| Connect React ↔ FastAPI | `fetch` calls |

---

## 📅 5-Session Plan

### ✅ Session 1 — Setup + Data (DONE)

Folder structure, dataset split, EDA, requirements.txt.

---

### Session 2 — Model A + Model B Core (Split, ~3 hrs)

**Person 1 (Model A):**

- [ ] Fit `TfidfVectorizer` on train: `combined = article + " " + question + " " + option`
- [ ] Build labels: `y=1` if option==answer, else `y=0` (4 rows per question — option-level)
- [ ] Train Logistic Regression with `class_weight='balanced'`
- [ ] Train LinearSVC with `class_weight='balanced'`
- [ ] Train ComplementNB (handles imbalance natively)
- [ ] Evaluate on val: per-class Precision/Recall/F1, Confusion Matrix, Accuracy, Macro F1, Exact Match
- [ ] Verify no model is collapsing to one class (check confusion matrix diagonal)
- [ ] Train K-Means on TF-IDF features → Silhouette Score + Purity
- [ ] Train Label Propagation on small labeled subset (e.g. 10%) → F1 vs supervised
- [ ] Save vectorizer + all models with `joblib.dump()`

**Person 2 (Model B):**

- [ ] Candidate extractor: frequent content words/phrases from article
- [ ] Score candidates: TF-IDF cosine similarity vs correct answer
- [ ] Select top-3 non-answer candidates with diversity penalty → distractors
- [ ] Hint generator: split article into sentences, TF-IDF score vs question
- [ ] Train Logistic Regression hint scorer (features: keyword overlap, position, length)
- [ ] Rank → Hint 1 (vague), Hint 2 (moderate), Hint 3 (specific)

**End of session:** Both models produce output on val set with proper metrics.

---

### Session 3 — Ensemble + Generation + Backend + UI Skeleton (~3 hrs)

**Person 1:**

- [ ] Build soft-vote ensemble (average LR + SVM + NB probabilities)
- [ ] Build template-based question generator:
  - Extract candidate sentences via TF-IDF keyword overlap
  - Apply Wh-word templates (Who/What/Where/When/Why)
  - Rank generated questions with SVM/Random Forest classifier
- [ ] `backend/main.py` with 5 endpoints + CORS
- [ ] Test all endpoints at `http://localhost:8000/docs`

**Person 2:**

- [ ] React Router setup
- [ ] Page 1: textarea + "Load Random Sample" + "Generate Quiz" buttons
- [ ] Page 2: question + 4 radio buttons + "Check Answer"
- [ ] Page 3: 3 collapsible hint accordions + "Reveal Answer"
- [ ] Page 4: metric cards + confusion matrix chart placeholder

**End of session:** Backend running; React shows all 4 pages.

---

### Session 4 — Integration + Polish (Both, ~3 hrs)

- [ ] Wire all 4 pages to backend endpoints
- [ ] Loading states + error handling
- [ ] **Inference latency < 10s per request** — measure and log
- [ ] **Model transparency banner**: "Answers are AI-generated and may be wrong"
- [ ] Accessibility: color contrast + keyboard navigation
- [ ] Test 10 random samples end-to-end

**End of session:** Full working demo.

---

### Session 5 — Baselines + Final Polish (Both, ~3 hrs)

- [ ] **BERT/T5 inference baseline** on test set (uses GPU):
  - Load `bert-base-uncased` for verification (no fine-tuning, just inference)
  - Load `t5-small` for question generation comparison
  - Compute same metrics on the same test set
  - Build comparison table: traditional ML vs neural baselines
- [ ] Hyperparameter sweep on best Model A (grid/randomized search)
- [ ] Final evaluation on **test set** (held out until now)
- [ ] Confusion matrix + per-class precision/recall figures saved to `notebooks/`
- [ ] Clean README with setup + run instructions
- [ ] Pin `requirements.txt` versions
- [ ] Run pipeline end-to-end to confirm reproducibility

**End of session:** Reproducible, polished demo with neural baseline comparison.

---

## 📁 Folder Structure

```
race_rc_project/
├── data/
│   ├── train.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── lr_classifier.pkl
│   ├── svm_classifier.pkl
│   ├── nb_classifier.pkl
│   ├── ensemble.pkl
│   ├── kmeans.pkl
│   ├── label_prop.pkl
│   └── question_ranker.pkl
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_a_unsupervised.py
│   ├── model_a_generate.py
│   ├── model_b.py
│   ├── baselines.py              ← BERT/T5 inference (Session 5)
│   └── evaluate.py
├── backend/
│   └── main.py
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── ArticleInput.jsx
│       │   ├── Quiz.jsx
│       │   ├── Hints.jsx
│       │   └── Analytics.jsx
│       ├── App.jsx
│       └── main.jsx
├── notebooks/
│   └── EDA.ipynb
├── requirements.txt
└── README.md
```

---

## ⚠️ Golden Rules

1. **No data leakage** — `fit_transform()` on train only, `transform()` on val/test
2. **Always report Confusion Matrix + per-class metrics** — not just Accuracy
3. **Use `class_weight='balanced'`** — option-level training has 1:3 imbalance
4. **No ML logic in React** — frontend only calls API
5. **Save every model with joblib** — a model not saved is a model retrained from scratch
6. **CORS must be set** — or React calls get blocked
7. **Inference < 10s per request** — rubric constraint
8. **Test set is sacred** — only touched in Session 5

---

## 📊 Required Metrics — Two Families

### Family 1: Classification Metrics (Model A verification — instructor confirmed)

- Accuracy
- Macro F1
- **Per-class Precision, Recall, F1** (critical due to imbalance)
- Exact Match (EM)
- Confusion Matrix

### Family 2: NLP Generation Metrics (instructor mandate — Models that generate text)

For Model A question generation, Model B distractor generation, and Model B hint generation:

- **BLEU Score** (`nltk.translate.bleu_score` or `sacrebleu`)
- **ROUGE Score** (`rouge-score` package — ROUGE-1, ROUGE-2, ROUGE-L)
- **METEOR Score** (`nltk.translate.meteor_score`)

### Family 3: Clustering Metrics (K-Means)

- Silhouette Score
- Purity
- Comparison table vs supervised baselines

### Family 4: Hint Scorer Regression

- R² Score (predicted relevance vs true relevance)
- Precision @ K (top-K hint sentence overlap with gold key sentence)

```python
# Classification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, silhouette_score, r2_score
)

# Generation
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
```

---

## 🔌 FastAPI Quickstart

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
classifier = joblib.load("models/ensemble.pkl")

class PredictRequest(BaseModel):
    article: str
    question: str
    options: dict

@app.post("/predict")
def predict(req: PredictRequest):
    scores = {}
    for label, text in req.options.items():
        combined = f"{req.article} {req.question} {text}"
        X = vectorizer.transform([combined])
        prob = classifier.predict_proba(X)[0][1]
        scores[label] = round(float(prob), 4)
    best = max(scores, key=scores.get)
    return {"predicted": best, "scores": scores}
```

Test at: **<http://localhost:8000/docs>**
