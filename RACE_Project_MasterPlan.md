# 🧠 RACE AI Project — Master Plan

### Intelligent Reading Comprehension & Quiz Generation System
>
> **Team:** 2 people | **Timeline:** 5 sessions | **UI:** React + FastAPI | **ML:** Classical only (scikit-learn) | **Features:** TF-IDF

---

## 🔍 What This Project Builds

An **AI-powered quiz system** trained on the RACE dataset (English reading passages + MCQs from Chinese school exams).

The system does 5 things:

1. Takes a reading passage as input
2. **Generates** a meaningful question from the passage *(Model A — generation)*
3. **Verifies** whether a chosen option is correct *(Model A — verification)*
4. **Generates** 3 plausible distractors + 3 graduated hints *(Model B)*
5. Exposes everything through a React UI with an analytics dashboard

---

## 🎯 Grading Rubric (drives scope)

| Component | Marks | Notes |
|-----------|------:|-------|
| EDA & Preprocessing | 10 | Visualizations, clean pipeline |
| Model A — Traditional ML (supervised) | 15 | ≥ 2 models + comparison table |
| **Model A — Unsupervised / Semi-Supervised** | **20** | **K-Means + Label Propagation** |
| Model A — Ensemble | 5 | Soft-vote / stacking |
| Model B — Distractor Generation | 15 | Precision, Recall, F1, Confusion Matrix |
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

TF-IDF (`TfidfVectorizer` from sklearn) is the chosen feature representation for this project. Cosine similarity is computed on top of TF-IDF vectors for distractor ranking and hint scoring.

**Golden rule:** `fit_transform()` on **train only**, `transform()` on val/test. Never refit. Save the fitted vectorizer with `joblib`.

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
      [React Frontend]
      Page 1: Article Input
      Page 2: Quiz View
      Page 3: Hint Panel
      Page 4: Analytics Dashboard
```

---

## 👥 Team Split

### 👤 Person 1 — Model A + Backend

| Task | Output |
|------|--------|
| EDA | `notebooks/EDA.ipynb` |
| Preprocessing + TF-IDF + cosine sim | `src/preprocessing.py` |
| Model A supervised: LR + SVM + NB | `src/model_a_train.py` |
| **Model A unsupervised: K-Means** (Silhouette + Purity) | `src/model_a_unsupervised.py` |
| **Model A semi-supervised: Label Propagation** | `src/model_a_unsupervised.py` |
| Soft-vote ensemble | `src/model_a_train.py` |
| Model A generation: Wh-templates + SVM/RF ranker | `src/model_a_generate.py` |
| Evaluation: Accuracy, Macro F1, **Exact Match**, Confusion Matrix | `src/evaluate.py` |
| FastAPI endpoints + CORS | `backend/main.py` |

### 👤 Person 2 — Model B + React UI

| Task | Output |
|------|--------|
| Distractor pipeline (frequency + TF-IDF cosine + diversity penalty) | `src/model_b.py` |
| Hint generator (sentence scoring + ML ranker) | `src/model_b.py` |
| Evaluation: Precision, Recall, F1, **R² for hint scorer** | `src/evaluate.py` |
| React + Vite setup | `frontend/` |
| Page 1: Article Input | `frontend/src/pages/ArticleInput.jsx` |
| Page 2: Quiz View | `frontend/src/pages/Quiz.jsx` |
| Page 3: Hint Panel | `frontend/src/pages/Hints.jsx` |
| Page 4: Analytics Dashboard | `frontend/src/pages/Analytics.jsx` |
| Connect React ↔ FastAPI | `fetch` calls |

---

## 📅 5-Session Plan

### Session 1 — Setup + Data (Both, ~3 hrs)

- [ ] Create folder structure
- [ ] Download dataset → `data/train.csv`
- [ ] Run 80/10/10 split, save `train_split.csv`, `val_split.csv`, `test_split.csv`
- [ ] EDA: answer distribution, article/question lengths, question types, summary stats
- [ ] `pip install fastapi uvicorn scikit-learn pandas numpy joblib matplotlib seaborn`
- [ ] `npm create vite@latest frontend -- --template react`
- [ ] Push to GitHub

**End of session:** `python src/preprocessing.py` runs and saves splits.

---

### Session 2 — Model A + Model B Core (Split, ~3 hrs)

**Person 1 (Model A — supervised + unsupervised + semi-supervised):**

- [ ] Fit `TfidfVectorizer` on train: `combined = article + " " + question + " " + option`
- [ ] Build labels: `y=1` if option==answer, else `y=0` (4 rows per question)
- [ ] Train + evaluate Logistic Regression on val (Accuracy, Macro F1, Exact Match)
- [ ] Train + evaluate SVM on val
- [ ] Train + evaluate Naive Bayes on val
- [ ] Train K-Means on TF-IDF features → report Silhouette Score + Purity
- [ ] Train Label Propagation on small labeled subset → compare F1 vs supervised
- [ ] Save vectorizer + all models with `joblib.dump()`

**Person 2 (Model B):**

- [ ] Candidate extractor: frequent content words/phrases from article
- [ ] Score candidates: TF-IDF cosine similarity vs correct answer
- [ ] Select top-3 non-answer candidates with diversity penalty → distractors
- [ ] Hint generator: split article into sentences, TF-IDF score vs question
- [ ] Train Logistic Regression hint scorer (features: keyword overlap, position, length)
- [ ] Rank → Hint 1 (vague), Hint 2 (moderate), Hint 3 (specific)

**End of session:** Both models produce output on val set.

---

### Session 3 — Ensemble + Generation + Backend + UI Skeleton (~3 hrs)

**Person 1:**

- [ ] Build soft-vote ensemble (average LR + SVM + NB probabilities)
- [ ] Build template-based question generator:
  - Extract candidate sentences via TF-IDF keyword overlap with article
  - Apply Wh-word templates (Who/What/Where/When/Why)
  - Rank generated questions with SVM/Random Forest classifier
- [ ] `backend/main.py` with 5 endpoints + CORS:
  - `POST /generate` → article in, generated question + answer out
  - `POST /predict` → verification (predicted answer + confidence)
  - `POST /distractors` → 3 ranked distractors
  - `POST /hints` → 3 graduated hints
  - `GET /analytics` → saved metrics JSON (incl. confusion matrix)
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

- [ ] Page 1 → `POST /generate` and `POST /distractors` on submit
- [ ] Page 2 → render question + options from API; `Check Answer` calls `/predict`, shows green/red
- [ ] Page 3 → `POST /hints`, populate accordions
- [ ] Page 4 → `GET /analytics`, render real metrics + confusion matrix chart
- [ ] Loading states (spinner during API calls)
- [ ] Error handling (empty input, API down, malformed response)
- [ ] **Inference latency < 10s per request** — measure and log on each call
- [ ] **Model transparency banner** in UI: "Answers are AI-generated and may be wrong"
- [ ] Accessibility: color contrast + keyboard navigation
- [ ] Test 10 random samples end-to-end

**End of session:** Full working demo.

---

### Session 5 — Final Polish (Both, ~2 hrs)

- [ ] Hyperparameter sweep on best Model A (grid/randomized search)
- [ ] Final evaluation on **test set** (held out until now): all metrics, comparison table
- [ ] Confusion matrix + per-class precision/recall figures saved to `notebooks/`
- [ ] Clean README with setup + run instructions
- [ ] Pin `requirements.txt` versions
- [ ] Run pipeline end-to-end from scratch to confirm reproducibility
- [ ] Final GitHub commit

**End of session:** Reproducible, polished demo.

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
│   ├── preprocessing.py          ← loading, cleaning, TF-IDF
│   ├── model_a_train.py          ← LR + SVM + NB + ensemble
│   ├── model_a_unsupervised.py   ← K-Means + Label Propagation
│   ├── model_a_generate.py       ← template-based question generation
│   ├── model_b.py                ← distractor + hint functions
│   └── evaluate.py               ← all metrics, confusion matrices
├── backend/
│   └── main.py                   ← FastAPI app (all endpoints)
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
    options: dict   # {"A": "...", "B": "...", "C": "...", "D": "..."}

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

---

## ⚛️ React API Call Pattern

```jsx
const checkAnswer = async () => {
  setLoading(true);
  try {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article, question, options }),
    });
    const data = await res.json();
    setResult(data);
  } catch (err) {
    console.error("API error:", err);
  } finally {
    setLoading(false);
  }
};
```

---

## ⚠️ Golden Rules

1. **No data leakage** — `fit_transform()` on train only, `transform()` on val/test
2. **No ML logic in React** — frontend only calls API and renders results
3. **Save every model with joblib** — a model not saved is a model retrained from scratch
4. **CORS must be set** — or React calls get blocked
5. **Test endpoints at `/docs` first** — confirm API works before touching React
6. **Inference < 10s per request** — rubric constraint, measure it
7. **Test set is sacred** — only touch it in Session 5 for final evaluation

---

## 📊 Required Metrics

**Model A (verification):**

- Accuracy
- Macro F1
- Exact Match (EM)
- Confusion Matrix

**Model A (unsupervised K-Means):**

- Silhouette Score
- Purity
- Comparison table vs supervised baselines

**Model A (semi-supervised Label Propagation):**

- F1 vs fully supervised LR

**Model B (distractor):**

- Precision, Recall, F1
- Distractor ranker Accuracy (top-1 is not the correct answer)
- Confusion Matrix

**Model B (hints):**

- Precision @ K (top-K hint sentence overlap with gold key sentence)
- R² Score (predicted relevance vs true relevance)

```python
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, silhouette_score, r2_score
)
```
