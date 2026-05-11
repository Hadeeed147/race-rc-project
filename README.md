# RACE Reading Comprehension AI

A **classical-ML** reading-comprehension and quiz system over the
[RACE dataset](https://www.kaggle.com/datasets/ankitdhiman7/race-dataset)
(English exam passages with multiple-choice questions). Given a passage,
the system generates a multiple-choice question, scores each option with
a soft-vote ensemble of TF-IDF classifiers, surfaces three graduated hints,
and exposes everything through a polished React + FastAPI UI.

> **No neural networks in the production pipeline** — BERT and T5 appear
> only as inference-only baselines for comparison.

---

## Features

| Component | Description |
|-----------|-------------|
| **Model A — Verification** | Soft-vote ensemble (LR + calibrated LinearSVC + ComplementNB) over 20 000-feature TF-IDF vectors; `lr_heavy` weights (0.50 / 0.25 / 0.25) found by val-set search |
| **Model A — Generation** | Wh-template question generator with a 7-feature RandomForest ranker and multi-stage stem post-processor (answer-leakage guard, length cap, forced `?`) |
| **Model B — Distractors** | Noun-phrase extraction + category-aware scoring + supervised RandomForest reranker (One-Hot cosine, char-level match, passage frequency) + diversity filter |
| **Model B — Hints** | LR scorer over 5 sentence-level features (cosine sim, keyword overlap, position, length); returns 3 hints at ~70 / 85 / 95th relevance percentiles |
| **Unsupervised** | K-Means (k=4, silhouette + purity) on TF-IDF features |
| **Semi-supervised** | Label Propagation (knn=7) over TruncatedSVD-100 projection; compared against supervised LR on same label budget |

---

## Results (held-out test set, 8 787 questions)

### Model A — Verification

| Model | Accuracy | Macro F1 | Exact Match |
|-------|:--------:|:--------:|:-----------:|
| Random baseline | — | — | 0.250 |
| Frozen BERT-base + LR head | 0.6345 | 0.4895 | 0.2359 |
| Logistic Regression (individual) | 0.6710 | 0.5408 | 0.3082 |
| LinearSVC — calibrated (individual) | 0.6690 | 0.5381 | 0.3040 |
| ComplementNB (individual) | 0.6629 | 0.5296 | 0.2948 |
| **Tuned Ensemble (lr_heavy)** | **0.6686** | **0.5372** | **0.3083** |

### Model B — Distractors (200-sample test subset)

| BLEU | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | METEOR | Token F1 |
|:----:|:---------:|:---------:|:---------:|:------:|:--------:|
| 0.0042 | 0.1056 | 0.0158 | 0.1036 | 0.0757 | 0.0961 |

### Model B — Hints (186-sample val subset)

| Precision @ 1 | Precision @ 3 | R² (scorer) |
|:-------------:|:-------------:|:-----------:|
| 0.57 | 0.74 | −2.16 |

> **Note on Hint R²:** The scorer is trained on question-relevance but
> evaluated against answer-relevance gold. P@3 = 0.74 is the more
> meaningful metric. See report for full discussion.

---

## Architecture

```
                        data/train.csv
                              │  80 / 10 / 10  random_state=42
                              ▼
                  ┌───────────────────────┐
                  │  src/preprocessing.py │  split → train/val/test CSVs
                  │  src/model_a_train.py │  TF-IDF fit + LR/SVC/NB train
                  └──────────┬────────────┘
                             │
           ┌─────────────────┴──────────────────┐
           ▼                                    ▼
  ┌─────────────────────┐         ┌──────────────────────────┐
  │ Model A             │         │ Model B                  │
  │  LR · SVC · NB      │         │  noun-phrase distractors │
  │  Soft-vote ensemble │         │  LR hint scorer          │
  │  K-Means            │         │  distractor RF ranker    │
  │  Label Propagation  │         └──────────────┬───────────┘
  │  Wh-template gen.   │                        │
  └─────────┬───────────┘                        │
            └──────────────────┐  ┌──────────────┘
                               ▼  ▼
                   ┌─────────────────────────┐
                   │  backend/main.py        │
                   │  FastAPI · 8 endpoints  │
                   └────────────┬────────────┘
                                │ JSON / CORS
                                ▼
                   ┌─────────────────────────┐
                   │  frontend/ (Vite/React) │
                   │  Tailwind · Recharts    │
                   │  Framer Motion · Zustand│
                   └─────────────────────────┘
```

---

## Quick Start (from Submission)

If you have downloaded the submission folder (all model `.pkl` files and `data/val_split.csv` are pre-bundled — **no retraining required**):

### Requirements

- **Python 3.11+** (tested on 3.13.7)
- **Node 18+** (tested on 20)

### 1. Create virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Start the backend

```powershell
# From the project root
python -m uvicorn backend.main:app --reload --port 8000
```

API explorer: **http://localhost:8000/docs**

### 3. Start the frontend (new terminal)

```powershell
cd frontend
npm install
npm run dev
```

App: **http://localhost:5173**

---

## Setup (Full — Train from Scratch)

### Requirements

- **Python 3.11+** (tested on 3.13.7)
- **Node 18+** (tested on 20)
- **RAM:** 8 GB minimum (16 GB recommended for the full hyperparameter sweep)
- **GPU:** Optional — needed only for `src/baselines.py` (BERT + T5)

### 1. Clone and create virtual environment

```powershell
git clone https://github.com/Hadeeed147/race-rc-project.git
cd race-rc-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Place the dataset

Download `train.csv` from the
[RACE Kaggle dataset](https://www.kaggle.com/datasets/ankitdhiman7/race-dataset)
and place it at:

```
data/train.csv
```

### 3. Build splits and train all models (~25 min on CPU)

Run these **in order** from the project root with the venv active:

```powershell
# Split into train / val / test CSVs + option-level rows
.\.venv\Scripts\python.exe src/preprocessing.py

# Fit TF-IDF vectorizer + train LR, SVC, NB + save base models
.\.venv\Scripts\python.exe src/model_a_train.py

# K-Means clustering + Label Propagation (unsupervised / semi-supervised)
.\.venv\Scripts\python.exe src/model_a_unsupervised.py

# Wh-template question generator + RandomForest ranker
.\.venv\Scripts\python.exe src/model_a_generate.py

# Distractor pipeline + Hint scorer training
.\.venv\Scripts\python.exe src/model_b.py

# Hyperparameter sweep (GridSearchCV) + ensemble weight search
.\.venv\Scripts\python.exe src/hyperparameter_sweep.py

# Final test-set evaluation + publication figures
.\.venv\Scripts\python.exe src/evaluate.py

# Refresh backend/metrics.json served by GET /analytics
.\.venv\Scripts\python.exe src/build_metrics_json.py
```

### 4. (Optional) Neural baselines — requires GPU

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.46.3 sentencepiece==0.2.0
.\.venv\Scripts\python.exe src/baselines.py
```

> Falls back to CPU automatically if no CUDA GPU is detected (much slower).

### 5. Run the backend

```powershell
# From the project root
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

API explorer: **http://localhost:8000/docs**

### 6. Run the frontend (separate terminal)

```powershell
cd frontend
npm install
npm run dev
```

App: **http://localhost:5173**

---

## Run Tests

```powershell
# Inference latency test (verifies < 10s per request requirement)
.\.venv\Scripts\python.exe tests/test_inference.py

# Backend smoke test
.\.venv\Scripts\python.exe tests/smoke_test_backend.py

# End-to-end smoke test (requires backend running on port 8000)
.\.venv\Scripts\python.exe tests/smoke_test_e2e.py
```

---

## API Endpoints

Base URL: `http://localhost:8000`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service banner + endpoint list |
| `GET` | `/healthz` | `{ok, models_loaded}` liveness check |
| `GET` | `/sample` | Random RACE val article (no gold question) |
| `GET` | `/sample_with_question` | Random RACE val row with gold question + options |
| `POST` | `/generate` | Article → generated question + 4 options + latency |
| `POST` | `/predict` | Article + question + 4 options → ensemble prediction + per-option scores |
| `POST` | `/distractors` | Article + correct answer → 3 distractors |
| `POST` | `/hints` | Article + question → 3 graduated hints (vague / moderate / specific) |
| `GET` | `/analytics` | Cached val/test metrics for the dashboard |

Every response includes a `latency_ms` field.

---

## Project Structure

```
race-rc-project/
├── data/
│   ├── train.csv                 ← place Kaggle download here
│   ├── train_split.csv           ← generated by preprocessing.py
│   ├── val_split.csv
│   ├── test_split.csv
│   ├── X_train.npz               ← TF-IDF sparse matrices
│   ├── X_val.npz
│   ├── X_test.npz
│   ├── y_train.npy               ← option-level labels
│   ├── y_val.npy
│   └── y_test.npy
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── ensemble.pkl              ← {models, weights}
│   ├── hint_scorer.pkl
│   ├── distractor_ranker.pkl
│   ├── question_ranker.pkl
│   ├── kmeans_model.pkl
│   ├── model_a_metrics.json
│   ├── model_b_metrics.json
│   ├── model_a_gen_metrics.json
│   ├── unsupervised_metrics.json
│   ├── hyperparameter_sweep.json
│   └── final_test_metrics.json
├── src/
│   ├── preprocessing.py          80/10/10 split
│   ├── model_a_train.py          TF-IDF + LR/SVC/NB training
│   ├── model_a_unsupervised.py   K-Means + Label Propagation
│   ├── model_a_generate.py       Wh-template generator + RF ranker
│   ├── model_b.py                Distractors + Hint scorer
│   ├── hyperparameter_sweep.py   GridSearchCV + ensemble weight search
│   ├── evaluate.py               Final test-set evaluation + figures
│   ├── build_metrics_json.py     Refresh backend/metrics.json
│   └── baselines.py              BERT-LR + T5-small (GPU, optional)
├── backend/
│   ├── main.py                   FastAPI app · 8 endpoints
│   └── metrics.json              Served by GET /analytics
├── frontend/                     Vite + React + Tailwind + Recharts
├── tests/
│   ├── test_inference.py         Latency test (< 10s requirement)
│   ├── smoke_test_backend.py     Backend endpoint smoke tests
│   └── smoke_test_e2e.py         End-to-end smoke tests
├── notebooks/
│   ├── EDA.ipynb
│   └── figures/                  Confusion matrices + comparison charts
├── requirements.txt
└── README.md
```

---

## Reproducibility

- All `random_state` values are pinned to **42**.
- `requirements.txt` is generated by `pip freeze` — all transitive dependencies are pinned.
- The test split is **never touched** until `src/evaluate.py` runs.
- Delete any `.npz` or `.pkl` file and the corresponding script will regenerate it cleanly.
- GPU step (`src/baselines.py`) is fully optional — the rest of the pipeline runs on CPU only.

---

## Known Limitations

- **Template-based generation:** Questions are syntactically clean but lexically formulaic; T5-small beats them by ~0.02 METEOR on the same 200-sample subset.
- **Hint R² is negative:** The scorer is trained on question-relevance (only information available at inference time) but evaluated against answer-relevance gold. P@3 = 0.74 is the correct headline metric.
- **Calibrated SVC probabilities:** Sigmoid calibration under 1:3 imbalance pushes probabilities below 0.5 for the correct class; mitigated by per-question argmax.
- **RACE-only:** No cross-domain validation. Performance on other MCQ datasets is unknown.

---

## License

Educational use only. Dataset license follows the original
[RACE distribution](http://www.cs.cmu.edu/~glai1/data/race/).
