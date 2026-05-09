# RACE Reading Comprehension AI

A classical-ML reading-comprehension and quiz system over the
[RACE dataset](https://www.kaggle.com/datasets/ankitdhiman7/race-dataset)
(English exam passages with multiple-choice questions). Given a passage,
the system generates a multiple-choice question, scores each option with
a soft-vote ensemble of TF-IDF classifiers, surfaces three graduated hints,
and exposes everything through a polished React UI. **No neural networks
in the production pipeline** — BERT and T5 are reported as inference-only
baselines.

---

## ✨ Features

- **Generation** — Wh-template question generator with answer-span
  extraction, length-capped stems, and a Random-Forest informativeness
  ranker over 7 sentence-level features.
- **Verification (Model A)** — Logistic Regression, calibrated LinearSVC,
  and ComplementNB classifiers over a 20 000-feature TF-IDF
  vocabulary; soft-vote ensemble with searched weights.
- **Distractors (Model B)** — Frequent content-word/bigram extraction +
  TF-IDF cosine similarity to the correct answer + diversity penalty.
- **Hints (Model B)** — Sentence-level informativeness scorer (LR over
  cosine sim, keyword overlap, position, length); three graduated hints
  ranked vague → moderate → specific.
- **K-Means + Label Propagation** — Unsupervised clusters (silhouette /
  purity) and a semi-supervised baseline over a TruncatedSVD projection.

---

## 🏗 Architecture

```
                                ┌───────────────────────┐
                                │   data/train.csv     │
                                └──────────┬────────────┘
                                           │ 80/10/10
                                           ▼
                              ┌──────────────────────────┐
                              │  src/preprocessing.py    │
                              │  src/features.py         │  TF-IDF (20 000)
                              └──────────┬───────────────┘
                                         │
                ┌────────────────────────┴──────────────────────────┐
                ▼                                                   ▼
   ┌─────────────────────────┐                       ┌──────────────────────────┐
   │ Model A — verification  │                       │ Model B — distractors    │
   │  LR · SVC · NB           │                       │ + graduated hints        │
   │  Ensemble (weight search)│                       │ TF-IDF cosine + LR scorer│
   │  K-Means · LabelProp     │                       │                          │
   │  Wh-template generator   │                       │                          │
   └─────────┬───────────────┘                       └──────────────┬───────────┘
             └─────────────────────┐    ┌──────────────────────────┘
                                   ▼    ▼
                       ┌─────────────────────────┐
                       │  backend/main.py        │
                       │  FastAPI · 7 endpoints  │
                       │  /predict /generate     │
                       │  /distractors /hints    │
                       │  /sample /analytics     │
                       │  /healthz               │
                       └────────────┬────────────┘
                                    │ JSON over CORS
                                    ▼
                       ┌─────────────────────────┐
                       │  frontend/ (Vite/React) │
                       │  Tailwind · shadcn-style│
                       │  Recharts · Framer M.   │
                       │  Zustand · lucide-react │
                       └─────────────────────────┘
```

---

## 📊 Final results (held-out test set)

> The **test split (≈ 8 787 questions)** is sacred — it was untouched until
> the final evaluation step. Numbers below are computed by
> `src/final_evaluation.py` and are also the ones served by
> `GET /analytics`.

| Model | Accuracy | Macro F1 | Exact Match |
|-------|---------:|---------:|------------:|
| Random baseline                              | — | — | 0.250 |
| BERT-base (frozen) + LR head                 | _see backend/metrics.json_ | _ ′ _ | _ ′ _ |
| **Tuned ensemble (LR + SVC + NB)**            | _see backend/metrics.json_ | _ ′ _ | _ ′ _ |

| Distractors (200-sample test subset) | BLEU | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | METEOR | F1 |
|--------------------------------------|-----:|----------:|----------:|----------:|-------:|---:|
| TF-IDF cosine + diversity            | _see model_b on test_  |   |   |   |   |   |

| Hints (test, n=200) | P @ 1 | P @ 3 | R² (scorer) |
|---------------------|------:|------:|------------:|
| LR scorer           | _see test json_ | _ ′ _ | _ ′ _ |

> Exact numbers are also written into `models/final_test_metrics.json`,
> `models/baselines_metrics.json`, and `models/hyperparameter_sweep.json`.
> Three publication-quality figures are saved under
> `notebooks/figures/`: `final_confusion_matrix.png`,
> `final_model_comparison.png`, `final_metric_breakdown.png`.

---

## ⚙ Setup

### Requirements

- Python **3.13** (we use 3.13.7; any 3.11+ works after re-pinning)
- Node **18+** (we use 20)
- 8 GB RAM minimum. GPU is **optional** — needed only for the BERT/T5
  baselines (`src/baselines.py`).

### Windows (PowerShell)

```powershell
git clone <your-fork-url> race_rc_project
cd race_rc_project

# 1. Python venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Place the dataset
#    Download train.csv from
#    https://www.kaggle.com/datasets/ankitdhiman7/race-dataset
#    and drop it at  data\train.csv

# 3. Build the splits, features, and models (≈ 25 min on CPU)
python src\preprocessing.py
python src\features.py
python src\model_a_train.py
python src\model_a_unsupervised.py
python src\ensemble.py
python src\model_a_generate.py
python src\model_b.py
python src\evaluate_model_b.py

# 4. (Optional, GPU) Neural baselines (~25 min on RTX 3050)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers==4.46.3 sentencepiece==0.2.0
python src\baselines.py

# 5. Hyperparameter sweep + final test evaluation
python src\hyperparameter_sweep.py
python src\final_evaluation.py
python src\build_metrics_json.py

# 6. Backend
python -m uvicorn backend.main:app --reload --port 8000

# 7. Frontend (new shell)
cd frontend
npm install
npm run dev               # http://localhost:5173
```

### Linux / macOS

Same steps with these differences:
- `python -m venv .venv && source .venv/bin/activate`
- Drop the `\` path separators
- `pip install torch` without the `--index-url` if you don't have CUDA;
  the `baselines.py` script falls back to CPU automatically (slower).

---

## 🗂 Project layout

```
race_rc_project/
├── data/                    train + split CSVs, sparse TF-IDF, label arrays
├── models/                  every trained pickle + the metrics JSONs
├── src/
│   ├── preprocessing.py     80/10/10 split
│   ├── features.py          option-level reshape + TF-IDF
│   ├── model_a_train.py     LR · SVC · NB
│   ├── model_a_unsupervised.py  K-Means · LabelProp · TruncatedSVD
│   ├── model_b.py           distractors + hints + hint-scorer
│   ├── ensemble.py          soft-vote ensemble
│   ├── model_a_generate.py  improved Wh-template generator + RF ranker
│   ├── evaluate_model_b.py  BLEU / ROUGE / METEOR / R²
│   ├── baselines.py         BERT-LR + T5-small (GPU)
│   ├── hyperparameter_sweep.py   GridSearchCV + ensemble-weight search
│   ├── final_evaluation.py  test-set numbers + figures
│   ├── build_metrics_json.py    refresh backend/metrics.json
│   └── append_eda_cells.py
├── backend/
│   ├── main.py              FastAPI app · 7 endpoints · CORS
│   └── metrics.json         served by GET /analytics
├── frontend/                Vite · React Router · Tailwind · shadcn-style
├── notebooks/
│   ├── EDA.ipynb
│   └── figures/             confusion matrix + model comparison + breakdown
├── requirements.txt         pinned versions
└── README.md
```

---

## 🌐 API endpoints (from `backend/main.py`)

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/`            | service banner |
| `GET`  | `/healthz`     | `{ok, models_loaded}` |
| `GET`  | `/sample`      | random RACE val row (article + gold question + options) |
| `POST` | `/generate`    | article → generated question + 4 options |
| `POST` | `/predict`     | article + question + 4 options → ensemble prediction + per-option scores |
| `POST` | `/distractors` | article + correct answer → 3 distractors |
| `POST` | `/hints`       | article + question → 3 graduated hints |
| `GET`  | `/analytics`   | val/test metrics for the dashboard |

Every response includes a `latency_ms` field. The OpenAPI explorer is at
`http://localhost:8000/docs`.

---

## 🔁 Reproducibility

The full pipeline is runnable from a fresh shell with the seven `python
src/*.py` commands above (in order). All `random_state` values are pinned
to `42`. `requirements.txt` is generated by `pip freeze` and pins every
transitive dependency.

> If you delete `data/X_*.npz` or any `.pkl` in `models/`, the relevant
> script will re-create it. Skip the GPU step entirely if you don't have
> CUDA — the rest of the pipeline does not depend on it.

---

## 🧠 Honest limitations

- **Question generator is template-based**, so generated stems are
  syntactically clean but lexically formulaic. We compare against
  T5-small in `models/baselines_metrics.json`.
- **Hint scorer R² is negative** because the eval gold (sentence-overlap
  with the *answer*) is a different objective from the training target
  (top-20% by cosine to the *question*). Documented; not a bug.
- **Calibrated LinearSVC's `predict()` collapses at threshold 0.5** under
  the 1:3 imbalance; we side-step this by using `predict_proba` and
  per-question argmax everywhere.

---

## License

Educational use; dataset license follows the original RACE distribution.
