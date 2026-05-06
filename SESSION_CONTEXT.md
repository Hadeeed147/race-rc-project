# RACE Project — Session Context
> Update this file at the END of every session. Paste it at the START of the next one.

---

## 🔧 Project Stack
- **Dataset:** RACE (one CSV, 80/10/10 split)
- **Backend:** Python + FastAPI + scikit-learn
- **Frontend:** React (Vite or CRA)
- **Models:** Classical ML only (no neural nets)
- **Features:** TF-IDF only
- **Team:** 2 people

---

## ✅ Completed So Far
<!-- Check items off as you finish them -->
- [x] Project folder created
- [x] Dataset downloaded + 80/10/10 split done
- [x] EDA notebook done
- [ ] TF-IDF vectorizer trained + saved
- [ ] Logistic Regression trained + saved
- [ ] SVM trained + saved
- [ ] Naive Bayes trained + saved
- [ ] K-Means clustering done
- [ ] Ensemble (soft voting) done
- [ ] Model A evaluation done (Accuracy, F1, Confusion Matrix)
- [ ] Distractor pipeline done
- [ ] Hint generator done
- [ ] Model B evaluation done
- [ ] FastAPI backend running (`/predict`, `/distractors`, `/hints`)
- [ ] React frontend Screen 1 (Article Input)
- [ ] React frontend Screen 2 (Quiz View)
- [ ] React frontend Screen 3 (Hint Panel)
- [ ] React frontend Screen 4 (Analytics Dashboard)
- [ ] Frontend ↔ Backend integrated
- [ ] Final testing done

---

## 📊 Current Model Results
<!-- Fill in as you get results -->
| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression | — | — | |
| SVM | — | — | |
| Naive Bayes | — | — | |
| Ensemble | — | — | |
| K-Means (Silhouette) | — | — | |

---

## 🐛 Known Issues / Blockers
<!-- Write current problems here so Claude can help immediately -->
- **Split Python install on this machine.** `python` on PATH resolves to
  `C:\Python313\python.exe` (no stdlib, no pip); the working stdlib lives at
  `C:\Users\Victus\AppData\Local\Programs\Python\Python313\`. **Workaround
  in use:** project-local venv at `.venv/` created with
  `PYTHONHOME=C:\Users\Victus\AppData\Local\Programs\Python\Python313`. Always
  activate the venv (`.\.venv\Scripts\Activate.ps1`) before running anything;
  Jupyter kernel `race-rc` is registered to point at the same venv.
- **GitHub push pending** — repo to be created manually, then `git init` /
  remote / push from project root. Add `.venv/`, `data/*.csv`, `models/*.pkl`,
  `__pycache__/`, `.ipynb_checkpoints/` to `.gitignore` before first commit.

---

## 📁 Current File Structure
<!-- Update this as files get created -->
```
race_rc_project/
├── .venv/                      (local venv — kernel "race-rc" registered)
├── data/
│   ├── train.csv               (full RACE train set, ~150 MB)
│   ├── train_split.csv         (~70k rows — 80%)
│   ├── val_split.csv           (~8.7k rows — 10%)
│   └── test_split.csv          (~8.7k rows — 10%, untouched until Session 5)
├── models/                     (empty — Session 2)
├── src/
│   └── preprocessing.py        (loads train.csv, 80/10/10 split, saves splits)
├── backend/                    (empty — Session 3)
├── frontend/                   (empty — Session 3)
├── notebooks/
│   └── EDA.ipynb               (answer dist, length histos, summary stats — RUN)
├── requirements.txt            (cp313-compatible pins: pandas 2.2.3, numpy
│                                2.2.2, sklearn 1.6.1, fastapi, uvicorn,
│                                joblib, jupyter, matplotlib, seaborn)
├── RACE_Project_MasterPlan.md
└── SESSION_CONTEXT.md
```

---

## 🔜 Next Session Goal
<!-- Write 1-3 specific goals for the NEXT session -->
- **Session 2** — train TF-IDF vectorizer + LR/SVM/NB classifiers + K-Means
  clustering (Model A); start Model B distractor + hint pipelines (Person 2).

---

## 💬 Last Session Notes
<!-- Paste any important decisions or code snippets from last session -->
- **Session 1 (this session):** scaffolded folder structure; wrote
  `src/preprocessing.py` (80/10/10 split, random_state=42); pinned
  `requirements.txt` to cp313-compatible versions; authored and **ran**
  `notebooks/EDA.ipynb` (answer-dist bar chart, article + question word-count
  histograms, summary stats table, sample rows); created project-local
  `.venv/` and registered Jupyter kernel `race-rc`; placed `data/train.csv`
  and produced all three splits.
- **Frontend (React/Vite) intentionally skipped** until Session 3 per plan.
- **Feature representation locked to TF-IDF only** (no One-Hot Encoding).
- Preprocessing script paths are anchored from project root via
  `os.path.dirname(...)` so it can be invoked from any cwd.
- Initial requirements pin (numpy 1.26.4) failed to build on Python 3.13 — no
  cp313 wheels. Bumped to numpy 2.2.2 / pandas 2.2.3 / sklearn 1.6.1 /
  matplotlib 3.10.0; install succeeded inside the venv.
- Daily flow: `.\.venv\Scripts\Activate.ps1` → `python src\preprocessing.py`
  or `python -m jupyter notebook notebooks\EDA.ipynb` (pick kernel
  `Python (race-rc)`).
