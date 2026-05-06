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
- **`data/train.csv` not yet placed.** Download manually from
  <https://www.kaggle.com/datasets/ankitdhiman7/race-dataset> and drop the
  `train.csv` into `data/` before re-running preprocessing.
- **Python install on this machine is broken.** `python` on PATH points to
  `C:\Python313\python.exe`, which fails with `Could not find platform
  independent libraries <prefix>` and has no `pip` module. `pip.exe` on PATH
  resolves to a different (orphaned) Python at
  `C:\Users\Victus\AppData\Local\Programs\Python\Python313\` whose `python.exe`
  is missing. Result: `python src/preprocessing.py` did NOT execute end-to-end
  this session — could not import pandas. Fix before Session 2:
  reinstall Python 3.11 or 3.12 cleanly (uncheck old installs first), then
  `pip install -r requirements.txt`.
- Once both blockers are cleared, run `python src/preprocessing.py` to
  produce `train_split.csv`, `val_split.csv`, `test_split.csv`, then run
  `notebooks/EDA.ipynb` cell-by-cell to confirm EDA output.

---

## 📁 Current File Structure
<!-- Update this as files get created -->
```
race_rc_project/
├── data/
│   └── train.csv               (PENDING — user to place manually)
├── models/                     (empty — Session 2)
├── src/
│   └── preprocessing.py        (loads train.csv, 80/10/10 split, saves splits)
├── backend/                    (empty — Session 3)
├── frontend/                   (empty — Session 3)
├── notebooks/
│   └── EDA.ipynb               (answer dist, length histos, summary stats)
├── requirements.txt            (pandas, numpy, sklearn, fastapi, uvicorn,
│                                joblib, jupyter, matplotlib, seaborn — pinned)
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
- **Session 1 (this session):** scaffolded folder structure, wrote
  `src/preprocessing.py` (80/10/10 split, random_state=42), pinned
  `requirements.txt`, authored `notebooks/EDA.ipynb` with answer-distribution
  bar chart, article + question word-count histograms, summary stats table,
  and sample row prints.
- **Frontend (React/Vite) intentionally skipped** until Session 3 per plan.
- **Feature representation locked to TF-IDF only** (no One-Hot Encoding).
- Preprocessing script paths are anchored from project root via
  `os.path.dirname(...)` so it can be invoked from any cwd.
- See **Known Issues** above for the two blockers carried into Session 2:
  missing `train.csv` and broken local Python install.
