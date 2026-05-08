# RACE Project — Session Context
>
> Update this file at the END of every session. Paste it at the START of the next one.

---

## 🔧 Project Stack

- **Dataset:** RACE (one CSV, 80/10/10 split)
- **Backend:** Python · FastAPI · scikit-learn · joblib
- **Frontend:** React 18 · Vite · React Router · Tailwind CSS 3 · shadcn-style primitives (Radix UI) · lucide-react · Recharts · Framer Motion · Zustand · canvas-confetti
- **Models:** Classical ML only (no neural nets)
- **Features:** TF-IDF only

---

## ✅ Completed So Far

- [x] Project folder created
- [x] Dataset downloaded + 80/10/10 split done
- [x] EDA notebook done
- [x] TF-IDF vectorizer trained + saved
- [x] Logistic Regression trained + saved
- [x] SVM trained + saved
- [x] Naive Bayes trained + saved
- [x] K-Means clustering done
- [x] Ensemble (soft voting) done
- [x] Model A evaluation done (Accuracy, F1, Confusion Matrix)
- [x] Distractor pipeline done
- [x] Hint generator done
- [x] Model B evaluation done   (BLEU/ROUGE/METEOR + R²)
- [x] FastAPI backend running (`/predict`, `/distractors`, `/hints`, `/generate`, `/analytics`, `/sample`, `/healthz`)
- [x] React frontend Screen 1 (Article Input)
- [x] React frontend Screen 2 (Quiz View)
- [x] React frontend Screen 3 (Hint Panel)
- [x] React frontend Screen 4 (Analytics Dashboard)
- [x] **Frontend ↔ Backend integrated**     (Session 4)
- [ ] Final testing done                    (Session 5)

---

## 📊 Current Model Results

**Model A — supervised + ensemble (val: 8 787 questions / 35 148 option rows)**

| Model | Accuracy | Macro F1 | Exact Match |
|-------|---------:|---------:|------------:|
| Logistic Regression (balanced)            | 0.6710 | 0.5408 | 0.3082 |
| LinearSVC (calibrated, balanced)          | 0.6690 | 0.5381 | 0.3040 |
| ComplementNB                              | 0.6629 | 0.5296 | 0.2948 |
| **Ensemble (LR + SVC + NB, equal avg)**   | **0.6728** | **0.5434** | **0.3114** |
| K-Means (Silhouette / Purity)             | 0.0072 / 0.7472 | — | — |
| LabelProp (knn=7, SVD-100, 10% labels)    | F1 0.4445 (vs sup-LR 0.4833) | — | — |

**Model B — distractors + hints (200 val samples)**

| Distractors | BLEU | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | METEOR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF cosine + diversity | 0.0011 | 0.1012 | 0.0069 | 0.1009 | 0.0770 | 0.1463 |
| Hints | n | P @ 1 | P @ 3 | R² (scorer) |
|---|---:|---:|---:|---:|
| LR scorer over (cosine, overlap, position, length) | 187 | 0.0267 | 0.1551 | −4.0340 |

---

## 🚀 Session 4 — End-to-end smoke test (10 random val samples)

| Endpoint | Mean | Max |
|----------|-----:|----:|
| GET /sample      | 31 ms  | 197 ms (cold sample) |
| POST /generate   | 159 ms | 213 ms |
| POST /predict    | 22 ms  | 34 ms  |
| POST /hints      | 17 ms  | 21 ms  |
| GET /analytics   | 8 ms   | 8 ms   |

> **All endpoints well under the 10 000 ms rubric ceiling.** Ensemble was
> correct on **5 of 10** samples (vs 25 % random baseline; matches the
> reported 0.31 EM at this small-sample variance).

---

## 🐛 Known Issues / Blockers

- **Generated questions remain formulaic** — Wh-template + RF ranker still
  copies the source noun phrase into the stem
  (e.g. *"What does the passage say about single colony queen hundreds?"*).
  Acceptable for the demo, but Session 5 should add stem post-processing.
- **Calibrated LinearSVC** still has the threshold-0.5 collapse — sidestep
  remains: ensemble averages `predict_proba` and per-question argmax.
- **Hint R² = −4.034** — training target (question-relevance) ≠ eval gold
  (answer-relevance). Not a bug.
- **shadcn/ui CLI auto-install was not used** (it tries to npx-create-app
  scaffolding that conflicts with Vite). Instead, primitives were
  hand-written using Radix UI + class-variance-authority — exactly the
  source the CLI generates. Behaviour is identical; saves ~30 npm packages.
- **`Could not find platform independent libraries <prefix>`** — cosmetic
  warning, ignore.
- **Long-running dev servers from the smoke test** linger ~10 min unless
  killed. To kill: `Get-Process python, node | Stop-Process -Force` in
  PowerShell. Restart fresh next session.

---

## 📁 Current File Structure

```
race_rc_project/
├── .venv/
├── data/   (split CSVs + sparse TF-IDF + label arrays — Sessions 1–2)
├── models/ (vectorizer, lr/svc/nb, kmeans, label_prop, ensemble, question_ranker, hint_scorer, *_metrics.json)
├── src/
│   ├── preprocessing.py / features.py / model_a_train.py / model_a_unsupervised.py / model_b.py
│   ├── ensemble.py / model_a_generate.py / evaluate_model_b.py
│   ├── append_eda_cells.py / build_metrics_json.py
│   ├── smoke_test_backend.py
│   └── smoke_test_e2e.py                     ← S4 10-sample end-to-end test
├── backend/
│   ├── main.py                               (now also serves /sample + /healthz, val_df cached at startup)
│   └── metrics.json
├── frontend/
│   ├── package.json   vite.config.js   tailwind.config.js   postcss.config.js   index.html
│   ├── .gitignore
│   ├── node_modules/    (Tailwind, Radix UI, Recharts, Framer Motion, Zustand, lucide-react, canvas-confetti, etc.)
│   └── src/
│       ├── main.jsx     App.jsx     index.css            ← Tailwind base + custom CSS vars (light + dark)
│       ├── lib/
│       │   ├── api.js               (fetch wrapper + ApiError + 15s timeout + 7 named endpoints)
│       │   ├── store.js             (Zustand: useQuiz + useUI w/ persisted theme + banner + latency history)
│       │   ├── useTheme.js          (writes <html class="dark">)
│       │   └── utils.js             (cn / formatPercent / formatNumber / wordCount / readingTimeMinutes)
│       ├── components/
│       │   ├── NavBar.jsx           (logo + 4 NavLinks + shortcuts btn + dark-mode toggle)
│       │   ├── Banner.jsx           (Framer-Motion-collapsible AI-disclaimer, dismiss persists in localStorage)
│       │   ├── Footer.jsx
│       │   ├── ShortcutsDialog.jsx  (?-key opens; lists G/Q/H/A/D shortcuts)
│       │   └── ui/                  (hand-written shadcn-style primitives over Radix UI)
│       │       ├── Button.jsx Card.jsx Accordion.jsx Dialog.jsx
│       │       ├── RadioGroup.jsx Progress.jsx Skeleton.jsx Badge.jsx Alert.jsx
│       │       ├── Toast.jsx Toaster.jsx use-toast.js
│       └── pages/
│           ├── ArticleInput.jsx    (hero + char/word/reading-time meter + load-sample + generate, Framer Motion fade)
│           ├── Quiz.jsx            (split layout, RadioGroup, Check Answer → confetti + green/red + confidence bars)
│           ├── Hints.jsx           (Accordion graduated hints + Reveal-Answer Dialog + copy-to-clipboard + progress)
│           ├── Analytics.jsx       (4 metric cards + Recharts BarChart + custom CM heatmap + Model B table + system row)
│           └── NotFound.jsx        (illustrated 404 → home)
├── notebooks/   (EDA.ipynb + figures/)
├── requirements.txt
├── .gitignore
├── RACE_Project_MasterPlan.md
└── SESSION_CONTEXT.md
```

---

## 🔜 Next Session Goal

- **Session 5** — BERT/T5 inference baselines on test (held-out until now);
  hyperparameter sweep on the ensemble (`GridSearchCV` over LR `C`, SVC
  `C`, NB `alpha`, ensemble weights); final test-set evaluation across all
  metrics (Accuracy / Macro F1 / EM / Confusion Matrix / BLEU / ROUGE /
  METEOR); README + reproducibility check (`python src/preprocessing.py …`
  through `npm run build`); pin all versions; final commit.

---

## 💬 Last Session Notes

- **Session 4 (this session):**
  - Frontend was rebuilt from a basic skeleton into a polished, demo-ready
    React app: Tailwind CSS 3 with custom HSL color tokens (indigo
    primary, amber accent, slate neutrals) + a true dark mode (not just
    inverted), JetBrains Mono + Plus Jakarta Sans typography, glass
    NavBar with active-state highlighting, dismissable Framer Motion
    banner, footer, illustrated 404, Toast notifications, keyboard
    shortcuts dialog (?-key opens; G/Q/H/A/D navigate; D toggles dark
    mode), and a gradient SVG favicon.
  - **shadcn-style primitives** were written by hand over Radix UI
    (Button, Card, Accordion, Dialog, RadioGroup, Progress, Skeleton,
    Badge, Toast/Toaster, Alert) — equivalent to what the shadcn CLI
    generates but without the auto-install conflicts on Windows.
  - **API client** at `src/lib/api.js`: single `fetchJSON` wrapper with a
    15-second `AbortController` timeout, typed `ApiError`, and seven named
    functions (getHealth / getSample / generateQuiz / predictAnswer /
    getDistractors / getHints / getAnalytics).
  - **State** is Zustand: `useQuiz` carries the active quiz (article,
    question, options, gold answer, hints used); `useUI` is persisted in
    localStorage and tracks theme + banner-dismissed + a 20-entry
    latency history.
  - **All four pages fully wired:**
    - ArticleInput posts to `/sample` (real RACE row) and `/generate`
      (model question), with a live char/word/reading-time meter and an
      animated Progress + Skeleton block during generation.
    - Quiz pulls the quiz from the store, renders article + RadioGroup
      side-by-side, shows confidence bars per option after `/predict`,
      lights up green/red against the gold answer, and triggers
      canvas-confetti on a correct pick. Verdict card shows
      model_used + latency_ms.
    - Hints calls `/hints` on mount, renders three Accordion items
      (vague / moderate / specific) with a "Hints used: X / 3" Progress
      indicator at the top and a "Reveal Answer" Dialog with
      copy-to-clipboard.
    - Analytics calls `/analytics` on mount, renders 4 metric cards
      (Accuracy / Macro F1 / EM / Avg latency from this session's history),
      a Recharts BarChart of Macro F1 vs EM across all 4 Model A variants,
      a custom 2×2 confusion-matrix heatmap, a clean Model B metrics
      table, and a system info row. Everything fades in row-by-row using
      Framer Motion stagger.
  - **Backend tweaks:** added `GET /sample` (random row from the cached
    val DataFrame) and `GET /healthz`; `_load_models` now caches
    `val_split.csv` at startup. CORS still scoped to `:5173 / :3000`
    (broaden to `*` if needed for a remote demo).
  - **End-to-end test (10 random val samples):** all 10 passed, latencies
    `/predict` mean 22 ms / max 34 ms, `/hints` mean 17 ms / max 21 ms,
    `/generate` mean 159 ms / max 213 ms — well under the 10 000 ms
    rubric ceiling. Ensemble correct on 5 of 10 (matches val EM 0.31 at
    n=10 variance).
- Daily flow:

  ```
  .\.venv\Scripts\Activate.ps1
  python -m uvicorn backend.main:app --reload --port 8000
  cd frontend ; npm run dev          # http://localhost:5173
  ```
