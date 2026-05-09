# RACE Project — Report Data Dump
> Source-of-truth file for final report writing. All numbers from the
> frozen JSONs in `models/`. No prose; this is data + outline only.

---

## 1. Project Identity

- **Title:** Intelligent Reading Comprehension and Quiz Generation System using Machine Learning
- **Course:** AI Lab — BS (CS) Spring 2026 — FAST NUCES Islamabad
- **Team size:** 2
- **Dataset:** RACE (1 CSV, 80/10/10 split, `random_state=42`)
- **Splits:** 70 292 train / 8 787 val / 8 787 test (option-level: 281 168 / 35 148 / 35 148)
- **Feature representation:** TF-IDF
  - `max_features=20 000`
  - `ngram_range=(1, 2)`
  - `min_df=2`
  - `stop_words='english'`
  - `sublinear_tf=True`
  - Fit on TRAIN only; `transform()` on val/test.

---

## 2. Architecture

```
                    [data/train.csv]
                          │  80 / 10 / 10  random_state=42
                          ▼
                ┌─────────────────────────┐
                │  src/preprocessing.py    │
                │  src/features.py         │   TF-IDF (20 000 vocab)
                └────────────┬────────────┘
                             │
        ┌────────────────────┴──────────────────────┐
        ▼                                            ▼
  ┌──────────────────────────┐       ┌─────────────────────────────┐
  │ Model A — verification    │       │ Model B — distractors + hints│
  │   LR · LinearSVC · ComNB  │       │   noun-phrase extraction     │
  │   Soft-vote ensemble      │       │   TF-IDF cosine + diversity  │
  │     (weights 0.50/0.25/0.25)│      │   sentence scorer (LR over   │
  │   K-Means · LabelProp      │       │     cosine, overlap, posn,   │
  │   Wh-template generator   │       │     length)                   │
  └────────────┬──────────────┘       └─────────────┬────────────────┘
               └─────────────────┐  ┌────────────────┘
                                 ▼  ▼
                       ┌──────────────────────────┐
                       │  backend/main.py         │
                       │  FastAPI · 8 endpoints   │
                       │  /generate /predict      │
                       │  /distractors /hints     │
                       │  /sample /sample_with_q  │
                       │  /analytics /healthz     │
                       └──────────────┬───────────┘
                                      │ JSON over CORS
                                      ▼
                       ┌──────────────────────────┐
                       │  frontend/ (Vite + React) │
                       │  Tailwind · shadcn-style  │
                       │  Recharts · Framer Motion │
                       │  Zustand · lucide-react   │
                       └──────────────────────────┘
```

---

## 3. Final Test-Set Results — Model A

> Test split was untouched until Session 5. Numbers below come from
> `models/final_test_metrics.json`.

| Model (test, n=8 787 questions / 35 148 options) | Accuracy | Macro F1 | Exact Match |
|--------------------------------------------------|---------:|---------:|------------:|
| Random baseline                                  | —        | —        | 0.2500 |
| BERT-base (frozen) + LR head                     | 0.6345   | 0.4895   | 0.2359 |
| **Tuned soft-vote ensemble (LR-heavy 0.5/0.25/0.25)** | **0.6718** | **0.5417** | **0.3083** |

> Ours beats random by **+0.058** EM and beats frozen BERT-LR by **+0.072** EM.
> Frozen BERT scores below random (0.2359 vs 0.2500) — task-tuned
> features beat off-the-shelf transformer embeddings here.

### 3a. Confusion matrices (test)

**Tuned ensemble (per-option, after per-question argmax):**
| | pred 0 | pred 1 |
|---|------:|------:|
| true 0 | 21 171 | 5 190 |
| true 1 | 6 344  | 2 443 |

**BERT-LR head:**
| | pred 0 | pred 1 |
|---|------:|------:|
| true 0 | 20 514 | 5 847 |
| true 1 | 7 001  | 1 786 |

---

## 4. Final Test-Set Results — Model B

> 200-sample test subset (full set is too slow for METEOR / ROUGE on CPU).

| Distractors | n | BLEU | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | METEOR | Precision | Recall | F1 |
|-------------|--:|-----:|----------:|----------:|----------:|-------:|----------:|-------:|---:|
| TF-IDF cosine + diversity (S2 single-token version, used at eval time) | 200 | 0.0011 | 0.0904 | 0.0059 | 0.0901 | 0.0653 | 0.2518 | 0.0950 | 0.1313 |

> **Note:** `model_b.py` was upgraded in Session 6 to use multi-word
> noun-phrase candidates for the demo. The test-set numbers above use the
> S2 frozen extractor; metrics were not re-computed because all test
> numbers were frozen at Session 5. Demo output now looks substantially
> better (see §5d).

| Hints | n | Precision @ 1 | Precision @ 3 | R² (scorer) |
|-------|--:|--------------:|--------------:|-----------:|
| LR scorer over (cosine, overlap, position, length) | 195 | 0.0718 | 0.1846 | −3.5107 |

> R² is negative because the eval gold (sentence with highest token
> overlap with the *answer*) is a different objective from the training
> target (top-20 % by cosine to the *question*). The scorer learned
> question-relevance, not answer-relevance — by design.

---

## 5. Validation-Set Comparison Tables

### 5a. Model A — supervised (val, after per-question argmax)

| Model (val) | Accuracy | Macro F1 | Exact Match |
|-------------|---------:|---------:|------------:|
| Logistic Regression (balanced)            | 0.6710 | 0.5408 | 0.3082 |
| LinearSVC (calibrated, balanced)          | 0.6690 | 0.5381 | 0.3040 |
| ComplementNB                              | 0.6629 | 0.5296 | 0.2948 |
| **Ensemble (LR + SVC + NB, lr_heavy weights)** | **0.6740** | **0.5438** | **0.3143** |

### 5b. Hyperparameter sweep summary

| Family | Best param | CV macro-F1 |
|--------|-----------|-----------:|
| LR     | `C = 0.1` | 0.4573 |
| ComplementNB | `alpha = 1.0` | 0.4742 |
| LinearSVC (calibrated, manual val-EM search) | `C = 1.0` | val EM 0.2328 |

| Ensemble weight scheme | LR / SVC / NB | val Macro F1 | val EM |
|-----------------------|---------------|-------------:|------:|
| equal                 | 0.333 / 0.333 / 0.333 | 0.5441 | 0.3123 |
| **lr_heavy (chosen)**  | **0.50 / 0.25 / 0.25** | **0.5451** | **0.3143** |
| svc_heavy             | 0.25 / 0.50 / 0.25 | 0.5440 | 0.3122 |
| nb_heavy              | 0.25 / 0.25 / 0.50 | 0.5418 | 0.3093 |
| em_weighted           | 0.340 / 0.335 / 0.325 | 0.5438 | 0.3122 |

> Net val EM lift from tuning (default S3 → tuned S5): **0.3114 → 0.3143** (+0.0029).
> The lift comes almost entirely from the weight search; per-model
> hyperparameters were already at sensible values from S2.

### 5c. Unsupervised + semi-supervised (val)

| Component | Metric | Value |
|-----------|--------|------:|
| MiniBatchKMeans (k=4, 20k sample, cosine) | Silhouette | 0.0072 |
| MiniBatchKMeans (k=4, 20k sample, cosine) | Cluster Purity | 0.7472 |
| LabelPropagation (knn=7, SVD-100, n=5000, 10 % labelled) | Macro F1 | 0.4445 |
| Supervised LR baseline on the same 500 labels (dense SVD) | Macro F1 | 0.4833 |

> K-Means silhouette ≈ 0 confirms that TF-IDF on these articles does not
> form well-separated clusters; purity ≈ base rate 0.75 reflects that
> the same article appears across all 4 options of a question, so the
> answer signal is not captured by unsupervised features.

### 5d. Question-generation baselines (val, n=200)

| Generator | BLEU | ROUGE-L | METEOR | Inference |
|-----------|-----:|--------:|-------:|----------:|
| **Ours · Wh-templates + RF ranker** | 0.0024 | 0.1062 | 0.0552 | CPU, ~150 ms / passage |
| **T5-small (60 M params, frozen)**   | 0.0087 | 0.1140 | 0.0778 | RTX 3050, beam search 4 |

> T5-small wins by Δ ≈ 0.01–0.02 on every surface metric, but needs a
> GPU and ~250 MB of weights. Our template generator runs on CPU, is
> deterministic, and is fully classical. Acceptable trade-off for the
> classical-ML scope of this project.

### 5e. Frozen BERT-LR neural baseline (val)

| Configuration | Accuracy | Macro F1 | Exact Match |
|---------------|---------:|---------:|------------:|
| `bert-base-uncased` [CLS] frozen + balanced LR head (50 k train, 35 k val) | 0.6331 | 0.4880 | 0.2331 |

> Below random EM 0.25 — confirms that for this verification task,
> off-the-shelf transformer embeddings without fine-tuning underperform
> task-tuned TF-IDF. Embedding cost: ~17 min on RTX 3050 @ batch 8.

---

## 6. Per-Class Classification Report (test, tuned ensemble)

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| wrong (0)    | 0.7694 | 0.8031 | 0.7859 | 26 361 |
| correct (1)  | 0.3201 | 0.2780 | 0.2976 |  8 787 |
| accuracy     | —      | —      | 0.6718 | 35 148 |
| macro avg    | 0.5447 | 0.5406 | 0.5417 | 35 148 |
| weighted avg | 0.6571 | 0.6718 | 0.6638 | 35 148 |

> The 1:3 class imbalance (1 correct out of 4 options) is reflected in
> the support column. `class_weight='balanced'` for LR / SVC and
> ComplementNB for NB are the chosen mitigations; we report Macro F1 in
> addition to accuracy so the minority class is not hidden.

---

## 7. Figures (6 total)

| Path | Caption |
|------|---------|
| `notebooks/figures/confusion_matrices.png`           | (S3) Side-by-side confusion matrices for LR / LinearSVC / ComplementNB on val. |
| `notebooks/figures/model_comparison.png`             | (S3) Bar chart: Macro F1 + Exact Match across LR / SVC / NB / Ensemble (val). |
| `notebooks/figures/kmeans_pca.png`                   | (S3) 2-D TruncatedSVD projection of TF-IDF features coloured by K-Means cluster (5 k sample). |
| `notebooks/figures/final_confusion_matrix.png`       | (S5) Test-set confusion matrix for the tuned ensemble. |
| `notebooks/figures/final_model_comparison.png`       | (S5) Test-set comparison: random baseline · BERT-LR · Ours, on EM + Macro F1. |
| `notebooks/figures/final_metric_breakdown.png`       | (S5) Per-class precision / recall / F1 heatmap for the tuned ensemble on test. |

---

## 8. Hardware + Reproducibility

- **Local hardware:** RTX 3050 6 GB Laptop GPU (used only for `src/baselines.py` — frozen BERT [CLS] + T5-small generation).
- **Software stack:**
  - Python 3.13.7 (project venv `.venv/`)
  - scikit-learn 1.6.1, pandas 2.2.3, numpy 2.2.2, scipy 1.14.1
  - FastAPI 0.115.6, uvicorn 0.32.1
  - PyTorch 2.6.0+cu124, transformers 4.46.3, sentencepiece 0.2.1
  - React 18.3.1, Vite 5.4, Tailwind CSS 3.4, Radix UI primitives, Recharts 3, Framer Motion 12, Zustand 5
- **Determinism:** every script that takes `random_state` uses **42**.
- **13-script reproducible pipeline** (run in order from a fresh shell):
  1. `python src/preprocessing.py`
  2. `python src/features.py`
  3. `python src/model_a_train.py`
  4. `python src/model_a_unsupervised.py`
  5. `python src/ensemble.py`
  6. `python src/model_a_generate.py`
  7. `python src/model_b.py`
  8. `python src/evaluate_model_b.py`
  9. `python src/baselines.py` (GPU optional)
  10. `python src/hyperparameter_sweep.py`
  11. `python src/final_evaluation.py`
  12. `python src/build_metrics_json.py`
  13. `python -m uvicorn backend.main:app --reload --port 8000`  (and `cd frontend ; npm run dev` in another shell)

---

## 9. Section-by-Section Outline (for the report writer)

### 9.1 Abstract (≤ 200 words)
- Problem framing: multiple-choice reading comprehension over the RACE benchmark.
- Approach: classical-ML pipeline with TF-IDF + soft-vote ensemble.
- Best result: tuned ensemble achieves **EM 0.3083** on the held-out test
  set (vs random 0.250 and frozen BERT-LR 0.2359).
- Demo system: FastAPI backend + React UI with Real / AI-Generated
  question modes, three graduated hints, and a live analytics dashboard.

### 9.2 Introduction
- Why reading comprehension matters (assessment, ed-tech).
- Why classical ML on this task is interesting (small footprint, fast,
  no GPU at inference).
- Contributions:
  - Soft-vote ensemble + tuned weights (val EM lift 0.3114 → 0.3143).
  - Comparison vs frozen BERT-LR (Ours wins by +0.072 test EM).
  - Wh-template question generator with stem post-processing.
  - Multi-word noun-phrase distractor extraction.
  - Production-quality React UI (dashboard, dark mode, keyboard shortcuts).

### 9.3 Related Work (≥ 5 papers — TODOs to cite, do NOT fabricate citations)
- TODO: Lai et al. 2017 — *RACE: Large-scale ReAding Comprehension Dataset From Examinations.*
- TODO: Devlin et al. 2018 — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- TODO: Raffel et al. 2020 — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).*
- TODO: Pedregosa et al. 2011 — *Scikit-learn: Machine Learning in Python.*
- TODO: a TF-IDF / classical-ML for QA paper (e.g. Chen & Cardie 2018, or a SQuAD-baseline paper).

### 9.4 Dataset Analysis
- Schema: `id, article, question, A, B, C, D, answer`.
- Splits: 70 292 / 8 787 / 8 787 (questions); 281 168 / 35 148 / 35 148 (options).
- Answer distribution from EDA notebook (insert bar chart).
- Article + question length distributions (histograms in EDA).
- Class imbalance: 1:3 at option-level — mitigated with
  `class_weight='balanced'` (LR, SVC) and ComplementNB (NB).

### 9.5 Methodology — Model A
- TF-IDF feature pipeline (parameters in §1).
- LR / LinearSVC / ComplementNB with class-weight or Complement variant.
- Soft-vote ensemble (`lr_heavy = (0.50, 0.25, 0.25)`, justified by §5b).
- K-Means (Silhouette / Purity) + Label Propagation (TruncatedSVD-100,
  10 % labelled).
- Template-based question generator: Wh-templates + RandomForest
  ranker over 7 sentence-level features. Stem post-processor (drop
  answer span, length cap, capitalisation, force `?`).
- Hyperparameter sweep: 3-fold StratifiedKFold on a 80 k sample of TRAIN,
  refit best on full TRAIN; ensemble-weight search on val.

### 9.6 Methodology — Model B
- Distractor pipeline (Session 6 update):
  - Multi-word noun-phrase extractor (regex `NP_PATTERN` for 2–3 word phrases).
  - TF-IDF cosine similarity to the correct answer.
  - Diversity penalty + substring filter + stopword guard.
  - Fallback to single-token / bigram extractor only if the NP pool runs out.
- Hint generator:
  - Sentence-level features (cosine sim to question, keyword overlap,
    position, length).
  - LogisticRegression scorer trained on heuristic gold (top-20 % by cosine).
  - Returns three sentences at relevance percentiles ~70 % / ~85 % / ~95 %.

### 9.7 Results
- Test-set table from §3.
- Comparison vs BERT-LR (§5e) and T5 (§5d).
- Ensemble vs individual models (§5a).
- Per-class breakdown (§6).

### 9.8 Discussion
- Why frozen BERT underperforms — the [CLS] token of an out-of-domain
  transformer is not optimised for option discrimination. Fine-tuning
  would likely close the gap; we did NOT fine-tune by design.
- Why the ensemble works — LR / SVC / NB make complementary errors;
  averaging probabilities + per-question argmax exploits this without
  needing a stacked meta-learner.
- K-Means weak silhouette = honest, not a bug — same article
  shared across the 4 options means unsupervised features can't
  separate the answer signal.
- Hint R² mismatch — the eval gold (answer-overlap) is not what the
  scorer was trained on (question-cosine). Documented; not a bug.
- Calibrated SVC predict-collapse — sigmoid calibration on imbalanced
  data drops everything below 0.5; we side-step by using `predict_proba`
  + per-question argmax.

### 9.9 Limitations
- Template generator can't paraphrase — classical-ML ceiling.
- Hint scorer trained for question-relevance, not answer-relevance.
- Single dataset (RACE only) — no cross-domain validation.
- Confidence variance is small in absolute terms (raw probabilities
  cluster within ~1 %); UI normalises for visual clarity.

### 9.10 Ethical Considerations
- **Bias:** RACE is built from Chinese English-exam materials; the
  domain bias may not generalise to other testing contexts.
- **Accessibility:** UI ships with dark mode, keyboard navigation,
  shortcuts dialog (`?` key), and a transparency banner.
- **Academic integrity:** AI-generated content is clearly marked in
  the UI ("AI-generated" badge + an explanatory Alert).
- **Model transparency:** confidence scores shown to the user with
  per-option bars and a numeric percentage.

### 9.11 Conclusion + Future Work
- Summary of contributions (recap §9.2).
- Future work:
  - Fine-tune BERT on RACE for a fair neural comparison.
  - Train on additional MCQ datasets (CommonsenseQA, OpenBookQA) to
    test generalisation.
  - Replace the template generator with a fine-tuned T5-small.
  - Improve distractor diversity with a learned reranker.
  - Address hint R² objective mismatch with answer-relevance training.

---

## 10. UI Screenshots Checklist (for the writer to capture)

- [ ] Article Input page — **Real RACE Question** mode, with a benchmark loaded
- [ ] Article Input page — **AI-Generated** mode showing the experimental Alert
- [ ] Quiz page after Check Answer — confidence bars at four different widths,
      verdict card visible (success or failure)
- [ ] Hints page with all three accordions expanded
- [ ] Reveal Answer dialog (modal)
- [ ] Analytics dashboard top section (4 metric cards + Recharts bar chart)
- [ ] Analytics dashboard mid section (CM heatmap + Model B table)
- [ ] Dark mode + light mode side-by-side (use the toggle in the NavBar)
