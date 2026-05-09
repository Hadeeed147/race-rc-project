"""
Session 5 — Final test-set evaluation. The TEST split has been untouched
since Session 1; this is the ONLY time we read it.

Runs:
  (1) Tuned ensemble (LR + SVC + NB) on test         — Acc / F1 / EM / CM / per-class report
  (2) Distractors + hints on 200-sample test subset  — BLEU / ROUGE / METEOR / P@K / token-PRF
  (3) BERT-LR head (Session 5 step 1) on test        — Acc / F1 / EM     [skipped if no head]

Saves everything to models/final_test_metrics.json + three publication
figures under notebooks/figures/.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIG_DIR = os.path.join(ROOT_DIR, "notebooks", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_MODEL_B_SAMPLES = 200


# -------------------- helpers --------------------
def load_split(split):
    X = sp.load_npz(os.path.join(DATA_DIR, f"X_{split}.npz"))
    y = np.load(os.path.join(DATA_DIR, f"y_{split}.npy"))
    qid = np.load(os.path.join(DATA_DIR, f"qid_{split}.npy"), allow_pickle=True)
    opt = np.load(os.path.join(DATA_DIR, f"opt_{split}.npy"), allow_pickle=True)
    return X, y, qid, opt


def question_argmax_em(probs_pos, qids, opts, y):
    df = pd.DataFrame({"qid": qids, "opt": opts, "p": probs_pos, "y": y})
    pred = df.loc[df.groupby("qid")["p"].idxmax(), ["qid", "opt"]].rename(columns={"opt": "pred"})
    gold = df.loc[df["y"] == 1, ["qid", "opt"]].rename(columns={"opt": "gold"})
    merged = pred.merge(gold, on="qid", how="inner")
    return float((merged["pred"] == merged["gold"]).mean()), len(merged)


def per_option_y_pred(probs_pos, qids):
    df = pd.DataFrame({"qid": qids, "p": probs_pos})
    df["y_pred"] = 0
    df.loc[df.groupby("qid")["p"].idxmax(), "y_pred"] = 1
    return df["y_pred"].values


# -------------------- (1) ensemble on test --------------------
def evaluate_ensemble_on_test():
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    print("=" * 60)
    print("(1) Tuned ensemble on TEST")
    print("=" * 60)
    bundle = joblib.load(os.path.join(MODELS_DIR, "ensemble.pkl"))
    models = bundle["models"]
    weights = bundle["weights"]
    print(f"  ensemble weights: {weights}")
    X_te, y_te, qid_te, opt_te = load_split("test")
    print(f"  TEST: {X_te.shape}")

    t0 = time.time()
    probs = np.zeros(X_te.shape[0])
    for m, w in zip(models, weights):
        probs += w * m.predict_proba(X_te)[:, 1]
    yp = per_option_y_pred(probs, qid_te)
    em, n_q = question_argmax_em(probs, qid_te, opt_te, y_te)
    acc = accuracy_score(y_te, yp)
    macro_f1 = f1_score(y_te, yp, average="macro")
    cm = confusion_matrix(y_te, yp)
    rep = classification_report(y_te, yp, digits=4,
                                target_names=["wrong (0)", "correct (1)"], output_dict=True)
    print(f"  inference: {time.time()-t0:.1f}s")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Exact Match (per-question, n={n_q}): {em:.4f}")
    print(classification_report(y_te, yp, digits=4,
                                target_names=["wrong (0)", "correct (1)"]))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "exact_match": em,
        "n_questions": int(n_q),
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
    }


# -------------------- (2) Model B on test subset --------------------
def evaluate_model_b_on_test():
    print("\n" + "=" * 60)
    print(f"(2) Model B on TEST ({N_MODEL_B_SAMPLES}-sample subset)")
    print("=" * 60)
    from model_b import extract_distractors, generate_hints, split_sentences, tokenize, _sentence_features
    from sklearn.metrics import r2_score
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try: nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try: nltk.download(pkg, quiet=True)
            except Exception: pass

    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    scorer = joblib.load(os.path.join(MODELS_DIR, "hint_scorer.pkl"))

    test = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv")) \
            .sample(N_MODEL_B_SAMPLES, random_state=42).reset_index(drop=True)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    refs, hyps = [], []
    rouge1, rouge2, rougeL, meteors = [], [], [], []
    p_list, r_list, f_list = [], [], []
    n_skip_d = 0
    for _, row in test.iterrows():
        article = row["article"]
        correct = str(row[row["answer"]])
        gold_d = [str(row[L]) for L in ["A", "B", "C", "D"] if L != row["answer"]]
        gens = extract_distractors(article, correct, vec, top_k=3)
        if not gens:
            n_skip_d += 1
            continue
        # token PRF
        gtoks, atoks = set(), set()
        for g in gens: gtoks |= set(tokenize(g))
        for a in gold_d: atoks |= set(tokenize(a))
        if gtoks and atoks:
            tp = len(gtoks & atoks)
            p = tp / max(1, len(gtoks)); r = tp / max(1, len(atoks))
            f = (2*p*r/(p+r)) if (p+r) else 0.0
            p_list.append(p); r_list.append(r); f_list.append(f)
        # rouge / meteor / bleu refs
        for g in gens:
            best_r1 = best_r2 = best_rL = 0.0
            for ref in gold_d:
                sc = rouge.score(ref, g)
                best_r1 = max(best_r1, sc["rouge1"].fmeasure)
                best_r2 = max(best_r2, sc["rouge2"].fmeasure)
                best_rL = max(best_rL, sc["rougeL"].fmeasure)
            rouge1.append(best_r1); rouge2.append(best_r2); rougeL.append(best_rL)
            try:
                hyp_tok = tokenize(g)
                ref_toks = [tokenize(d) for d in gold_d]
                if hyp_tok and all(ref_toks):
                    meteors.append(float(meteor_score(ref_toks, hyp_tok)))
            except Exception:
                pass
            refs.append([tokenize(d) for d in gold_d]); hyps.append(tokenize(g))
    bleu = float(corpus_bleu(refs, hyps, smoothing_function=smooth)) if refs else 0.0
    distractors_metrics = {
        "n_questions": int(len(test) - n_skip_d),
        "bleu": bleu,
        "rouge1_f": float(np.mean(rouge1)) if rouge1 else 0.0,
        "rouge2_f": float(np.mean(rouge2)) if rouge2 else 0.0,
        "rougeL_f": float(np.mean(rougeL)) if rougeL else 0.0,
        "meteor":   float(np.mean(meteors)) if meteors else 0.0,
        "precision": float(np.mean(p_list)) if p_list else 0.0,
        "recall":    float(np.mean(r_list)) if r_list else 0.0,
        "f1":        float(np.mean(f_list)) if f_list else 0.0,
    }
    print("  distractors:")
    for k, v in distractors_metrics.items():
        if isinstance(v, float): print(f"    {k:<12} {v:.4f}")
        else: print(f"    {k:<12} {v}")

    # hints
    p1_list, p3_list, r2_pred, r2_true = [], [], [], []
    n_skip_h = 0
    for _, row in test.iterrows():
        sents = split_sentences(row["article"])
        if len(sents) < 3:
            n_skip_h += 1; continue
        ans_tokens = set(tokenize(str(row[row["answer"]])))
        if not ans_tokens:
            n_skip_h += 1; continue
        overlaps = []
        for s in sents:
            st = set(tokenize(s))
            ov = len(st & ans_tokens) / max(1, len(ans_tokens))
            overlaps.append(ov)
        gold_idx = int(np.argmax(overlaps))
        gold_sent = sents[gold_idx]
        hints = generate_hints(row["article"], row["question"], vec, scorer)
        p1_list.append(1.0 if hints and hints[-1].strip() == gold_sent.strip() else 0.0)
        p3_list.append(1.0 if any(h.strip() == gold_sent.strip() for h in hints) else 0.0)
        feats = _sentence_features(sents, row["question"], vec)
        pred = scorer.predict_proba(feats)[:, 1]
        r2_pred.extend(pred.tolist()); r2_true.extend(overlaps)
    r2 = float(r2_score(r2_true, r2_pred)) if r2_pred else 0.0
    hints_metrics = {
        "n_questions": int(len(test) - n_skip_h),
        "precision_at_1": float(np.mean(p1_list)) if p1_list else 0.0,
        "precision_at_3": float(np.mean(p3_list)) if p3_list else 0.0,
        "r2_scorer": r2,
    }
    print("  hints:")
    for k, v in hints_metrics.items():
        if isinstance(v, float): print(f"    {k:<14} {v:.4f}")
        else: print(f"    {k:<14} {v}")

    return {"distractors": distractors_metrics, "hints": hints_metrics,
            "n_samples": N_MODEL_B_SAMPLES}


# -------------------- (3) BERT-LR head on test (optional) --------------------
def evaluate_bert_lr_on_test():
    print("\n" + "=" * 60)
    print("(3) BERT-LR head on TEST (optional)")
    print("=" * 60)
    head_path = os.path.join(MODELS_DIR, "bert_lr_head.pkl")
    if not os.path.exists(head_path):
        print("  no models/bert_lr_head.pkl -> skipping. Run src/baselines.py first.")
        return None

    test_emb_path = os.path.join(DATA_DIR, "bert_emb_test.npy")
    if not os.path.exists(test_emb_path):
        print("  no data/bert_emb_test.npy -> embedding test split now.")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except Exception as e:
            print(f"  torch/transformers not available: {e}; skipping.")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device); model.eval()
        df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
        long = df.melt(
            id_vars=["id", "article", "question", "answer"],
            value_vars=["A", "B", "C", "D"],
            var_name="option_label", value_name="option_text",
        )
        long["__ord"] = long["option_label"].map({"A":0,"B":1,"C":2,"D":3})
        long = long.sort_values(["id", "__ord"]).drop(columns="__ord").reset_index(drop=True)
        long["text"] = (
            long["article"].astype(str).fillna("") + " [SEP] " +
            long["question"].astype(str).fillna("") + " [SEP] " +
            long["option_text"].astype(str).fillna("")
        )
        BATCH = 8; MAX_LEN = 256
        emb = np.empty((len(long), 768), dtype=np.float32)
        import torch as _t
        with _t.no_grad():
            for i in range(0, len(long), BATCH):
                batch = long["text"].tolist()[i:i+BATCH]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors="pt").to(device)
                emb[i:i+len(batch)] = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        np.save(test_emb_path, emb)
        del model, tok
        import gc; gc.collect()
        if _t.cuda.is_available(): _t.cuda.empty_cache()

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    head = joblib.load(head_path)
    X_emb = np.load(test_emb_path)
    df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    long = df.melt(
        id_vars=["id", "article", "question", "answer"],
        value_vars=["A","B","C","D"],
        var_name="option_label", value_name="option_text",
    )
    long["__ord"] = long["option_label"].map({"A":0,"B":1,"C":2,"D":3})
    long = long.sort_values(["id","__ord"]).drop(columns="__ord").reset_index(drop=True)
    long["y"] = (long["answer"] == long["option_label"]).astype(int)
    p = head.predict_proba(X_emb)[:, 1]
    yp = per_option_y_pred(p, long["id"].values)
    em, n_q = question_argmax_em(p, long["id"].values, long["option_label"].values, long["y"].values)
    acc = accuracy_score(long["y"].values, yp)
    macro_f1 = f1_score(long["y"].values, yp, average="macro")
    cm = confusion_matrix(long["y"].values, yp)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Exact Match (per-question, n={n_q}): {em:.4f}")
    return {
        "accuracy": float(acc), "macro_f1": float(macro_f1), "exact_match": em,
        "n_questions": int(n_q), "confusion_matrix": cm.tolist(),
    }


# -------------------- figures --------------------
def make_figures(ensemble_test, model_b_test, bert_test):
    sns.set_theme(style="whitegrid", context="talk")
    palette = {"primary": "#4f46e5", "accent": "#f59e0b", "muted": "#94a3b8",
               "good": "#16a34a", "bad": "#dc2626"}

    # 1. Confusion matrix
    cm = np.array(ensemble_test["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["pred 0", "pred 1"],
                yticklabels=["true 0", "true 1"], annot_kws={"size": 14})
    ax.set_title("Ensemble — Test Confusion Matrix", fontsize=14)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "final_confusion_matrix.png")
    plt.savefig(p1, dpi=140); plt.close(fig)
    print(f"  saved {p1}")

    # 2. Comparison bar
    rows = [("Random baseline", 0.25, None)]
    if bert_test:
        rows.append(("BERT-LR (frozen + LR head)", bert_test["exact_match"],
                     bert_test["macro_f1"]))
    rows.append(("Ours — Tuned Ensemble", ensemble_test["exact_match"],
                 ensemble_test["macro_f1"]))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(rows)); w = 0.36
    ems = [r[1] for r in rows]
    f1s = [r[2] if r[2] is not None else 0 for r in rows]
    ax.bar(x - w/2, ems, w, label="Exact Match", color=palette["primary"])
    ax.bar(x + w/2, f1s, w, label="Macro F1", color=palette["accent"])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=12, ha="right")
    ax.set_ylim(0, 0.75)
    for i, (em, f1) in enumerate(zip(ems, f1s)):
        ax.text(i - w/2, em + 0.01, f"{em:.3f}", ha="center", fontsize=11)
        if f1: ax.text(i + w/2, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=11)
    ax.legend(loc="upper left")
    ax.set_title("Test-set comparison — Exact Match + Macro F1")
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "final_model_comparison.png")
    plt.savefig(p2, dpi=140); plt.close(fig)
    print(f"  saved {p2}")

    # 3. Per-class P/R/F1
    rep = ensemble_test["classification_report"]
    classes = ["wrong (0)", "correct (1)"]
    metrics = ["precision", "recall", "f1-score"]
    matrix = np.array([[rep[c][m] for m in metrics] for c in classes])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Purples", cbar=False, ax=ax,
                xticklabels=metrics, yticklabels=classes, annot_kws={"size": 13})
    ax.set_title("Ensemble — Test Per-Class P / R / F1")
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "final_metric_breakdown.png")
    plt.savefig(p3, dpi=140); plt.close(fig)
    print(f"  saved {p3}")
    return [p1, p2, p3]


# -------------------- driver --------------------
def main():
    out = {}
    out["ensemble_test"] = evaluate_ensemble_on_test()
    out["model_b_test"] = evaluate_model_b_on_test()
    out["bert_lr_test"] = evaluate_bert_lr_on_test()

    # figures
    print("\n" + "=" * 60)
    print("Figures")
    print("=" * 60)
    figs = make_figures(out["ensemble_test"], out["model_b_test"], out["bert_lr_test"])
    out["figures"] = figs

    out_path = os.path.join(MODELS_DIR, "final_test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("FINAL TEST-SET SUMMARY")
    print("=" * 60)
    e = out["ensemble_test"]
    b = out["bert_lr_test"]
    d = out["model_b_test"]["distractors"]
    h = out["model_b_test"]["hints"]
    print(f"  Ours · Tuned ensemble    Acc={e['accuracy']:.4f}  F1={e['macro_f1']:.4f}  EM={e['exact_match']:.4f}")
    if b:
        print(f"  BERT-LR (frozen + LR)    Acc={b['accuracy']:.4f}  F1={b['macro_f1']:.4f}  EM={b['exact_match']:.4f}")
    print(f"  Distractors (test, n=200) BLEU={d['bleu']:.4f}  ROUGE-L={d['rougeL_f']:.4f}  METEOR={d['meteor']:.4f}  F1={d['f1']:.4f}")
    print(f"  Hints (test, n=200)       P@1={h['precision_at_1']:.4f}  P@3={h['precision_at_3']:.4f}  R²={h['r2_scorer']:.4f}")
    print(f"  Random baseline EM = 0.25  · Ours +{(e['exact_match']-0.25):+.3f}")


if __name__ == "__main__":
    main()
