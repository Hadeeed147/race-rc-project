"""
Session 5 — Neural baselines on GPU.

Two comparisons against the classical-ML pipeline:

(A) BERT-LR verification baseline
    bert-base-uncased as a frozen feature extractor.
    For each option row, build "[CLS] article [SEP] question [SEP] option [SEP]",
    truncate to 256 tokens, take the [CLS] hidden state, train a single
    LogisticRegression head on TRAIN, evaluate on VAL.
    Saves embeddings to data/bert_emb_{split}.npy so you can re-run the LR
    head without re-embedding.

(B) T5-small question-generation reference
    Prompts: "generate question: <article>" on 200 val rows.
    Compares against the gold val questions with BLEU + ROUGE-L + METEOR,
    and also against our template generator on the same 200 rows.

GPU expected: RTX 3050 (6 GB). Batch size = 8 by default. If OOM, set
BATCH_SIZE = 4 below.

Outputs: models/baselines_metrics.json + a printed comparison table.
"""

import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

BATCH_SIZE = 8                       # drop to 4 if OOM
MAX_LEN = 256
N_TRAIN_OPTIONS = 50_000             # subsample option rows for embedding
N_T5_SAMPLES = 200
T5_MAX_INPUT = 384
T5_MAX_OUTPUT = 32

OPTION_LABELS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------
# (A) BERT-LR verification baseline
# --------------------------------------------------------------------------
def reshape_to_option_level(df):
    long = df.melt(
        id_vars=["id", "article", "question", "answer"],
        value_vars=OPTION_LABELS,
        var_name="option_label",
        value_name="option_text",
    )
    long["y"] = (long["answer"] == long["option_label"]).astype(np.int8)
    long["text"] = (
        long["article"].astype(str).fillna("") + " [SEP] " +
        long["question"].astype(str).fillna("") + " [SEP] " +
        long["option_text"].astype(str).fillna("")
    )
    long["__ord"] = long["option_label"].map({l: i for i, l in enumerate(OPTION_LABELS)})
    long = long.sort_values(["id", "__ord"]).drop(columns="__ord").reset_index(drop=True)
    return long


def embed_split(model, tokenizer, texts, device, desc=""):
    """Run BERT and grab the [CLS] hidden state for every text. Returns (n,768)."""
    import torch
    out = np.empty((len(texts), 768), dtype=np.float32)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state[:, 0, :]   # [CLS]
            out[i:i + len(batch)] = h.detach().cpu().numpy()
            if (i // BATCH_SIZE) % 200 == 0:
                el = time.time() - t0
                rate = (i + len(batch)) / max(0.1, el)
                eta = (len(texts) - i - len(batch)) / max(1, rate)
                print(f"  [{desc}] {i + len(batch):>6}/{len(texts):>6}  "
                      f"{rate:5.1f} rows/s  eta {eta/60:5.1f} min")
    return out


def run_bert_lr_baseline():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    print("=" * 60)
    print("(A) BERT-LR baseline")
    print("=" * 60)
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading bert-base-uncased ...")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    print("Reading splits ...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv"))
    train_long = reshape_to_option_level(train_df)
    val_long = reshape_to_option_level(val_df)
    if N_TRAIN_OPTIONS and len(train_long) > N_TRAIN_OPTIONS:
        train_long = train_long.sample(N_TRAIN_OPTIONS, random_state=42).reset_index(drop=True)
    print(f"  train option rows: {len(train_long)},  val option rows: {len(val_long)}")

    # Embed (with caching)
    train_emb_path = os.path.join(DATA_DIR, "bert_emb_train.npy")
    val_emb_path = os.path.join(DATA_DIR, "bert_emb_val.npy")

    t0 = time.time()
    if os.path.exists(train_emb_path):
        print(f"  Found cached {train_emb_path}, loading ...")
        X_train_emb = np.load(train_emb_path)
        if X_train_emb.shape[0] != len(train_long):
            print("  Cache size mismatch — re-embedding.")
            X_train_emb = embed_split(model, tok, train_long["text"].tolist(), device, "train")
            np.save(train_emb_path, X_train_emb)
    else:
        X_train_emb = embed_split(model, tok, train_long["text"].tolist(), device, "train")
        np.save(train_emb_path, X_train_emb)

    if os.path.exists(val_emb_path):
        print(f"  Found cached {val_emb_path}, loading ...")
        X_val_emb = np.load(val_emb_path)
        if X_val_emb.shape[0] != len(val_long):
            print("  Cache size mismatch — re-embedding.")
            X_val_emb = embed_split(model, tok, val_long["text"].tolist(), device, "val")
            np.save(val_emb_path, X_val_emb)
    else:
        X_val_emb = embed_split(model, tok, val_long["text"].tolist(), device, "val")
        np.save(val_emb_path, X_val_emb)
    embedding_time_s = time.time() - t0

    # Free GPU memory before running anything else
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nTraining LR head on BERT [CLS] features ...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1)
    t0 = time.time()
    lr.fit(X_train_emb, train_long["y"].values)
    inference_time_s = time.time() - t0
    probs = lr.predict_proba(X_val_emb)[:, 1]

    # Per-question argmax across A/B/C/D
    from sklearn.metrics import classification_report, confusion_matrix
    df = pd.DataFrame({
        "qid": val_long["id"].values,
        "opt": val_long["option_label"].values,
        "p": probs,
        "y": val_long["y"].values,
    })
    pred = df.loc[df.groupby("qid")["p"].idxmax(), ["qid", "opt"]].rename(columns={"opt": "pred"})
    gold = df.loc[df["y"] == 1, ["qid", "opt"]].rename(columns={"opt": "gold"})
    merged = pred.merge(gold, on="qid", how="inner")
    em = float((merged["pred"] == merged["gold"]).mean())

    df["y_pred"] = 0
    df.loc[df.groupby("qid")["p"].idxmax(), "y_pred"] = 1
    acc = accuracy_score(df["y"].values, df["y_pred"].values)
    macro_f1 = f1_score(df["y"].values, df["y_pred"].values, average="macro")
    cm = confusion_matrix(df["y"].values, df["y_pred"].values).tolist()

    print(f"  BERT-LR  Accuracy={acc:.4f}  Macro F1={macro_f1:.4f}  EM={em:.4f}")

    # Save the LR head so we can reuse on test in step 4
    import joblib
    joblib.dump(lr, os.path.join(MODELS_DIR, "bert_lr_head.pkl"))

    return {
        "accuracy": acc, "macro_f1": macro_f1, "exact_match": em,
        "confusion_matrix": cm,
        "embedding_time_s": round(embedding_time_s, 2),
        "lr_fit_time_s": round(inference_time_s, 2),
        "n_train_options": int(len(train_long)),
        "n_val_options": int(len(val_long)),
    }


# --------------------------------------------------------------------------
# (B) T5-small question-generation reference
# --------------------------------------------------------------------------
def run_t5_baseline():
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try: nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try: nltk.download(pkg, quiet=True)
            except Exception: pass

    print("\n" + "=" * 60)
    print("(B) T5-small question generation")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    model.eval()

    val = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv")) \
            .sample(N_T5_SAMPLES, random_state=42).reset_index(drop=True)

    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    refs, hyps_t5 = [], []
    rougel_t5, meteor_t5 = [], []

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(val), BATCH_SIZE):
            batch = val.iloc[i:i + BATCH_SIZE]
            inputs = tok(
                ["generate question: " + str(a)[:2000] for a in batch["article"]],
                return_tensors="pt", padding=True, truncation=True, max_length=T5_MAX_INPUT,
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=T5_MAX_OUTPUT,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
            decoded = tok.batch_decode(outputs, skip_special_tokens=True)
            for q_gen, q_gold in zip(decoded, batch["question"].astype(str).tolist()):
                ref_tok = q_gold.lower().split()
                hyp_tok = q_gen.lower().split()
                refs.append([ref_tok])
                hyps_t5.append(hyp_tok)
                rougel_t5.append(rouge.score(q_gold, q_gen)["rougeL"].fmeasure)
                try:
                    meteor_t5.append(float(meteor_score([ref_tok], hyp_tok)))
                except Exception:
                    pass
    inference_time_s = time.time() - t0
    bleu_t5 = float(corpus_bleu(refs, hyps_t5, smoothing_function=smooth))

    # Compare to our template generator on the same 200 samples
    print("\nComparing to template generator on the same samples ...")
    import joblib
    from model_a_generate import generate_questions
    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    try:
        ranker = joblib.load(os.path.join(MODELS_DIR, "question_ranker.pkl"))
    except Exception:
        ranker = None
    rougel_ours, meteor_ours, refs2, hyps_ours = [], [], [], []
    for _, row in val.iterrows():
        gens = generate_questions(row["article"], vec, ranker, top_k=1)
        q_gen = gens[0]["question"] if gens else ""
        q_gold = str(row["question"])
        ref_tok = q_gold.lower().split()
        hyp_tok = q_gen.lower().split()
        refs2.append([ref_tok])
        hyps_ours.append(hyp_tok)
        rougel_ours.append(rouge.score(q_gold, q_gen)["rougeL"].fmeasure)
        try: meteor_ours.append(float(meteor_score([ref_tok], hyp_tok)))
        except Exception: pass
    bleu_ours = float(corpus_bleu(refs2, hyps_ours, smoothing_function=smooth))

    # Free GPU memory
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "t5_small": {
            "bleu": bleu_t5,
            "rouge_l": float(np.mean(rougel_t5)) if rougel_t5 else 0.0,
            "meteor": float(np.mean(meteor_t5)) if meteor_t5 else 0.0,
            "n_samples": int(len(val)),
            "inference_time_s": round(inference_time_s, 2),
        },
        "template_generator": {
            "bleu": bleu_ours,
            "rouge_l": float(np.mean(rougel_ours)) if rougel_ours else 0.0,
            "meteor": float(np.mean(meteor_ours)) if meteor_ours else 0.0,
            "n_samples": int(len(val)),
        },
    }


# --------------------------------------------------------------------------
def main():
    out = {"gpu_used": "RTX 3050 6GB", "batch_size": BATCH_SIZE,
           "max_len": MAX_LEN, "n_train_options": N_TRAIN_OPTIONS}

    # (A) BERT
    bert = run_bert_lr_baseline()
    out["bert_lr"] = bert

    # (B) T5
    t5 = run_t5_baseline()
    out["t5_small"] = t5["t5_small"]
    out["template_generator"] = t5["template_generator"]

    # Pretty summary
    print("\n" + "=" * 60)
    print("Final neural-baseline comparison (val)")
    print("=" * 60)
    print(f"  BERT-LR (ours)        Acc={bert['accuracy']:.4f}  "
          f"F1={bert['macro_f1']:.4f}  EM={bert['exact_match']:.4f}")
    print(f"  T5-small generation   BLEU={t5['t5_small']['bleu']:.4f}  "
          f"ROUGE-L={t5['t5_small']['rouge_l']:.4f}  METEOR={t5['t5_small']['meteor']:.4f}")
    print(f"  Template generator    BLEU={t5['template_generator']['bleu']:.4f}  "
          f"ROUGE-L={t5['template_generator']['rouge_l']:.4f}  METEOR={t5['template_generator']['meteor']:.4f}")

    out_path = os.path.join(MODELS_DIR, "baselines_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
