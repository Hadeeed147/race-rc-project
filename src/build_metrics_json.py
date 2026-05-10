"""
Compose backend/metrics.json from the per-component metrics produced in
Sessions 2-5. Re-run any time results change.

Includes:
  - val-set metrics for every Model A variant + Model B
  - test_set: tuned-ensemble + Model B test metrics + BERT-LR (optional)
  - hyperparameter_sweep summary (best params + ensemble-weight table)
  - baselines: BERT-LR + T5-small (if available)
"""

import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
BACKEND = os.path.join(ROOT, "backend")
os.makedirs(BACKEND, exist_ok=True)


def _load(name):
    p = os.path.join(MODELS, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    out = {"val_set_size": 8787, "option_rows": 35148}

    a = _load("model_a_metrics.json")
    if a is not None:
        out["model_a"] = a

    b = _load("model_b_metrics.json")
    if b is not None:
        out["model_b"] = b

    sweep = _load("hyperparameter_sweep.json")
    if sweep is not None:
        out["hyperparameter_sweep"] = sweep

    test = _load("final_test_metrics.json")
    if test is not None:
        out["test_set"] = {
            "ensemble":  test.get("ensemble_test"),
            "bert_lr":   test.get("bert_lr_test"),
            "model_b":   test.get("model_b_test"),
            "n_test_questions": 8787,
            "n_test_options": 35148,
        }

    # Model A generation (Family 2)
    a_gen = _load("model_a_gen_metrics.json")
    if a_gen:
        out["model_a_gen"] = a_gen

    # Unsupervised / Semi-supervised (Family 3)
    unsupervised = _load("unsupervised_metrics.json")
    if unsupervised:
        out["unsupervised"] = unsupervised

    baselines = _load("baselines_metrics.json")
    if baselines is not None:
        out["baselines"] = baselines

    out_path = os.path.join(BACKEND, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    keys = list(out.keys())
    print(f"  top-level keys: {keys}")


if __name__ == "__main__":
    main()
