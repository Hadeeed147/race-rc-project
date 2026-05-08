"""
Compose backend/metrics.json from the per-component metrics produced in
Sessions 2 + 3. Re-run any time results change.
"""

import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
BACKEND = os.path.join(ROOT, "backend")
os.makedirs(BACKEND, exist_ok=True)


def main():
    out = {"val_set_size": 8787, "option_rows": 35148}
    a = os.path.join(MODELS, "model_a_metrics.json")
    b = os.path.join(MODELS, "model_b_metrics.json")
    if os.path.exists(a):
        with open(a) as f:
            out["model_a"] = json.load(f)
    if os.path.exists(b):
        with open(b) as f:
            out["model_b"] = json.load(f)
    out_path = os.path.join(BACKEND, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
