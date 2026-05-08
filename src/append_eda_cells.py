"""
One-shot helper: appends Session 3 visualization cells to notebooks/EDA.ipynb
without disturbing existing cells. Idempotent — re-running is safe (it
removes any previously appended Session 3 block before adding a fresh one).
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "notebooks", "EDA.ipynb")
MARKER = "<!-- session-3-visualizations -->"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


SESSION_3_CELLS = [
    md([f"{MARKER}\n",
        "# Session 3 — Visualizations\n",
        "Confusion matrices, model comparison, and a 2D PCA scatter of the K-Means clusters.\n"]),
    code([
        "import os, json, joblib, numpy as np, pandas as pd\n",
        "import matplotlib.pyplot as plt, seaborn as sns\n",
        "import scipy.sparse as sp\n",
        "from sklearn.decomposition import TruncatedSVD\n",
        "\n",
        "ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
        "MODELS = os.path.join(ROOT, 'models')\n",
        "DATA = os.path.join(ROOT, 'data')\n",
        "FIG_DIR = os.path.join(ROOT, 'notebooks', 'figures')\n",
        "os.makedirs(FIG_DIR, exist_ok=True)\n",
        "\n",
        "with open(os.path.join(MODELS, 'model_a_metrics.json')) as f:\n",
        "    metrics = json.load(f)\n",
        "metrics_df = pd.DataFrame(metrics)[['name', 'accuracy', 'macro_f1', 'exact_match']]\n",
        "metrics_df\n"]),
    md(["## Confusion matrices (LR / SVC / NB)"]),
    code([
        "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
        "names = ['Logistic Regression', 'LinearSVC (calibrated)', 'ComplementNB']\n",
        "for ax, name in zip(axes, names):\n",
        "    cm = next(m for m in metrics if m['name'] == name)['confusion_matrix']\n",
        "    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', cbar=False,\n",
        "                xticklabels=['pred 0', 'pred 1'],\n",
        "                yticklabels=['true 0', 'true 1'], ax=ax)\n",
        "    ax.set_title(name)\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR, 'confusion_matrices.png'), dpi=140)\n",
        "plt.show()\n"]),
    md(["## Model comparison — Macro F1 vs Exact Match"]),
    code([
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "x = np.arange(len(metrics_df)); w = 0.35\n",
        "ax.bar(x - w/2, metrics_df['macro_f1'], w, label='Macro F1', color='#6aa6ff')\n",
        "ax.bar(x + w/2, metrics_df['exact_match'], w, label='Exact Match', color='#59c285')\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(metrics_df['name'], rotation=15, ha='right')\n",
        "ax.set_ylim(0, 0.65)\n",
        "ax.legend(loc='upper left')\n",
        "ax.set_title('Model A — Macro F1 vs Exact Match (val)')\n",
        "for i, (f1, em) in enumerate(zip(metrics_df['macro_f1'], metrics_df['exact_match'])):\n",
        "    ax.text(i - w/2, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=9)\n",
        "    ax.text(i + w/2, em + 0.01, f'{em:.3f}', ha='center', fontsize=9)\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR, 'model_comparison.png'), dpi=140)\n",
        "plt.show()\n"]),
    md(["## 2D PCA of TF-IDF features, coloured by K-Means cluster (5k sample)"]),
    code([
        "X_train = sp.load_npz(os.path.join(DATA, 'X_train.npz'))\n",
        "rng = np.random.RandomState(0)\n",
        "idx = rng.choice(X_train.shape[0], size=5000, replace=False)\n",
        "X_sub = X_train[idx]\n",
        "\n",
        "km = joblib.load(os.path.join(MODELS, 'kmeans.pkl'))\n",
        "clusters = km.predict(X_sub)\n",
        "\n",
        "svd = TruncatedSVD(n_components=2, random_state=42)\n",
        "X2 = svd.fit_transform(X_sub)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(7, 5))\n",
        "scatter = ax.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='viridis', s=6, alpha=0.6)\n",
        "ax.set_title('TF-IDF (TruncatedSVD 2D) coloured by K-Means cluster')\n",
        "ax.set_xlabel('component 1'); ax.set_ylabel('component 2')\n",
        "plt.colorbar(scatter, ax=ax, label='cluster')\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR, 'kmeans_pca.png'), dpi=140)\n",
        "plt.show()\n"]),
]


def main():
    with open(NB_PATH) as f:
        nb = json.load(f)
    cells = nb.get("cells", [])

    # Strip any previously-appended Session 3 block
    keep = []
    in_block = False
    for c in cells:
        text = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
        if MARKER in text:
            in_block = True
            continue
        if in_block:
            # Stop trimming once we encounter a code/markdown cell that doesn't
            # look like it belongs to the appended block. We just trim the rest
            # of the file because we always append at the end.
            continue
        keep.append(c)

    nb["cells"] = keep + SESSION_3_CELLS
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Appended {len(SESSION_3_CELLS)} Session-3 cells -> {NB_PATH}")
    print("Open Jupyter, pick kernel 'Python (race-rc)', and Run All.")


if __name__ == "__main__":
    main()
