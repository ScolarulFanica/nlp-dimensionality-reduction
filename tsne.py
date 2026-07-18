"""
Stage 2c — t-SNE visualization.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "20news"
DATA_DIR = PROJECT_ROOT / "data" / CORPUS
FIG_DIR = PROJECT_ROOT / "figures" / CORPUS
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[tsne] corpus = {CORPUS}  data = {DATA_DIR}  figures = {FIG_DIR}")

PERPLEXITIES = [5, 30, 50]
LSA_PRECONDITIONING_DIMS = 50  
RANDOM_STATE = 42

PALETTE = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#ff7f0e", "#17becf"]


def load_data():
    X = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    y = np.load(DATA_DIR / "labels.npy")
    with open(DATA_DIR / "target_names.json") as f:
        names = json.load(f)
    return X, y, names


def run_tsne(X_high_dim, perplexity: int):
    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=RANDOM_STATE,
        metric="cosine",
    )
    Y = tsne.fit_transform(X_high_dim)
    elapsed = time.time() - t0
    kl = float(tsne.kl_divergence_)
    return Y, elapsed, kl


def scatter_2d(ax, Y, labels, names, title):
    for cls in range(len(names)):
        mask = labels == cls
        ax.scatter(Y[mask, 0], Y[mask, 1], s=10, alpha=0.7,
                   color=PALETTE[cls], label=names[cls], edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def plot_perplexity_sweep(results_by_perp, labels, names):
    fig, axes = plt.subplots(1, len(PERPLEXITIES), figsize=(15, 5.0))
    for ax, perp in zip(axes, PERPLEXITIES):
        Y = results_by_perp[perp]["embedding"]
        sil = results_by_perp[perp]["silhouette"]
        kl = results_by_perp[perp]["kl_divergence"]
        title = f"perplexity = {perp}\nsilhouette = {sil:.3f}, KL = {kl:.3f}"
        scatter_2d(ax, Y, labels, names, title)
    # Single legend on the right
    handles, lbls = axes[-1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=True)
    fig.suptitle("t-SNE perplexity sweep (run on top of 50-D LSA embeddings)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(FIG_DIR / "tsne_perplexity_sweep.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(pca_2d, lsa_2d, tsne_2d, labels, names, sils):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    panels = [
        ("PCA (2D, centered)", pca_2d, sils["pca_2d"]),
        ("LSA (2D, TruncatedSVD)", lsa_2d, sils["lsa_2d"]),
        ("t-SNE (perplexity=30, on 50-D LSA)", tsne_2d, sils["tsne"]),
    ]
    for ax, (name, Y, sil) in zip(axes, panels):
        scatter_2d(ax, Y, labels, names, f"{name}\nsilhouette = {sil:.3f}")
    handles, lbls = axes[-1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=True)
    fig.suptitle("Linear (PCA, LSA) vs. non-linear (t-SNE) projections to 2D",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(FIG_DIR / "dimred_2d_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    X, y, names = load_data()
    print(f"Loaded TF-IDF matrix: {X.shape}, {len(names)} classes")

    print(f"\nPre-reducing to {LSA_PRECONDITIONING_DIMS} LSA dimensions...")
    pre_svd = TruncatedSVD(
        n_components=LSA_PRECONDITIONING_DIMS,
        random_state=RANDOM_STATE,
        algorithm="randomized",
        n_iter=7,
    )
    X_lsa50 = pre_svd.fit_transform(X)

    print("\nt-SNE perplexity sweep:")
    results_by_perp = {}
    for perp in PERPLEXITIES:
        Y, elapsed, kl = run_tsne(X_lsa50, perp)
        sil = float(silhouette_score(Y, y))
        print(f"  perplexity={perp:>3}  time={elapsed:.1f}s  KL={kl:.3f}  silhouette={sil:.3f}")
        results_by_perp[perp] = {
            "embedding": Y,
            "elapsed_seconds": elapsed,
            "kl_divergence": kl,
            "silhouette": sil,
        }

    print("\n2D linear baselines for comparison...")
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X.toarray())
    lsa_2d = TruncatedSVD(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

    sils = {
        "pca_2d": float(silhouette_score(pca_2d, y)),
        "lsa_2d": float(silhouette_score(lsa_2d, y)),
        "tsne": results_by_perp[30]["silhouette"],
    }
    print(f"  PCA 2D silhouette: {sils['pca_2d']:.3f}")
    print(f"  LSA 2D silhouette: {sils['lsa_2d']:.3f}")
    print(f"  t-SNE silhouette:  {sils['tsne']:.3f} (perp=30)")

    plot_perplexity_sweep(results_by_perp, y, names)
    plot_method_comparison(pca_2d, lsa_2d, results_by_perp[30]["embedding"], y, names, sils)

    out = {
        "perplexity_sweep": [
            {
                "perplexity": p,
                "elapsed_seconds": results_by_perp[p]["elapsed_seconds"],
                "kl_divergence": results_by_perp[p]["kl_divergence"],
                "silhouette_score": results_by_perp[p]["silhouette"],
            }
            for p in PERPLEXITIES
        ],
        "linear_baseline_silhouettes": sils,
        "pre_reduction": {
            "method": "TruncatedSVD (LSA)",
            "dimensions": LSA_PRECONDITIONING_DIMS,
        },
        "tsne_settings": {
            "n_components": 2,
            "init": "pca",
            "max_iter": 1000,
            "metric": "cosine",
            "learning_rate": "auto",
            "random_state": RANDOM_STATE,
        },
    }
    with open(DATA_DIR / "tsne_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()