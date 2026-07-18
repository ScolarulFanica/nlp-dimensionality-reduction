"""
Stage 2a — PCA on the TF-IDF matrix.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "20news"
DATA_DIR = PROJECT_ROOT / "data" / CORPUS
FIG_DIR = PROJECT_ROOT / "figures" / CORPUS
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[pca] corpus = {CORPUS}  data = {DATA_DIR}  figures = {FIG_DIR}")

K_VALUES = [10, 25, 50, 100, 200]
RANDOM_STATE = 42


def load_data():
    X = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    y = np.load(DATA_DIR / "labels.npy")
    with open(DATA_DIR / "target_names.json") as f:
        names = json.load(f)
    return X, y, names


def evaluate_classifier(X_proj, y) -> tuple[float, float]:
    """5-fold cross-validated logistic regression in the projected space."""
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_proj, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(scores.mean()), float(scores.std())


def run_pca(X_dense, y, k_values):
    """Run centered PCA at several k and record metrics."""
    print("\nPCA (centered) — eigendecomposition of the covariance matrix")
    print(f"  input shape: {X_dense.shape}")
    print(f"  testing k = {k_values}")

    # Fit PCA at the largest k once; for smaller k take the first columns.
    k_max = max(k_values)
    t0 = time.time()
    pca = PCA(n_components=k_max, random_state=RANDOM_STATE)
    full_proj = pca.fit_transform(X_dense)
    fit_time = time.time() - t0
    print(f"  fit time at k={k_max}: {fit_time:.2f}s")

    results = []
    for k in k_values:
        proj = full_proj[:, :k]
        var_explained = float(pca.explained_variance_ratio_[:k].sum())
        acc_mean, acc_std = evaluate_classifier(proj, y)
        results.append({
            "k": k,
            "explained_variance_ratio": var_explained,
            "classification_accuracy_mean": acc_mean,
            "classification_accuracy_std": acc_std,
        })
        print(f"  k={k:>3}  explained_var={var_explained:.4f}  "
              f"accuracy={acc_mean:.4f} ± {acc_std:.4f}")

    return pca, full_proj, results, fit_time


def run_truncated_svd(X_sparse, y, k_values):
    """Run TruncatedSVD (uncentered) for comparison"""
    print("\nTruncatedSVD (uncentered, sparse) — for comparison with PCA")

    k_max = max(k_values)
    t0 = time.time()
    svd = TruncatedSVD(n_components=k_max, random_state=RANDOM_STATE)
    full_proj = svd.fit_transform(X_sparse)
    fit_time = time.time() - t0
    print(f"  fit time at k={k_max}: {fit_time:.2f}s")

    results = []
    for k in k_values:
        proj = full_proj[:, :k]
        var_explained = float(svd.explained_variance_ratio_[:k].sum())
        acc_mean, acc_std = evaluate_classifier(proj, y)
        results.append({
            "k": k,
            "explained_variance_ratio": var_explained,
            "classification_accuracy_mean": acc_mean,
            "classification_accuracy_std": acc_std,
        })
        print(f"  k={k:>3}  explained_var={var_explained:.4f}  "
              f"accuracy={acc_mean:.4f} ± {acc_std:.4f}")

    return svd, full_proj, results, fit_time


def plot_explained_variance(pca_results, svd_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    pca_k = [r["k"] for r in pca_results]
    pca_var = [r["explained_variance_ratio"] for r in pca_results]
    svd_var = [r["explained_variance_ratio"] for r in svd_results]
    ax.plot(pca_k, pca_var, "o-", label="PCA (centered)", linewidth=2, markersize=7)
    ax.plot(pca_k, svd_var, "s--", label="TruncatedSVD (uncentered)", linewidth=2, markersize=7)
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("Explained variance vs. number of components")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_explained_variance.png", dpi=140)
    plt.close(fig)


def plot_classification_accuracy(pca_results, svd_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    pca_k = [r["k"] for r in pca_results]
    pca_acc = [r["classification_accuracy_mean"] for r in pca_results]
    pca_err = [r["classification_accuracy_std"] for r in pca_results]
    svd_acc = [r["classification_accuracy_mean"] for r in svd_results]
    svd_err = [r["classification_accuracy_std"] for r in svd_results]
    ax.errorbar(pca_k, pca_acc, yerr=pca_err, fmt="o-", capsize=4, label="PCA (centered)")
    ax.errorbar(pca_k, svd_acc, yerr=svd_err, fmt="s--", capsize=4, label="TruncatedSVD (uncentered)")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("5-fold CV classification accuracy")
    ax.set_title("Classification accuracy in the reduced space")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_classification_accuracy.png", dpi=140)
    plt.close(fig)


def main():
    X, y, names = load_data()
    print(f"Loaded TF-IDF matrix: {X.shape}, {len(names)} classes")

    # PCA needs a dense matrix
    X_dense = X.toarray()

    pca, pca_proj, pca_results, pca_time = run_pca(X_dense, y, K_VALUES)
    svd, svd_proj, svd_results, svd_time = run_truncated_svd(X, y, K_VALUES)

    plot_explained_variance(pca_results, svd_results)
    plot_classification_accuracy(pca_results, svd_results)

    # Save the projection at the largest k for downstream use
    np.save(DATA_DIR / "pca_projections.npy", pca_proj)
    np.save(DATA_DIR / "pca_components.npy", pca.components_)

    out = {
        "pca": {
            "fit_time_seconds": pca_time,
            "results_per_k": pca_results,
        },
        "truncated_svd_uncentered": {
            "fit_time_seconds": svd_time,
            "results_per_k": svd_results,
        },
        "k_values_tested": K_VALUES,
    }
    with open(DATA_DIR / "pca_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFigures saved to {FIG_DIR}/")
    print(f"Results saved to {DATA_DIR / 'pca_results.json'}")


if __name__ == "__main__":
    main()