"""
Stage 2b — Latent Semantic Analysis (LSA).
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
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "20news"
DATA_DIR = PROJECT_ROOT / "data" / CORPUS
FIG_DIR = PROJECT_ROOT / "figures" / CORPUS
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[lsa] corpus = {CORPUS}  data = {DATA_DIR}  figures = {FIG_DIR}")

K_VALUES = [10, 25, 50, 100, 200]
RANDOM_STATE = 42


def load_data():
    X = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    y = np.load(DATA_DIR / "labels.npy")
    with open(DATA_DIR / "target_names.json") as f:
        names = json.load(f)
    with open(DATA_DIR / "vocabulary.json") as f:
        vocab = json.load(f)
    return X, y, names, vocab


def evaluate_classification(X_proj, y) -> tuple[float, float]:
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_proj, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def mean_average_precision_retrieval(X_proj, y) -> float:
    X_norm = normalize(X_proj, norm="l2")
    sims = X_norm @ X_norm.T
    np.fill_diagonal(sims, -np.inf) 

    n = X_proj.shape[0]
    aps = []
    for i in range(n):
        order = np.argsort(-sims[i])
        relevant = (y[order] == y[i]).astype(np.float64)
        cum_rel = np.cumsum(relevant)
        ranks = np.arange(1, n) 
        relevant = relevant[: n - 1]
        cum_rel = cum_rel[: n - 1]
        precisions_at_k = cum_rel / ranks
        n_relevant = relevant.sum()
        if n_relevant > 0:
            ap = float((precisions_at_k * relevant).sum() / n_relevant)
            aps.append(ap)
    return float(np.mean(aps))


def run_lsa(X, y, k_values):
    print("\nLSA — TruncatedSVD on TF-IDF (uncentered, sparse)")
    k_max = max(k_values)
    t0 = time.time()
    svd = TruncatedSVD(n_components=k_max, random_state=RANDOM_STATE, algorithm="randomized", n_iter=7)
    doc_proj = svd.fit_transform(X)
    fit_time = time.time() - t0
    print(f"  fit time at k={k_max}: {fit_time:.2f}s")

    sigma = svd.singular_values_ 
    term_emb_unscaled = svd.components_.T 
    term_emb = term_emb_unscaled * sigma[np.newaxis, :]

    results = []
    for k in k_values:
        proj_k = doc_proj[:, :k]
        var_explained = float(svd.explained_variance_ratio_[:k].sum())
        acc_mean, acc_std = evaluate_classification(proj_k, y)
        map_score = mean_average_precision_retrieval(proj_k, y)
        results.append({
            "k": k,
            "explained_variance_ratio": var_explained,
            "classification_accuracy_mean": acc_mean,
            "classification_accuracy_std": acc_std,
            "retrieval_mAP": map_score,
        })
        print(f"  k={k:>3}  var={var_explained:.4f}  "
              f"acc={acc_mean:.4f}±{acc_std:.4f}  mAP={map_score:.4f}")

    print("\nBaseline (raw TF-IDF, no reduction):")
    raw_acc_mean, raw_acc_std = evaluate_classification(X, y)
    raw_map = mean_average_precision_retrieval(X.toarray(), y)
    print(f"  acc={raw_acc_mean:.4f}±{raw_acc_std:.4f}  mAP={raw_map:.4f}")
    baseline = {
        "classification_accuracy_mean": raw_acc_mean,
        "classification_accuracy_std": raw_acc_std,
        "retrieval_mAP": raw_map,
    }

    return svd, doc_proj, term_emb, sigma, results, baseline, fit_time


def top_terms_per_topic(svd, vocab, n_topics: int = 8, n_terms: int = 8):
    components = svd.components_ 
    out = []
    for t in range(n_topics):
        weights = components[t]
        idx = np.argsort(-np.abs(weights))[:n_terms]
        terms = [(vocab[i], float(weights[i])) for i in idx]
        out.append({"topic": t, "top_terms": terms})
    return out


def plot_metrics_vs_k(results, baseline):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ks = [r["k"] for r in results]

    ax = axes[0]
    ax.plot(ks, [r["explained_variance_ratio"] for r in results], "o-", color="#1f77b4", linewidth=2)
    ax.set_xlabel("k (number of latent dimensions)")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("Variance captured")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(ks, [r["classification_accuracy_mean"] for r in results],
                yerr=[r["classification_accuracy_std"] for r in results],
                fmt="o-", capsize=4, color="#2ca02c", linewidth=2, label="LSA")
    ax.axhline(baseline["classification_accuracy_mean"], color="gray",
               linestyle="--", label=f"Raw TF-IDF baseline ({baseline['classification_accuracy_mean']:.3f})")
    ax.set_xlabel("k")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Document classification")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[2]
    ax.plot(ks, [r["retrieval_mAP"] for r in results], "o-", color="#d62728", linewidth=2, label="LSA")
    ax.axhline(baseline["retrieval_mAP"], color="gray", linestyle="--",
               label=f"Raw TF-IDF baseline ({baseline['retrieval_mAP']:.3f})")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Average Precision")
    ax.set_title("Document retrieval")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("LSA performance as a function of latent dimensionality k", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lsa_metrics_vs_k.png", dpi=140)
    plt.close(fig)


def plot_top_terms_per_topic(topics):
    n = len(topics)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    axes = axes.flatten()
    for i, t in enumerate(topics):
        ax = axes[i]
        terms = [term for term, _ in t["top_terms"]]
        weights = [w for _, w in t["top_terms"]]
        colors = ["#2ca02c" if w > 0 else "#d62728" for w in weights]
        ax.barh(terms[::-1], [abs(w) for w in weights[::-1]], color=colors[::-1])
        ax.set_title(f"Topic {t['topic']}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("|weight|", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Top terms per LSA topic (first 8 latent dimensions)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lsa_top_terms_per_topic.png", dpi=140)
    plt.close(fig)


def main():
    X, y, names, vocab = load_data()
    print(f"Loaded TF-IDF matrix: {X.shape}, {len(names)} classes")

    svd, doc_proj, term_emb, sigma, results, baseline, fit_time = run_lsa(X, y, K_VALUES)

    topics = top_terms_per_topic(svd, vocab, n_topics=8, n_terms=8)
    print("\nTop 5 terms in the first 4 LSA topics:")
    for t in topics[:4]:
        terms_str = ", ".join(f"{term}({w:+.2f})" for term, w in t["top_terms"][:5])
        print(f"  Topic {t['topic']}: {terms_str}")

    plot_metrics_vs_k(results, baseline)
    plot_top_terms_per_topic(topics)

    np.save(DATA_DIR / "lsa_doc_embeddings.npy", doc_proj)
    np.save(DATA_DIR / "lsa_term_embeddings.npy", term_emb)
    np.save(DATA_DIR / "lsa_singular_values.npy", sigma)

    out = {
        "method": "TruncatedSVD on TF-IDF (uncentered)",
        "fit_time_seconds": fit_time,
        "k_values_tested": K_VALUES,
        "results_per_k": results,
        "raw_tfidf_baseline": baseline,
        "top_terms_per_topic": topics,
        "first_20_singular_values": sigma[:20].tolist(),
    }
    with open(DATA_DIR / "lsa_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()