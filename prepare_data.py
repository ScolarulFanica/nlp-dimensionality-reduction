"""
Stage 1 — Data preparation pipeline (English 20 Newsgroups + Romanian MOROCO).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from synthetic_corpus import build_corpus as build_synthetic_corpus

PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42

EN_CATEGORIES = [
    "sci.space", "sci.med", "sci.electronics",
    "rec.sport.hockey", "talk.politics.guns", "comp.graphics",
]

RO_CATEGORIES = ["culture", "finance", "politics", "science", "sports", "tech"]

MOROCO_DOCS_PER_CATEGORY = 900  

TFIDF_PARAMS_COMMON = dict(
    min_df=2,
    max_df=0.95,
    max_features=10_000,
    sublinear_tf=True,
    norm="l2",
    lowercase=True,
)

RO_DIACRITIC_MAP = str.maketrans({
    "ş": "ș", "Ş": "Ș",
    "ţ": "ț", "Ţ": "Ț",
})

RO_STOPWORDS = sorted(set("""
a abia acea aceasta această aceea acei acel acela acelasi acele acelea acest
acesta aceste acestea acestei acestia acestui acolo acord acum adica ai aia
aibă aici al ala ale alea alt alta altceva altcineva alte altfel alti altul
am anume apoi ar are as asa asemenea asta astazi astea astfel astăzi asupra
atare atat atata atatea atatia ati atunci au avea aveam aveau avem aveti avut
azi aş aţi b ba bine bună c ca cam cand care careia carora caruia cat catre
ce cea ceea cei cel cele celor ceva chiar ci cinci cind cine cineva cit cita
cite citeva citi citiva conform contra cu cui cum cumva curând când cât câte
că căci cărei căror cărui către d da daca dacă dar dat dată de deasupra deci
decit deja deoarece departe desi despre deşi din dintr dintre doar doi doilea
două drept dupa după e ea ei el ele era eram este eu eşti face fara fata fel
fi fie fiecare fii fim fiu fiţi foarte fost fără h i ia iar ieri ii il imi in
inainte inapoi inca incit insa intre isi iti î îi îl îmi împotriva în înainte
înaintea încotro încât între întrucât îţi l la le li lor lui m ma mai mare
mea mei mele meu mi mie mine mod mult multa multe multi multă mulţi mâine mă
n ne nevoie ni nici niciodata nicăieri nimeni nimic niste nişte noastre
noastră noi nostri nostru nouă noştri nu numai o opt or ori oricare orice
oricine oricum oricând oricât oriunde p parca patra patru patrulea pe pentru
peste pic pina poate pot prea prima primul prin printr puţin puţina puţină
până r rău rând s sa sai sale sau se si și sint sintem spre sub sunt suntem
sunteţi sus s-a să săi sînt sîntem t ta tale te ti timp tine toata toate toti
totul totusi totuşi toţi trei treia treilea tu tuturor tăi tău u ul ului un
una unde undeva unei uneia unele uneori unii unor unora unu unui unuia unul
v va vi voastre voastră voi vom vor vostru vouă voştri vreme vreo vreun vă z
zece zero zi zice ăla ălea ăsta ăstea ăştia ș ț ă â î șa ție mie ție mâine
ieri astăzi alaltăieri ședinței ședințele ședinței
""".split()))

_NEWSGROUP_HEADER = re.compile(r"^[\w-]+:\s.*$", flags=re.MULTILINE)
_QUOTED_LINE = re.compile(r"^>.*$", flags=re.MULTILINE)
_EMAIL = re.compile(r"\S+@\S+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTI_WS = re.compile(r"\s+")

_NON_ALPHA_EN = re.compile(r"[^a-zA-Z\s]")
_NON_ALPHA_RO = re.compile(r"[^a-zA-ZăâîșțĂÂÎȘȚ\s]")
_MOROCO_NE = re.compile(r"\$NE\$")


def clean_document(text: str, language: str) -> str:
    if language == "ro":
        text = text.translate(RO_DIACRITIC_MAP)
        text = _MOROCO_NE.sub(" ", text)
        non_alpha = _NON_ALPHA_RO
    else:
        non_alpha = _NON_ALPHA_EN
    text = _NEWSGROUP_HEADER.sub("", text)
    text = _QUOTED_LINE.sub("", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = non_alpha.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text

def load_20news():
    try:
        from sklearn.datasets import fetch_20newsgroups
        print("Fetching 20 Newsgroups (English)...")
        bundle = fetch_20newsgroups(
            subset="all", categories=EN_CATEGORIES,
            remove=("headers", "footers", "quotes"),
            random_state=RANDOM_STATE, shuffle=True,
        )
        return bundle.data, list(bundle.target), list(bundle.target_names), "real"
    except Exception as e:
        print(f"  -> 20news unavailable ({type(e).__name__}); using synthetic fallback.")
        docs, labels, names = build_synthetic_corpus(docs_per_category=180, seed=RANDOM_STATE)
        return docs, labels, names, "synthetic"


def _download_moroco_files(target_dir: Path):
    import urllib.request

    base_url = "https://raw.githubusercontent.com/butnaruandrei/MOROCO/master/MOROCO/preprocessed/train"
    files = ["samples.txt", "category_labels.txt", "dialect_labels.txt"]

    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        dest = target_dir / fname
        if dest.exists() and dest.stat().st_size > 1000:
            continue  # already cached
        url = f"{base_url}/{fname}"
        print(f"  downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  ERROR: failed to download {url}")
            print(f"  {type(e).__name__}: {e}")
            print(f"  Manual fallback: clone https://github.com/butnaruandrei/MOROCO")
            print(f"  and copy MOROCO/training/ to {target_dir}/")
            sys.exit(1)


def _parse_moroco_file(path: Path) -> dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sample_id = parts[0]
            content = " ".join(parts[1:]).strip()
            if content:
                out[sample_id] = content
    return out


def load_moroco():
    """Category label mapping:
        1 => culture, 2 => finance, 3 => politics,
        4 => science, 5 => sports,  6 => tech
    """
    print("Fetching MOROCO (Romanian) directly from GitHub...")
    cache_dir = PROJECT_ROOT / "moroco_data" / "preprocessed_train"
    _download_moroco_files(cache_dir)

    samples = _parse_moroco_file(cache_dir / "samples.txt")
    cat_labels = _parse_moroco_file(cache_dir / "category_labels.txt")
    print(f"  parsed {len(samples):,} samples, {len(cat_labels):,} labels")

    aligned = []
    for sid, text in samples.items():
        if sid not in cat_labels:
            continue
        try:
            label = int(cat_labels[sid]) - 1  
        except ValueError:
            continue
        if 0 <= label < len(RO_CATEGORIES):
            aligned.append((text, label))

    texts = [t for t, _ in aligned]
    labels = [l for _, l in aligned]
    print(f"  aligned: {len(texts):,} documents")

    rng = np.random.default_rng(RANDOM_STATE)
    by_class: dict[int, list[int]] = {c: [] for c in range(len(RO_CATEGORIES))}
    for i, lab in enumerate(labels):
        by_class[lab].append(i)

    keep_idx = []
    for c, idxs in by_class.items():
        arr = np.array(idxs)
        rng.shuffle(arr)
        keep_idx.extend(arr[:MOROCO_DOCS_PER_CATEGORY].tolist())
    rng.shuffle(keep_idx)

    docs = [texts[i] for i in keep_idx]
    sub_labels = [labels[i] for i in keep_idx]
    print(f"  -> {len(docs):,} documents loaded (stratified, "
          f"up to {MOROCO_DOCS_PER_CATEGORY} per category)")
    return docs, sub_labels, RO_CATEGORIES, "real"



def filter_empty(docs, labels, min_tokens=5):
    keep = [i for i, d in enumerate(docs) if len(d.split()) >= min_tokens]
    kept_docs = [docs[i] for i in keep]
    kept_labels = labels[keep]
    dropped = len(docs) - len(kept_docs)
    if dropped:
        print(f"  -> dropped {dropped} documents under {min_tokens} tokens")
    return kept_docs, kept_labels


def vectorize(docs, language):
    print("Building TF-IDF matrix...")
    params = dict(TFIDF_PARAMS_COMMON)
    params["stop_words"] = RO_STOPWORDS if language == "ro" else "english"
    vectorizer = TfidfVectorizer(**params)
    X = vectorizer.fit_transform(docs)
    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"  -> shape: {X.shape[0]:,} docs x {X.shape[1]:,} terms")
    print(f"  -> non-zero entries: {X.nnz:,}  (density = {density:.4%})")
    return X, vectorizer


def collect_stats(X, labels, target_names, vectorizer, cleaned_docs, source, corpus, language):
    n_docs, vocab_size = X.shape
    nnz = int(X.nnz)
    density = nnz / (n_docs * vocab_size)

    doc_lengths = np.array([len(d.split()) for d in cleaned_docs])
    label_counts = np.bincount(labels, minlength=len(target_names))
    class_distribution = {target_names[i]: int(label_counts[i]) for i in range(len(target_names))}

    df = np.asarray((X > 0).sum(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()
    top_idx = np.argsort(-df)[:15]
    most_common_terms = [(str(vocab[i]), int(df[i])) for i in top_idx]

    return {
        "corpus_id": corpus,
        "corpus_source": source,
        "language": language,
        "categories": list(target_names),
        "n_documents": int(n_docs),
        "vocabulary_size": int(vocab_size),
        "matrix_nonzero_entries": nnz,
        "matrix_density": float(density),
        "matrix_sparsity": float(1.0 - density),
        "mean_tokens_per_doc": float(doc_lengths.mean()),
        "median_tokens_per_doc": float(np.median(doc_lengths)),
        "min_tokens_per_doc": int(doc_lengths.min()),
        "max_tokens_per_doc": int(doc_lengths.max()),
        "class_distribution": class_distribution,
        "tfidf_params": {k: v for k, v in TFIDF_PARAMS_COMMON.items()},
        "preprocessing": {
            "language": language,
            "diacritic_normalization": language == "ro",
            "removed": ["headers", "URLs", "emails", "non-alpha"] +
                       (["$NE$ placeholders"] if language == "ro" else []),
            "stopword_list": ("Romanian (embedded)" if language == "ro" else "sklearn english"),
            "stemming": False,
        },
        "top_15_most_frequent_terms": most_common_terms,
    }


def save_artifacts(out_dir, X, labels, target_names, vectorizer, cleaned_docs, stats):
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "tfidf_matrix.npz", X)
    np.save(out_dir / "labels.npy", labels)
    with open(out_dir / "target_names.json", "w") as f:
        json.dump(list(target_names), f, indent=2, ensure_ascii=False)
    with open(out_dir / "vocabulary.json", "w") as f:
        json.dump(vectorizer.get_feature_names_out().tolist(), f, ensure_ascii=False)
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    meta = [
        {"doc_id": i, "label": int(labels[i]),
         "label_name": target_names[labels[i]],
         "n_tokens": len(cleaned_docs[i].split()),
         "preview": cleaned_docs[i][:160]}
        for i in range(len(cleaned_docs))
    ]
    with open(out_dir / "documents_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"\nArtefacts saved to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["20news", "moroco"], default="20news",
                    help="Which corpus to prepare (default: 20news)")
    args = ap.parse_args()

    if args.corpus == "20news":
        raw_docs, labels, target_names, source = load_20news()
        language = "en"
    else:
        raw_docs, labels, target_names, source = load_moroco()
        language = "ro"

    labels = np.asarray(labels)
    print(f"  -> {len(raw_docs):,} documents loaded ({source})")

    print(f"Cleaning documents (language = {language})...")
    cleaned = [clean_document(d, language) for d in raw_docs]
    cleaned, labels = filter_empty(cleaned, labels)

    X, vectorizer = vectorize(cleaned, language)
    stats = collect_stats(X, labels, target_names, vectorizer, cleaned, source, args.corpus, language)

    out_dir = PROJECT_ROOT / "data" / args.corpus
    save_artifacts(out_dir, X, labels, target_names, vectorizer, cleaned, stats)

    print(f"\n=== {args.corpus} summary ===")
    print(f"  Language:         {language}")
    print(f"  Source:           {source}")
    print(f"  Documents:        {stats['n_documents']:,}")
    print(f"  Vocabulary size:  {stats['vocabulary_size']:,}")
    print(f"  Matrix density:   {stats['matrix_density']:.4%}")
    print(f"  Mean tokens/doc:  {stats['mean_tokens_per_doc']:.1f}")
    print(f"  Class balance:")
    for name, count in stats["class_distribution"].items():
        print(f"    {name:<25} {count:>5}")


if __name__ == "__main__":
    main()