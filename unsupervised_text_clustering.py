#!/usr/bin/env python3
"""
unsupervised_text_clustering.py

Improved version:
- Batch SBERT encoding
- Optional UMAP dimensionality reduction
- Parallel TM similarity computation
- Noise reassignment in refinement
- Sparse TF-IDF support
"""

import argparse
import concurrent.futures
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# sklearn
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# HDBSCAN
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# UMAP
try:
    import umap

    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# spaCy for TM-sim approximation
import spacy


# ---------- Utilities ----------
def read_text_files(folder: Path, encoding="utf-8") -> List[Tuple[Path, str]]:
    files = sorted([p for p in folder.rglob("*.txt") if p.is_file()])
    docs = []
    for p in files:
        try:
            text = p.read_text(encoding=encoding)
        except Exception:
            text = p.read_text(encoding="latin-1")
        docs.append((p, text))
    return docs


# ---------- Embedding ----------
class EmbeddingModel:
    def __init__(
        self, method="sbert", sbert_model_name="all-MiniLM-L6-v2", batch_size=64
    ):
        self.method = method
        self.sbert_model_name = sbert_model_name
        self.batch_size = batch_size
        self.tf = None
        self.model = None
        if method == "sbert":
            if not SBERT_AVAILABLE:
                raise RuntimeError(
                    "Install sentence-transformers or use method='tfidf'"
                )
            self.model = SentenceTransformer(sbert_model_name)
        elif method == "tfidf":
            self.tf = TfidfVectorizer(max_features=20000, stop_words="english")
        else:
            raise ValueError(f"Unknown embedding method '{method}'")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self.method == "sbert":
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        else:
            return self.tf.fit_transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.method == "sbert":
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            return self.tf.transform(texts)


# ---------- TM-sim (parallel) ----------
def load_spacy_model(name="en_core_web_sm"):
    try:
        return spacy.load(name)
    except Exception:
        raise RuntimeError(f"Run: python -m spacy download {name}")


def extract_topic_sequence(text: str, nlp) -> List[str]:
    doc = nlp(text)
    topics = []
    seen = set()
    for sent in doc.sents:
        local = []
        for ent in sent.ents:
            token = ent.text.strip().lower()
            if token:
                local.append(token)
        for nc in sent.noun_chunks:
            token = nc.text.strip().lower()
            if token:
                local.append(token)
        for t in local:
            if t not in seen:
                seen.add(t)
                topics.append(t)
    return topics


def lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    la, lb = len(seq_a), len(seq_b)
    if la == 0 or lb == 0:
        return 0
    prev = [0] * (lb + 1)
    for i in range(la - 1, -1, -1):
        curr = [0] * (lb + 1)
        for j in range(lb - 1, -1, -1):
            if seq_a[i] == seq_b[j]:
                curr[j] = 1 + prev[j + 1]
            else:
                curr[j] = max(prev[j], curr[j + 1])
        prev = curr
    return prev[0]


def tm_similarity_from_sequences(a_seq: List[str], b_seq: List[str]) -> float:
    la, lb = len(a_seq), len(b_seq)
    if la == 0 or lb == 0:
        return 0.0
    lcs = lcs_length(a_seq, b_seq)
    return (2.0 * lcs) / (la + lb)


def compute_tm_similarity_matrix(texts: List[str], nlp, n_jobs=4) -> np.ndarray:
    n = len(texts)
    seqs = [extract_topic_sequence(t, nlp) for t in texts]
    mat = np.zeros((n, n), dtype=float)

    def compute_row(i):
        row = np.zeros(n)
        for j in range(i, n):
            s = tm_similarity_from_sequences(seqs[i], seqs[j])
            row[j] = s
        return i, row

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for i, row in tqdm(
            executor.map(compute_row, range(n)), total=n, desc="TM-sim pairs"
        ):
            mat[i, i:] = row[i:]
            mat[i:, i] = row[i:]

    return mat


# ---------- Clustering ----------
def cluster_dbscan(
    X: np.ndarray, eps=0.5, min_samples=3, metric="cosine"
) -> np.ndarray:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(X)
    return labels


def cluster_hdbscan(
    X: np.ndarray, min_cluster_size=5, metric="euclidean"
) -> np.ndarray:
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("HDBSCAN not installed")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = clusterer.fit_predict(X)
    return labels


# ---------- Refinement ----------
def refine_clusters(
    X: np.ndarray, labels: np.ndarray, threshold_noise=0.7
) -> np.ndarray:
    """Reassign points to nearest cluster centroid, including noise points"""
    unique_labels = [l for l in np.unique(labels) if l != -1]
    if len(unique_labels) == 0:
        return labels
    centroids = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
    sim = cosine_similarity(X, centroids)
    new_labels = np.full_like(labels, fill_value=-1)
    for idx in range(len(labels)):
        best = sim[idx].argmax()
        if labels[idx] != -1 or sim[idx][best] > threshold_noise:
            new_labels[idx] = unique_labels[best]
    return new_labels


# ---------- Metrics ----------
def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) <= 1 or len(unique) == len(labels):
        return float("nan")
    try:
        return silhouette_score(X, labels, metric="cosine")
    except Exception:
        return float("nan")


def compute_metrics(
    X: np.ndarray, labels: np.ndarray, tm_matrix: np.ndarray = None
) -> Dict[str, Any]:
    metrics = {}
    metrics["silhouette_cosine"] = safe_silhouette(X, labels)
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
    except Exception:
        metrics["davies_bouldin"] = float("nan")
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    except Exception:
        metrics["calinski_harabasz"] = float("nan")
    if tm_matrix is not None:
        n = len(labels)
        intra_sims = []
        inter_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    intra_sims.append(tm_matrix[i, j])
                else:
                    inter_sims.append(tm_matrix[i, j])
        metrics["tm_mean_intra"] = float(np.mean(intra_sims)) if intra_sims else 0.0
        metrics["tm_mean_inter"] = float(np.mean(inter_sims)) if inter_sims else 0.0
        metrics["tm_diff_intra_minus_inter"] = (
            metrics["tm_mean_intra"] - metrics["tm_mean_inter"]
        )
        metrics["tm_ratio_intra_over_inter"] = (
            (metrics["tm_mean_intra"] / (metrics["tm_mean_inter"] + 1e-12))
            if metrics["tm_mean_inter"] > 0
            else float("inf")
        )
    return metrics


# ---------- Save ----------
def save_clusters_csv(
    out_path: Path, docs: List[Tuple[Path, str]], labels: np.ndarray, snippet_chars=200
):
    rows = []
    for (path, text), label in zip(docs, labels):
        rows.append(
            {
                "filename": str(path),
                "cluster": int(label),
                "snippet": text[:snippet_chars].replace("\n", " "),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# ---------- Main pipeline ----------
def run_pipeline(
    folder: Path,
    embedding_method="sbert",
    sbert_model="all-MiniLM-L6-v2",
    clustering_method="dbscan",
    eps=0.5,
    min_samples=3,
    min_cluster_size=5,
    compute_tm=True,
    spacy_model="en_core_web_sm",
    refine=True,
    umap_dim: int = 50,
    output_folder: Path = Path("."),
    random_state=42,
):

    t0 = time.time()
    output_folder.mkdir(parents=True, exist_ok=True)
    docs = read_text_files(folder)
    if not docs:
        raise RuntimeError(f"No .txt files found in {folder}")

    paths, texts = zip(*docs)
    texts = list(texts)

    # Embeddings
    logging.info("Building embeddings with method=%s", embedding_method)
    emb_model = EmbeddingModel(method=embedding_method, sbert_model_name=sbert_model)
    X = emb_model.fit_transform(texts)

    # Optional UMAP dimensionality reduction
    if UMAP_AVAILABLE and umap_dim < X.shape[1]:
        logging.info("Reducing embedding dimensionality to %d via UMAP", umap_dim)
        X = umap.UMAP(
            n_components=umap_dim, metric="cosine", random_state=random_state
        ).fit_transform(X)

    # Clustering
    logging.info("Clustering method=%s", clustering_method)
    if clustering_method == "dbscan":
        labels = cluster_dbscan(X, eps=eps, min_samples=min_samples)
    elif clustering_method == "hdbscan":
        labels = cluster_hdbscan(X, min_cluster_size=min_cluster_size)
    else:
        raise ValueError("Unsupported clustering method")

    # Refinement
    if refine:
        logging.info("Refining clusters via centroid reassignment")
        labels = refine_clusters(X, labels)

    # TM-sim
    tm_matrix = None
    if compute_tm:
        logging.info("Loading spaCy model '%s' for TM-sim approximation", spacy_model)
        nlp = load_spacy_model(spacy_model)
        tm_matrix = compute_tm_similarity_matrix(texts, nlp, n_jobs=8)
        np.save(output_folder / "tm_similarity_matrix.npy", tm_matrix)

    # Metrics
    metrics = compute_metrics(X, labels, tm_matrix=tm_matrix)
    metrics["n_docs"] = len(texts)
    metrics["clustering_method"] = clustering_method
    metrics["embedding_method"] = embedding_method
    metrics["n_clusters_found"] = int(len(set(labels)) - (1 if -1 in labels else 0))

    # Save outputs
    csv_path = output_folder / "clusters.csv"
    save_clusters_csv(csv_path, docs, labels)
    json_path = output_folder / "metrics.json"
    metrics["runtime_seconds"] = time.time() - t0
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logging.info("Saved clusters to %s and metrics to %s", csv_path, json_path)
    return {
        "clusters_csv": str(csv_path),
        "metrics_json": str(json_path),
        "metrics": metrics,
        "labels": labels,
        "tm_matrix_saved": (
            str(output_folder / "tm_similarity_matrix.npy")
            if tm_matrix is not None
            else None
        ),
    }


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Fully unsupervised semantic clustering for OCR'd .txt files"
    )
    parser.add_argument(
        "folder", type=Path, default=r"/home/lucapolenta/Desktop/Dataset_OCR"
    )
    parser.add_argument("--embedding", choices=["sbert", "tfidf"], default="sbert")
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--clustering", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=3)
    parser.add_argument("--min_cluster_size", type=int, default=5)
    parser.add_argument("--no_tm", action="store_true")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--umap_dim", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("clustering_out"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    res = run_pipeline(
        folder=args.folder,
        embedding_method=args.embedding,
        sbert_model=args.sbert_model,
        clustering_method=args.clustering,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        compute_tm=not args.no_tm,
        spacy_model=args.spacy_model,
        refine=not args.no_refine,
        umap_dim=args.umap_dim,
        output_folder=args.out,
    )
    print("Done. Outputs:")
    print(json.dumps(res["metrics"], indent=2))
    print("CSV:", res["clusters_csv"])
    print("Metrics JSON:", res["metrics_json"])
    if res["tm_matrix_saved"]:
        print("TM similarity matrix saved at:", res["tm_matrix_saved"])


if __name__ == "__main__":
    main()
