import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import faiss
from datetime import datetime
from dotenv import load_dotenv

from memory.embeddings import get_embedding, EMBEDDING_DIM

load_dotenv()

# where the FAISS index and metadata are saved on disk
INDEX_PATH    = "memory/faiss_index.bin"
METADATA_PATH = "memory/faiss_metadata.json"


def _load_index() -> tuple:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index    = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        # create a new flat L2 index
        index    = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []
    return index, metadata


def _save_index(index, metadata: list):
    os.makedirs("memory", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def save_analysis(
    question: str,
    final_report: str,
    confidence: float,
    rows_validated: int,
    dataset_path: str,
) -> bool:

    index, metadata = _load_index()

    # build text to embed — question + key parts of finding
    text_to_embed = f"Question: {question}\nFinding: {final_report[:500]}"
    vector = get_embedding(text_to_embed)

    # convert to numpy float32 — required by FAISS
    vector_np = np.array([vector], dtype=np.float32)

    # add vector to index
    index.add(vector_np)

    # store metadata alongside
    metadata.append({
        "id":             index.ntotal - 1,
        "question":       question,
        "final_report":   final_report,
        "confidence":     confidence,
        "rows_validated": rows_validated,
        "dataset_path":   dataset_path,
        "timestamp":      datetime.now().isoformat(),
    })

    _save_index(index, metadata)
    return True


def search_similar(
    query: str,
    k: int = 3,
) -> list[dict]:

    index, metadata = _load_index()

    if index.ntotal == 0:
        return []

    # embed the query
    vector    = get_embedding(query)
    vector_np = np.array([vector], dtype=np.float32)

    # search — returns distances and indices of k nearest vectors
    k_actual        = min(k, index.ntotal)
    distances, indices = index.search(vector_np, k_actual)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx].copy()
        entry["similarity_score"] = float(1 / (1 + dist))
        results.append(entry)

    # sort by similarity — highest first
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results


def get_memory_stats() -> dict:
    index, metadata = _load_index()
    return {
        "total_analyses": index.ntotal,
        "index_path":     INDEX_PATH,
        "metadata_path":  METADATA_PATH,
    }


if __name__ == "__main__":
    print("=== FAISS Store Test ===\n")

    # save a test analysis
    print("Saving test analysis...")
    save_analysis(
        question       = "What is the average revenue per region?",
        final_report   = "Asia leads at £1,940,644 average revenue — 1.76× Sub-Saharan Africa.",
        confidence     = 0.95,
        rows_validated = 100,
        dataset_path   = "data/sales.csv",
    )

    save_analysis(
        question       = "Are there any anomalies in Total Revenue?",
        final_report   = "3 outliers found above £5,127,029 — values of £5.99M, £5.51M, £5.39M.",
        confidence     = 0.92,
        rows_validated = 100,
        dataset_path   = "data/sales.csv",
    )

    stats = get_memory_stats()
    print(f"Total analyses in memory: {stats['total_analyses']}")

    # search for similar
    print("\nSearching for similar analyses...")
    results = search_similar("What is the mean revenue by region?", k=2)

    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Question:   {r['question']}")
        print(f"  Similarity: {r['similarity_score']:.4f}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  Timestamp:  {r['timestamp']}")