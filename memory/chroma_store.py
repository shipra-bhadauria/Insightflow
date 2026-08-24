"""
Per-user ChromaDB memory store.
Each user/tenant gets their own isolated collection.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

_client = None
CHROMA_PATH = os.getenv("CHROMA_MEMORY_PATH", "memory/chroma_db")


def _get_client():
    global _client
    if _client is None:
        try:
            import chromadb
            _client = chromadb.PersistentClient(path=CHROMA_PATH)
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
    return _client


def _get_collection(user_id: str = "default"):
    client = _get_client()
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
    collection_name = f"insightflow_{safe_id}"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"user_id": user_id}
    )


def save_analysis_chroma(
    question: str,
    final_report: str,
    confidence: float,
    rows_validated: int,
    dataset_path: str,
    user_id: str = "default",
) -> bool:
    try:
        collection = _get_collection(user_id)
        doc_id = f"{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        document = f"Question: {question}\nFinding: {final_report[:500]}"
        collection.add(
            documents=[document],
            metadatas=[{
                "question":       question,
                "final_report":   final_report[:1000],
                "confidence":     str(confidence),
                "rows_validated": str(rows_validated),
                "dataset_path":   dataset_path,
                "user_id":        user_id,
                "timestamp":      datetime.utcnow().isoformat(),
            }],
            ids=[doc_id],
        )
        return True
    except Exception as e:
        print(f"[chroma_store] save error: {e}")
        return False


def search_similar_chroma(
    query: str,
    k: int = 3,
    user_id: str = "default",
) -> list:
    try:
        collection = _get_collection(user_id)
        if collection.count() == 0:
            return []
        k_actual = min(k, collection.count())
        results  = collection.query(query_texts=[query], n_results=k_actual)
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = max(0.0, 1.0 - (dist / 2.0))
            output.append({
                "question":        meta.get("question", ""),
                "final_report":    meta.get("final_report", ""),
                "confidence":      float(meta.get("confidence", 0)),
                "rows_validated":  int(meta.get("rows_validated", 0)),
                "dataset_path":    meta.get("dataset_path", ""),
                "timestamp":       meta.get("timestamp", ""),
                "user_id":         meta.get("user_id", user_id),
                "similarity_score": round(score, 4),
            })
        output.sort(key=lambda x: x["similarity_score"], reverse=True)
        return output
    except Exception as e:
        print(f"[chroma_store] search error: {e}")
        return []


def get_user_stats(user_id: str = "default") -> dict:
    try:
        collection = _get_collection(user_id)
        return {
            "user_id":        user_id,
            "total_analyses": collection.count(),
            "collection":     f"insightflow_{user_id}",
            "storage_path":   CHROMA_PATH,
        }
    except Exception as e:
        return {"user_id": user_id, "error": str(e)}


def list_all_users() -> list:
    try:
        client      = _get_client()
        collections = client.list_collections()
        return [
            col.name.replace("insightflow_", "")
            for col in collections
            if col.name.startswith("insightflow_")
        ]
    except Exception as e:
        print(f"[chroma_store] list error: {e}")
        return []


def delete_user_memory(user_id: str) -> bool:
    try:
        client  = _get_client()
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        client.delete_collection(f"insightflow_{safe_id}")
        return True
    except Exception as e:
        print(f"[chroma_store] delete error: {e}")
        return False
