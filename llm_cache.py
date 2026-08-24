"""
InsightFlow — LLM Response Cache
Same question + same dataset = cached response, no API call needed.
"""
import os
import json
import hashlib
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL  = int(os.getenv("CACHE_TTL_HOURS", "24"))  # hours
MAX_CACHE  = int(os.getenv("CACHE_MAX_ENTRIES", "100"))


def _cache_key(question: str, dataset_path: str, mode: str) -> str:
    """Generate a unique cache key from question + dataset + mode."""
    raw = f"{question.strip().lower()}|{dataset_path}|{mode}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def get_cached(question: str, dataset_path: str, mode: str) -> dict | None:
    """
    Return cached result if available and not expired.
    Returns None if no cache hit.
    """
    key  = _cache_key(question, dataset_path, mode)
    path = _cache_path(key)

    if not os.path.exists(path):
        return None

    try:
        data = json.loads(open(path).read())
        cached_at = datetime.fromisoformat(data["cached_at"])

        # check TTL
        if datetime.now() - cached_at > timedelta(hours=CACHE_TTL):
            os.remove(path)
            return None

        return data["result"]
    except Exception:
        return None


def set_cache(question: str, dataset_path: str, mode: str, result: dict) -> bool:
    """Save result to cache."""
    try:
        # enforce max cache size
        _evict_old_entries()

        key  = _cache_key(question, dataset_path, mode)
        path = _cache_path(key)

        # only cache serializable fields
        safe_result = {
            "final_report":        result.get("final_report", ""),
            "follow_up_questions": result.get("follow_up_questions", []),
            "trace":               result.get("trace", []),
            "question":            question,
            "mode":                mode,
            "analysis_history_tools": [
                {
                    k: v for k, v in (a.tool_result or {}).items()
                    if isinstance(v, dict)
                }
                for a in result.get("analysis_history", [])
                if hasattr(a, "tool_result")
            ],
        }

        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "question":  question,
            "dataset":   dataset_path,
            "mode":      mode,
            "result":    safe_result,
        }

        open(path, "w").write(json.dumps(cache_data, ensure_ascii=False))
        return True
    except Exception:
        return False


def _evict_old_entries():
    """Remove oldest cache files if over MAX_CACHE limit."""
    try:
        if not os.path.exists(CACHE_DIR):
            return
        files = [
            (f, os.path.getmtime(os.path.join(CACHE_DIR, f)))
            for f in os.listdir(CACHE_DIR)
            if f.endswith(".json")
        ]
        if len(files) >= MAX_CACHE:
            files.sort(key=lambda x: x[1])
            for fname, _ in files[:len(files) - MAX_CACHE + 1]:
                os.remove(os.path.join(CACHE_DIR, fname))
    except Exception:
        pass


def get_cache_stats() -> dict:
    """Return cache statistics."""
    try:
        if not os.path.exists(CACHE_DIR):
            return {"total": 0, "size_kb": 0}
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
        size  = sum(
            os.path.getsize(os.path.join(CACHE_DIR, f))
            for f in files
        )
        return {
            "total":   len(files),
            "size_kb": round(size / 1024, 1),
            "ttl_h":   CACHE_TTL,
            "max":     MAX_CACHE,
        }
    except Exception:
        return {"total": 0, "size_kb": 0}


def clear_cache() -> int:
    """Clear all cache entries. Returns count deleted."""
    try:
        if not os.path.exists(CACHE_DIR):
            return 0
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
        for f in files:
            os.remove(os.path.join(CACHE_DIR, f))
        return len(files)
    except Exception:
        return 0
