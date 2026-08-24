import sys
import os
import time
import logging
import uuid
from collections import defaultdict
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import shutil
from dotenv import load_dotenv

from state                import new_state
from tools.quality        import run_quality_report
from tools.detect_columns import detect_columns
from graph.workflow       import workflow
from memory.faiss_store   import save_analysis, search_similar, get_memory_stats

load_dotenv()

# ── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("insightflow")

# ── Metrics ──────────────────────────────────────────
_metrics = {
    "total_requests":      0,
    "successful":          0,
    "failed":              0,
    "avg_latency_ms":      0.0,
    "latencies":           [],
    "requests_by_tenant":  defaultdict(int),
    "started_at":          datetime.utcnow().isoformat(),
}

# ── Rate Limiter ────────────────────────────────────────
RATE_LIMIT     = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
_rate_counters = defaultdict(list)

def _check_rate_limit(tenant_id: str):
    now    = time.time()
    window = [t for t in _rate_counters[tenant_id] if now - t < 60]
    if len(window) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT} req/min for tenant '{tenant_id}'"
        )
    window.append(now)
    _rate_counters[tenant_id] = window

# ── Multi-tenant helper ──────────────────────────────────
def _get_tenant(x_tenant_id: Optional[str]) -> str:
    return x_tenant_id or "default"

def _tenant_data_dir(tenant_id: str) -> str:
    path = os.path.join("data", "tenants", tenant_id)
    os.makedirs(path, exist_ok=True)
    return path

app = FastAPI(
    title="InsightFlow API",
    description="Agentic AI Data Analyst — Plan · Execute · Critique",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Timing middleware ───────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    latency  = (time.time() - start) * 1000

    _metrics["total_requests"] += 1
    _metrics["latencies"].append(latency)
    if len(_metrics["latencies"]) > 1000:
        _metrics["latencies"] = _metrics["latencies"][-1000:]
    _metrics["avg_latency_ms"] = round(
        sum(_metrics["latencies"]) / len(_metrics["latencies"]), 2
    )
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {latency:.0f}ms")
    response.headers["X-Response-Time-Ms"] = str(round(latency))
    return response

# ── request/response models ──────────────────────────────────

class AnalyzeRequest(BaseModel):
    question:     str
    dataset_path: str
    mode:         Optional[str] = "single"

class AnalyzeResponse(BaseModel):
    final_report:    str
    confidence:      float
    rows_validated:  int
    attempts:        int
    trace:           list[str]
    chart_paths:     list[str]
    question:        str
    mode:            str

class SaveMemoryRequest(BaseModel):
    question:       str
    final_report:   str
    confidence:     float
    rows_validated: int
    dataset_path:   str

class SearchMemoryRequest(BaseModel):
    query: str
    k:     Optional[int] = 3


# ── endpoints ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "InsightFlow API", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "uptime": datetime.utcnow().isoformat()}


@app.get("/metrics")
def get_metrics():
    """Observability — request counts, latency, per-tenant stats."""
    return {
        "total_requests":     _metrics["total_requests"],
        "successful":         _metrics["successful"],
        "failed":             _metrics["failed"],
        "avg_latency_ms":     _metrics["avg_latency_ms"],
        "requests_by_tenant": dict(_metrics["requests_by_tenant"]),
        "rate_limit_per_min": RATE_LIMIT,
        "started_at":         _metrics["started_at"],
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_tenant_id: Optional[str] = Header(None),
):
    tenant_id  = _get_tenant(x_tenant_id)
    _check_rate_limit(tenant_id)
    tenant_dir = _tenant_data_dir(tenant_id)
    save_path  = os.path.join(tenant_dir, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"[{tenant_id}] Uploaded: {file.filename}")

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(save_path)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(save_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported: {file.filename}")

        quality  = run_quality_report(df)
        detected = detect_columns(df)

        return {
            "filename":         file.filename,
            "dataset_path":     save_path,
            "tenant_id":        tenant_id,
            "rows":             quality["total_rows"],
            "columns":          quality["total_columns"],
            "health_label":     quality["health_label"],
            "completeness":     quality["completeness_score"],
            "detected_columns": detected,
            "null_counts":      quality["null_counts"],
        }
    except Exception as e:
        _metrics["failed"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    x_tenant_id: Optional[str] = Header(None),
):
    tenant_id  = _get_tenant(x_tenant_id)
    request_id = str(uuid.uuid4())[:8]
    _check_rate_limit(tenant_id)
    _metrics["requests_by_tenant"][tenant_id] += 1
    logger.info(f"[{tenant_id}][{request_id}] Analyze: '{request.question[:60]}'")

    if not os.path.exists(request.dataset_path):
        _metrics["failed"] += 1
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

    start = time.time()
    try:
        if request.dataset_path.endswith(".csv"):
            df = pd.read_csv(request.dataset_path)
        else:
            df = pd.read_excel(request.dataset_path)

        quality  = run_quality_report(df)
        detected = detect_columns(df)

        state = new_state(
            question=request.question,
            dataset_path=request.dataset_path,
            source_type="csv",
            mode=request.mode,
        )
        state["quality_report"]   = quality
        state["detected_columns"] = detected

        final_state = workflow.invoke(state)
        verdict     = final_state["critic_history"][-1]

        chart_paths = []
        for a in final_state["analysis_history"]:
            if hasattr(a, "chart_paths") and a.chart_paths:
                chart_paths.extend(a.chart_paths)
            elif a.chart_path:
                chart_paths.append(a.chart_path)
        chart_paths = list(dict.fromkeys(chart_paths))

        latency = round((time.time() - start) * 1000)
        _metrics["successful"] += 1
        logger.info(f"[{tenant_id}][{request_id}] Done {latency}ms | conf {verdict.confidence_score:.0%}")

        return AnalyzeResponse(
            final_report   = final_state["final_report"],
            confidence     = verdict.confidence_score,
            rows_validated = verdict.rows_validated or 0,
            attempts       = final_state["attempts"],
            trace          = final_state["trace"],
            chart_paths    = chart_paths,
            question       = request.question,
            mode           = request.mode,
        )

    except HTTPException:
        raise
    except Exception as e:
        _metrics["failed"] += 1
        logger.error(f"[{tenant_id}][{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/save")
def save_to_memory(request: SaveMemoryRequest):
    try:
        save_analysis(
            question       = request.question,
            final_report   = request.final_report,
            confidence     = request.confidence,
            rows_validated = request.rows_validated,
            dataset_path   = request.dataset_path,
        )
        stats = get_memory_stats()
        return {
            "saved":          True,
            "total_analyses": stats["total_analyses"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search")
def search_memory(request: SearchMemoryRequest):
    try:
        results = search_similar(query=request.query, k=request.k)
        return {
            "query":   request.query,
            "results": results,
            "count":   len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
def memory_stats():
    try:
        return get_memory_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )