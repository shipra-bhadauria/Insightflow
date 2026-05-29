import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
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

app = FastAPI(
    title="InsightFlow API",
    description="Agentic AI Data Analyst — Plan · Execute · Critique",
    version="1.0.0",
)

# allow Streamlit and any frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {
        "name":    "InsightFlow API",
        "version": "1.0.0",
        "status":  "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # save uploaded file to data/ folder
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # run pre-analysis
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(save_path)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(save_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )

        quality  = run_quality_report(df)
        detected = detect_columns(df)

        return {
            "filename":         file.filename,
            "dataset_path":     save_path,
            "rows":             quality["total_rows"],
            "columns":          quality["total_columns"],
            "health_label":     quality["health_label"],
            "completeness":     quality["completeness_score"],
            "detected_columns": detected,
            "null_counts":      quality["null_counts"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not os.path.exists(request.dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.dataset_path}"
        )

    try:
        # load file and run pre-analysis
        if request.dataset_path.endswith(".csv"):
            df = pd.read_csv(request.dataset_path)
        else:
            df = pd.read_excel(request.dataset_path)

        quality  = run_quality_report(df)
        detected = detect_columns(df)

        # create state and run agents
        state = new_state(
            question=request.question,
            dataset_path=request.dataset_path,
            source_type="csv",
            mode=request.mode,
        )
        state["quality_report"]   = quality
        state["detected_columns"] = detected

        final_state = workflow.invoke(state)

        # extract results
        verdict     = final_state["critic_history"][-1]
        attempt     = final_state["analysis_history"][-1]

        chart_paths = []
        for a in final_state["analysis_history"]:
            if hasattr(a, "chart_paths") and a.chart_paths:
                chart_paths.extend(a.chart_paths)
            elif a.chart_path:
                chart_paths.append(a.chart_path)
        chart_paths = list(dict.fromkeys(chart_paths))

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

    except Exception as e:
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