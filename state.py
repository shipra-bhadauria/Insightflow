import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_number: int = Field(...)
    description: str = Field(...)
    tool_to_use: str = Field(...)
    tool_args:   dict = Field(default_factory=dict)  # arguments for the tool

class RetrievedContext(BaseModel):
    source: str = Field(...)            # "faiss", "pdf", "image"
    content: str = Field(...)
    relevance_score: float = Field(...)


class AnalysisAttempt(BaseModel):
    attempt_number: int = Field(...)
    tool_called:    str = Field(...)
    tool_args:      dict = Field(...)
    tool_result:    dict = Field(...)
    chart_path:     Optional[str] = Field(None)
    chart_paths:    list[str] = Field(default_factory=list)  # add this


class CriticVerdict(BaseModel):
    attempt_number: int = Field(...)
    approved: bool = Field(...)
    confidence_score: float = Field(...)
    reason: str = Field(...)
    rows_validated: Optional[int] = Field(None)


class DataQualityReport(BaseModel):
    total_rows: int
    total_columns: int
    null_counts: dict
    duplicate_rows: int
    column_types: dict
    outlier_columns: list
    completeness_score: float           # 0.0 to 1.0 — how clean is the data
    detected_date_col: Optional[str]
    detected_value_col: Optional[str]
    detected_category_col: Optional[str]


class ConversationTurn(BaseModel):
    question: str
    answer: str


class ForecastResult(BaseModel):
    periods: int                        # how many future periods predicted
    forecast_values: list[float]
    forecast_dates: list[str]
    chart_path: Optional[str] = None


class WhatIfResult(BaseModel):
    column_changed: str
    change_pct: float                   # e.g. 10.0 means +10%
    original_total: float
    projected_total: float
    impact_summary: str                 # plain English e.g. "revenue increases by £240K"


class DashboardInsight(BaseModel):
    title: str                          # short heading e.g. "Enterprise leads by 3.1x"
    finding: str                        # full Reporter finding
    chart_path: Optional[str] = None
    insight_type: str                   # "kpi", "trend", "anomaly", "forecast", "correlation"


class InsightFlowState(TypedDict):
    # inputs — set once at the start
    question: str
    dataset_path: str
    document_paths: list[str]
    source_type: str                    # "csv", "excel", "pdf", "image", "gsheets"
    mode: str                           # "single" = one question, "dashboard" = full auto analysis

    # pre-analysis — filled before Planner runs
    quality_report: Optional[DataQualityReport]
    detected_columns: dict              # {"date": "order_date", "value": "revenue", "category": "segment"}

    # filled as agents run
    plan: list[PlanStep]
    context: list[RetrievedContext]

    # operator.add = append not overwrite
    analysis_history: Annotated[list[AnalysisAttempt], operator.add]
    critic_history:   Annotated[list[CriticVerdict], operator.add]
    trace:            Annotated[list[str], operator.add]

    # prediction outputs
    forecast_result: Optional[ForecastResult]
    what_if_result: Optional[WhatIfResult]

    # full dashboard outputs
    dashboard_insights: list[DashboardInsight]
    dashboard_kpis: dict                # {"Total Revenue": "£2.4M", "Top Segment": "Enterprise"}
    dashboard_charts: list[str]         # list of chart file paths

    # conversation memory — powers follow-up questions
    conversation_history: list[ConversationTurn]

    # retry loop control
    attempts: int
    max_attempts: int
    next_agent: str

    # final output
    final_report: Optional[str]
    approved_by_human: bool


def new_state(
    question: str,
    dataset_path: str,
    document_paths: list[str] = None,
    source_type: str = "csv",
    mode: str = "single",
    max_attempts: int = 3,
) -> InsightFlowState:
    return InsightFlowState(
        question=question,
        dataset_path=dataset_path,
        document_paths=document_paths or [],
        source_type=source_type,
        mode=mode,
        quality_report=None,
        detected_columns={},
        plan=[],
        context=[],
        analysis_history=[],
        critic_history=[],
        forecast_result=None,
        what_if_result=None,
        dashboard_insights=[],
        dashboard_kpis={},
        dashboard_charts=[],
        conversation_history=[],
        attempts=0,
        max_attempts=max_attempts,
        next_agent="planner",
        final_report=None,
        approved_by_human=False,
        trace=[f"SYSTEM: new run started — question: '{question}' — mode: {mode}"],
    )


if __name__ == "__main__":
    state = new_state(
        question="What is the average revenue per customer segment?",
        dataset_path="data/sales_q3.csv",
        source_type="csv",
        mode="single",
    )
    print("=== InsightFlow State ===\n")
    for key, value in state.items():
        print(f"  {key:25s} → {value}")
    print("\n✓ State initialised successfully.")