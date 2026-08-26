import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from dotenv import load_dotenv
from logger import agent_logger as logger

from state import InsightFlowState, AnalysisAttempt, PlanStep
from tools.describe  import describe_data
from tools.aggregate import aggregate
from tools.trend     import trend_over_time
from tools.anomaly   import detect_anomaly
from tools.correlate import correlate
from tools.chart     import make_chart
from tools.forecast  import forecast
from tools.what_if   import what_if

load_dotenv()

TOOLBOX = {
    "describe_data":   describe_data,
    "aggregate":       aggregate,
    "trend_over_time": trend_over_time,
    "detect_anomaly":  detect_anomaly,
    "correlate":       correlate,
    "make_chart":      make_chart,
    "forecast":        forecast,
    "what_if":         what_if,
}


def _load_dataframe(dataset_path: str) -> pd.DataFrame:
    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == ".csv":
        try:
            return pd.read_csv(dataset_path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(dataset_path, encoding="latin-1")
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _safe_chart(df: pd.DataFrame, kind: str, x: str, y: str,
                title: str = "", output_dir: str = "outputs",
                agg: str = "sum") -> dict:
    """Safe chart call — returns None chart_path instead of crashing."""
    try:
        return make_chart(df, kind=kind, x=x, y=y,
                          title=title, output_dir=output_dir, agg=agg)
    except Exception as e:
        return {"chart_path": None, "note": str(e)}


def _build_dashboard_steps(df: pd.DataFrame,
                            detected: dict) -> list:
    """
    Programmatically build 8-12 MEANINGFUL analysis steps for dashboard mode.
    No LLM — pure logic based on detected columns. Keeps only high-value charts,
    skips noisy/low-value ones (e.g. per-column trend-over-launch-date lines).
    """
    steps = []

    value_cols  = detected.get("all_value_cols", [])
    cat_cols    = detected.get("all_category_cols", [])
    skip_cols   = detected.get("skip_cols", [])

    # clean categories
    cat_cols   = [c for c in cat_cols if c not in skip_cols and c in df.columns]
    value_cols = [c for c in value_cols if c in df.columns]

    # ── STEP 1 — describe_data (always first, drives KPI cards) ────────────
    steps.append(("describe_data", {}))

    # ── STEP 2 — COUNT chart per category, max 4 categories ────────────────
    for cat in cat_cols[:4]:
        n_unique = df[cat].nunique()
        chart_kind = "pie" if n_unique <= 8 else "bar"

        steps.append(("aggregate", {
            "group_by":  cat,
            "value_col": cat,
            "agg":       "count",
        }))
        steps.append(("make_chart", {
            "kind": chart_kind,
            "x":    cat,
            "y":    cat,
            "agg":  "count",
        }))

    # ── STEP 3 — MEAN of primary numeric column, per category, max 2 ───────
    for cat in cat_cols[:2]:
        for val in value_cols[:1]:
            if cat == val:
                continue
            steps.append(("aggregate", {
                "group_by":  cat,
                "value_col": val,
                "agg":       "mean",
            }))
            steps.append(("make_chart", {
                "kind": "bar",
                "x":    cat,
                "y":    val,
                "agg":  "mean",
            }))

    # ── STEP 4 — Distribution histograms, top 2 numeric columns only ───────
    for val in value_cols[:2]:
        steps.append(("make_chart", {
            "kind": "histogram",
            "x":    val,
            "y":    val,
        }))

    # ── STEP 5 — Correlate top 2 numeric columns ────────────────────────────
    if len(value_cols) >= 2:
        steps.append(("correlate", {
            "col_a": value_cols[0],
            "col_b": value_cols[1],
        }))
        steps.append(("make_chart", {
            "kind": "scatter",
            "x":    value_cols[0],
            "y":    value_cols[1],
        }))

    # ── STEP 6 — Anomaly detection on primary numeric column ────────────────
    if value_cols:
        steps.append(("detect_anomaly", {
            "col": value_cols[0],
        }))

    # NOTE: trend_over_time/line charts are intentionally SKIPPED here —
    # a "launch date" style column doesn't give a meaningful time-series
    # trend and was cluttering the dashboard with 10+ noisy, low-value charts.
    # Total charts produced: ~9-10 (4 count + 2 mean + 2 histogram + 1 scatter).

    return steps


def _call_tool(tool_name: str, df: pd.DataFrame,
               tool_args: dict, previous_results: dict = None) -> dict:
    if tool_name not in TOOLBOX:
        return {"error": f"Unknown tool: {tool_name}"}

    tool_fn = TOOLBOX[tool_name]

    if tool_name == "describe_data":
        return tool_fn(df)

    # make_chart — pehle-se-aggregated result se safe chart banao
    if tool_name == "make_chart":
        chart_x    = tool_args.get("x", "")
        chart_y    = tool_args.get("y", "")
        chart_kind = tool_args.get("kind", "bar")
        chart_agg  = tool_args.get("agg", "sum")

        # count/mean charts ke liye — pehle previous aggregate result dhoondo
        if previous_results and chart_agg in ("count", "mean", "sum"):
            best_match = None
            for key, prev in reversed(list(previous_results.items())):
                if "aggregate" not in key or not isinstance(prev, dict):
                    continue
                if prev.get("group_by") == chart_x and prev.get("agg") == chart_agg:
                    best_match = prev
                    break
            if best_match and isinstance(best_match.get("result"), dict):
                try:
                    grp = best_match["group_by"]
                    val = best_match["value_col"]
                    if grp == val:
                        agg_df = pd.DataFrame(
                            list(best_match["result"].items()),
                            columns=[grp, "count"]
                        )
                        n_unique = len(agg_df)
                        use_kind = "pie" if (chart_kind == "pie" and n_unique <= 8) else "bar"
                        return _safe_chart(agg_df, kind=use_kind, x=grp, y="count", agg="sum")
                    else:
                        agg_df = pd.DataFrame(
                            list(best_match["result"].items()),
                            columns=[grp, val]
                        )
                        return _safe_chart(agg_df, kind=chart_kind, x=grp, y=val, agg="sum")
                except Exception:
                    pass

        if chart_x in df.columns and chart_y in df.columns:
            n_unique = df[chart_x].nunique() if chart_x in df.columns else 999
            if chart_kind == "pie" and (n_unique > 8 or chart_agg not in ("count", None)):
                chart_kind = "bar"
            return _safe_chart(df, kind=chart_kind, x=chart_x, y=chart_y, agg=chart_agg)

        return {"chart_path": None, "note": "No data for chart"}

    # trend_over_time chart — use trend result
    if tool_name == "make_chart" and previous_results:
        for key, prev in reversed(list(previous_results.items())):
            if "trend" in key and isinstance(prev, dict) and "periods" in prev:
                try:
                    periods = prev["periods"]
                    trend_df = pd.DataFrame([
                        {"date": k, "value": v["value"]}
                        for k, v in periods.items()
                    ])
                    trend_df["date"] = pd.to_datetime(trend_df["date"])
                    return _safe_chart(
                        trend_df,
                        kind="line",
                        x="date",
                        y="value",
                    )
                except Exception:
                    pass

    return tool_fn(df, **tool_args)


def _run_steps(df: pd.DataFrame, steps: list,
               critic_feedback: str = None) -> tuple:
    """Run a list of (tool_name, tool_args) steps."""
    all_results = []
    chart_paths = []

    for tool_name, tool_args in steps:
        tool_args = tool_args.copy()

        if critic_feedback and "null" in critic_feedback.lower():
            if tool_name in TOOLBOX and \
               "dropna" in str(TOOLBOX[tool_name].__code__.co_varnames):
                tool_args["dropna"] = True

        prev = {f"{r['tool']}_{i}": r["result"]
                for i, r in enumerate(all_results)}

        try:
            result = _call_tool(tool_name, df, tool_args,
                                previous_results=prev)
        except Exception as e:
            result = {"error": str(e)}

        if tool_name == "make_chart" and result.get("chart_path"):
            chart_paths.append(result["chart_path"])

        all_results.append({
            "tool":   tool_name,
            "args":   tool_args,
            "result": result,
        })

    return all_results, chart_paths


def analyst_node(state: InsightFlowState) -> dict:
    attempts = state.get("attempts", 0)
    logger.info(f"Analyst | attempt={attempts + 1}")

    plan             = state["plan"]
    attempts         = state["attempts"]
    mode             = state.get("mode", "single")
    detected_columns = state.get("detected_columns", {})

    df = _load_dataframe(state["dataset_path"])

    critic_feedback = None
    if state["critic_history"]:
        last_verdict = state["critic_history"][-1]
        if not last_verdict.approved:
            critic_feedback = last_verdict.reason

    # ── DASHBOARD MODE — programmatic smart analysis ───────────────────────────
    if mode == "dashboard":
        steps = _build_dashboard_steps(df, detected_columns)
        all_results, chart_paths = _run_steps(
            df, steps, critic_feedback=critic_feedback
        )

    # ── SINGLE MODE — follow Planner's plan ───────────────────────────────────
    else:
        plan_steps = [(s.tool_to_use, s.tool_args) for s in plan]
        all_results, chart_paths = _run_steps(
            df, plan_steps, critic_feedback=critic_feedback
        )

    # combine results — ensure all values are JSON serializable
    combined_result = {}
    for r in all_results:
        key = r["tool"]
        if key in combined_result:
            i = 2
            while f"{key}_{i}" in combined_result:
                i += 1
            key = f"{key}_{i}"
        result_val = r["result"]
        # convert result list to dict wrapper so Pydantic is happy
        if isinstance(result_val, dict) and isinstance(result_val.get("result"), list):
            result_val = {
                **{k: v for k, v in result_val.items() if k != "result"},
                "result": result_val["result"],
                "result_count": len(result_val["result"]),
            }
        combined_result[key] = result_val

    attempt_number = attempts + 1
    chart_path     = chart_paths[0] if chart_paths else None

    # use first plan step for metadata (single mode) or describe for dashboard
    ref_step = plan[0] if plan else PlanStep(
        step_number=1, description="dashboard", tool_to_use="describe_data", tool_args={}
    )

    analysis_attempt = AnalysisAttempt(
        attempt_number = attempt_number,
        tool_called    = ref_step.tool_to_use,
        tool_args      = ref_step.tool_args,
        tool_result    = combined_result,
        chart_path     = chart_path,
        chart_paths    = chart_paths,
    )

    feedback_note = f" (retry — {critic_feedback[:50]}...)" if critic_feedback else ""
    trace_entry = (
        f"ANALYST attempt {attempt_number}: "
        + ", ".join(r["tool"] for r in all_results[:6])
        + (f"... +{len(all_results)-6} more" if len(all_results) > 6 else "")
        + feedback_note
    )

    return {
        "analysis_history": [analysis_attempt],
        "attempts":          attempt_number,
        "next_agent":        "critic",
        "trace":             [trace_entry],
    }


if __name__ == "__main__":
    from tools.quality        import run_quality_report
    from tools.detect_columns import detect_columns
    from state                import new_state

    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    mode      = sys.argv[2] if len(sys.argv) > 2 else "dashboard"

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    quality  = run_quality_report(df)
    detected = detect_columns(df)

    print(f"Categories: {detected.get('all_category_cols')}")
    print(f"Numerics:   {detected.get('all_value_cols')}")
    print(f"Date:       {detected.get('date_col')}")

    test_state = new_state(
        question="Generate a complete dashboard analysis",
        dataset_path=file_path,
        mode=mode,
    )
    test_state["detected_columns"] = detected
    test_state["quality_report"]   = quality

    result = analyst_node(test_state)

    print(f"\nAttempts: {result['attempts']}")
    print(f"Charts:   {result['analysis_history'][0].chart_paths}")
    print(f"Trace:    {result['trace'][0]}")
