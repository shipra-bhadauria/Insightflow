import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from dotenv import load_dotenv

from state import InsightFlowState, AnalysisAttempt
from tools.describe       import describe_data
from tools.aggregate      import aggregate
from tools.trend          import trend_over_time
from tools.anomaly        import detect_anomaly
from tools.correlate      import correlate
from tools.chart          import make_chart
from tools.forecast       import forecast
from tools.what_if        import what_if

load_dotenv()

# the only tools the Analyst can call
TOOLBOX = {
    "describe_data":    describe_data,
    "aggregate":        aggregate,
    "trend_over_time":  trend_over_time,
    "detect_anomaly":   detect_anomaly,
    "correlate":        correlate,
    "make_chart":       make_chart,
    "forecast":         forecast,
    "what_if":          what_if,
}


def _load_dataframe(dataset_path: str) -> pd.DataFrame:
    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(dataset_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _call_tool(tool_name: str, df: pd.DataFrame, tool_args: dict, previous_results: dict = None) -> dict:
    if tool_name not in TOOLBOX:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_fn = TOOLBOX[tool_name]

    # describe_data takes only df
    if tool_name == "describe_data":
        return tool_fn(df)

    # make_chart — try to build df from previous aggregate result
    if tool_name == "make_chart" and previous_results:
        chart_x = tool_args.get("x", "")
        chart_y = tool_args.get("y", "")

        best_match = None

        # try to find aggregate whose group_by or value_col matches chart axes
        for key, prev_result in reversed(list(previous_results.items())):
            if "aggregate" in key and isinstance(prev_result, dict) and "result" in prev_result:
                if (prev_result.get("group_by") == chart_x or
                        prev_result.get("value_col") == chart_y):
                    best_match = prev_result
                    break

        # fallback — use most recent aggregate
        if not best_match:
            for key, prev_result in reversed(list(previous_results.items())):
                if "aggregate" in key and isinstance(prev_result, dict) and "result" in prev_result:
                    best_match = prev_result
                    break

        # if we found an aggregate result — build DataFrame from it
        if best_match and isinstance(best_match.get("result"), dict):
            try:
                agg_df = pd.DataFrame(
                    list(best_match["result"].items()),
                    columns=[best_match["group_by"], best_match["value_col"]]
                )
                return tool_fn(agg_df, **tool_args)
            except Exception:
                pass  # fall through to normal call below

    # default — call tool with original df
    return tool_fn(df, **tool_args)


def analyst_node(state: InsightFlowState) -> dict:

    plan     = state["plan"]
    attempts = state["attempts"]

    # load the dataframe
    df = _load_dataframe(state["dataset_path"])

    # check if Critic rejected last attempt — read the reason
    critic_feedback = None
    if state["critic_history"]:
        last_verdict = state["critic_history"][-1]
        if not last_verdict.approved:
            critic_feedback = last_verdict.reason

    # run each step in the plan
    all_results = []
    chart_paths = []

    for step in plan:
        tool_name = step.tool_to_use
        tool_args = step.tool_args.copy()

        if critic_feedback and "null" in critic_feedback.lower():
            if "dropna" in str(TOOLBOX[tool_name].__code__.co_varnames):
                tool_args["dropna"] = True

        prev = {f"{r['tool']}_{i}": r["result"] for i, r in enumerate(all_results)}
        result = _call_tool(tool_name, df, tool_args, previous_results=prev)

        if tool_name == "make_chart" and "chart_path" in result:
            chart_paths.append(result["chart_path"])

        all_results.append({
            "tool":   tool_name,
            "args":   tool_args,
            "result": result,
        })

    # combine all results into one dict
    combined_result = {}
    for r in all_results:
        combined_result[r["tool"]] = r["result"]

    # set attempt number
    attempt_number = attempts + 1

    chart_path = chart_paths[0] if chart_paths else None

    analysis_attempt = AnalysisAttempt(
        attempt_number=attempt_number,
        tool_called=plan[0].tool_to_use,
        tool_args=plan[0].tool_args,
        tool_result=combined_result,
        chart_path=chart_path,
        chart_paths=chart_paths,
    )

    feedback_note = f" (retry — {critic_feedback[:50]}...)" if critic_feedback else ""
    trace_entry = (
        f"ANALYST attempt {attempt_number}: "
        + ", ".join(r["tool"] for r in all_results)
        + feedback_note
    )

    return {
        "analysis_history": [analysis_attempt],
        "attempts":          attempt_number,
        "next_agent":        "critic",
        "trace":             [trace_entry],
    }

if __name__ == "__main__":
    import pandas as pd
    from tools.quality        import run_quality_report
    from tools.detect_columns import detect_columns
    from state                import PlanStep, new_state

    df       = pd.read_csv("data/sales.csv")
    quality  = run_quality_report(df)
    detected = detect_columns(df)

    # simulate what the Planner would have written
    step1 = PlanStep(
        step_number=1,
        description="group by region and calculate mean revenue",
        tool_to_use="aggregate",
        tool_args={
            "group_by":  "Region",
            "value_col": "Total Revenue",
            "agg":       "mean",
        }
    )

    test_state = new_state(
        question="What is the average revenue per region?",
        dataset_path="data/sales.csv",
    )
    test_state["detected_columns"] = detected
    test_state["quality_report"]   = quality
    test_state["plan"]             = [step1]

    result = analyst_node(test_state)

    print("=== Analyst output ===\n")
    attempt = result["analysis_history"][0]
    print(f"Attempt number: {attempt.attempt_number}")
    print(f"Tool called:    {attempt.tool_called}")
    print(f"Tool args:      {attempt.tool_args}")
    print(f"\nResult:")
    for tool, output in attempt.tool_result.items():
        print(f"\n  {tool}:")
        for k, v in output.items():
            print(f"    {k}: {v}")
    print(f"\nTrace: {result['trace'][0]}")