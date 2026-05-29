import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from state import InsightFlowState, PlanStep

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

SYSTEM_PROMPT = """You are the Planner agent in InsightFlow — an AI data analyst system.

Your job: read the user's question and the dataset schema, then produce a step-by-step
analysis plan as JSON.

Available tools:
- describe_data — summary statistics. No arguments needed.
- aggregate — group by a column and calculate mean/sum/count/min/max
- trend_over_time — % change over time periods
- detect_anomaly — find outliers in a numeric column using IQR
- correlate — Pearson correlation between two numeric columns
- make_chart — save a matplotlib chart
- forecast — Prophet time series forecast
- what_if — simulate a % change on a column

CRITICAL RULES:
- ONLY use column names that exist in the dataset schema provided below
- NEVER invent or guess column names
- NEVER use column names from other datasets like "Region", "Total Revenue" etc
- Always read the detected columns from the schema and use those exact names
- If no date column exists, skip trend_over_time and forecast

Tool argument names — use EXACTLY these:
- aggregate: group_by, value_col, agg (mean/sum/count/min/max)
- make_chart: kind (bar/line/scatter/histogram), x, y
- trend_over_time: date_col, value_col, freq (ME/W/QE/YE)
- detect_anomaly: col
- correlate: col_a, col_b
- forecast: date_col, value_col, periods, freq
- what_if: col, change_pct, group_by (optional)
- describe_data: no arguments

For dashboard mode — create steps using ONLY the detected columns from the schema:
1. describe_data
2. aggregate using detected category_col and value_col, agg="sum"
3. make_chart kind="bar" using same columns as step 2
4. aggregate using a different category if available, agg="mean"
5. make_chart kind="bar" for step 4
6. trend_over_time using detected date_col and value_col (ONLY if date_col exists)
7. make_chart kind="line" for step 6 (ONLY if date_col exists)
8. correlate using two numeric columns from schema
9. detect_anomaly using detected value_col

Respond ONLY in valid JSON, no extra text, no trailing commas:
{
  "steps": [
    {
      "step_number": 1,
      "description": "description here",
      "tool_to_use": "tool_name",
      "tool_args": {}
    }
  ]
}"""


def _parse_json_safely(content: str) -> dict:
    content = content.strip()

    # strip markdown code blocks
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break

    # try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # try extracting JSON object
    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # fallback plan
    return {
        "steps": [{
            "step_number": 1,
            "description": "Get overview of the dataset",
            "tool_to_use": "describe_data",
            "tool_args": {}
        }]
    }


def _validate_args(step_args: dict, actual_columns: list, detected: dict) -> dict:
    column_args = ["group_by", "value_col", "date_col", "col", "col_a", "col_b", "x", "y"]
    validated = {}

    for arg_key, arg_val in step_args.items():
        if isinstance(arg_val, str) and arg_key in column_args:
            if arg_val in actual_columns:
                validated[arg_key] = arg_val
            else:
                # replace with detected column
                if arg_key in ["value_col", "col", "col_a", "y"]:
                    validated[arg_key] = detected.get("value_col", arg_val)
                elif arg_key in ["date_col"]:
                    validated[arg_key] = detected.get("date_col", arg_val)
                elif arg_key in ["group_by", "x"]:
                    validated[arg_key] = detected.get("category_col", arg_val)
                elif arg_key == "col_b":
                    validated[arg_key] = detected.get("date_col", arg_val)
                else:
                    validated[arg_key] = arg_val
        else:
            validated[arg_key] = arg_val

    return validated


def planner_node(state: InsightFlowState) -> dict:

    question         = state["question"]
    detected_columns = state["detected_columns"]
    quality_report   = state["quality_report"]
    mode             = state["mode"]

    # build schema context — only detected columns and types
    date_col     = detected_columns.get("date_col", "none")
    value_col    = detected_columns.get("value_col", "none")
    category_col = detected_columns.get("category_col", "none")
    id_col       = detected_columns.get("id_col", "none")

    schema_info = f"""Detected columns:
- date_col: {date_col}
- value_col: {value_col}
- category_col: {category_col}
- id_col: {id_col}

All columns in dataset: {list(quality_report['null_counts'].keys()) if quality_report else []}
Column types: {json.dumps(quality_report['column_types']) if quality_report else {}}
Data health: {quality_report['health_label'] if quality_report else 'unknown'}"""

    user_message = f"""Question: {question}
Mode: {mode}

Dataset schema:
{schema_info}

REMINDER: Only use column names listed above. Never use column names not in this schema.

Create the analysis plan."""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ])

    parsed = _parse_json_safely(response.content)

    # build PlanStep objects
    plan = []
    actual_columns = list(quality_report['null_counts'].keys()) if quality_report else []

    for step in parsed.get("steps", []):
        tool_name = step.get("tool_to_use", "describe_data")
        tool_args = step.get("tool_args", {})

        # validate all column args
        tool_args = _validate_args(tool_args, actual_columns, detected_columns)

        # skip trend/forecast if no date column
        if tool_name in ["trend_over_time", "forecast"] and not detected_columns.get("date_col"):
            continue

        plan_step = PlanStep(
            step_number=step.get("step_number", len(plan) + 1),
            description=step.get("description", tool_name),
            tool_to_use=tool_name,
            tool_args=tool_args,
        )
        plan.append(plan_step)

    # ensure we always have at least one step
    if not plan:
        plan.append(PlanStep(
            step_number=1,
            description="Get overview of the dataset",
            tool_to_use="describe_data",
            tool_args={},
        ))

    trace_entry = (
        f"PLANNER: {len(plan)} step(s) — "
        + ", ".join(s.tool_to_use for s in plan)
    )

    return {
        "plan":       plan,
        "next_agent": "analyst",
        "trace":      [trace_entry],
    }


if __name__ == "__main__":
    import pandas as pd
    from tools.quality        import run_quality_report
    from tools.detect_columns import detect_columns

    df       = pd.read_csv("data/sales.csv")
    quality  = run_quality_report(df)
    detected = detect_columns(df)

    test_state = {
        "question":         "What is the average revenue per region?",
        "dataset_path":     "data/sales.csv",
        "detected_columns": detected,
        "quality_report":   quality,
        "mode":             "single",
        "source_type":      "csv",
    }

    result = planner_node(test_state)

    print("=== Planner output ===\n")
    for step in result["plan"]:
        print(f"  Step {step.step_number}: {step.description}")
        print(f"  Tool: {step.tool_to_use}")
        print(f"  Args: {step.tool_args}")
        print()
    print(f"Trace: {result['trace'][0]}")