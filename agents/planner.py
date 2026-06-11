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
- describe_data    — summary statistics. No arguments needed.
- aggregate        — group by a column and calculate mean/sum/count/min/max
- trend_over_time  — % change over time periods
- detect_anomaly   — find outliers in a numeric column using IQR
- correlate        — Pearson correlation between two numeric columns
- make_chart       — save a matplotlib chart
- what_if          — simulate % change impact

CHART SELECTION RULES — follow strictly:
- category with <= 8 unique values → pie chart
- category with > 8 unique values  → bar chart (TOP 10 ONLY)
- time series (date column)        → line chart
- correlation results              → scatter chart
- distribution/anomaly             → histogram
- USE VARIETY — never use same chart type for every step
- NEVER use columns from DO NOT USE list

make_chart argument names: kind (bar/line/scatter/histogram/pie/area), x, y

CRITICAL RULES:
- ONLY use column names that exist in the dataset schema provided
- NEVER invent or guess column names
- NEVER use column names from other datasets
- If no date column exists, skip trend_over_time and forecast

Tool argument names — use EXACTLY these:
- aggregate:       group_by, value_col, agg (mean/sum/count/min/max)
- make_chart:      kind, x, y
- trend_over_time: date_col, value_col, freq (ME/W/QE/YE)
- detect_anomaly:  col
- correlate:       col_a, col_b
- what_if:         col, change_pct, group_by (optional)
- describe_data:   no arguments

DASHBOARD MODE INSTRUCTIONS:

STEP 1 — describe_data (always first)

STEP 2 — For EACH category in "ALL category columns":
  Choose the MOST MEANINGFUL combination — not just mean of value_col:
  
  a. COUNT — how many records per category:
     aggregate(group_by=category, value_col=category, agg="count")
     Chart: pie if <=8 unique values, bar if >8 (top 10 only)
     
  b. SUM — total value per category (only if meaningful):
     aggregate(group_by=category, value_col=<numeric>, agg="sum")
     Chart: bar
     
  c. MEAN — average value per category (only if insightful):
     aggregate(group_by=category, value_col=<numeric>, agg="mean")
     Chart: bar

  RULE: For each category pick COUNT first — it is always meaningful.
  Then decide if SUM or MEAN adds additional insight.
  Do NOT use same value_col for every category.

STEP 3 — For EACH numeric column in "Numeric columns available":
  a. Distribution — histogram of that numeric column:
     make_chart(kind="histogram", x=numeric_col, y=numeric_col)
  b. Trend — how does it change over time (only if date_col exists):
     trend_over_time(date_col=date_col, value_col=numeric_col)
     make_chart(kind="line", x=date_col, y=numeric_col)

STEP 4 — Correlate numeric columns if 2+ exist:
  correlate(col_a=numeric1, col_b=numeric2)
  make_chart(kind="scatter", x=numeric1, y=numeric2)

STEP 5 — detect_anomaly on primary value_col

SMART ANALYSIS RULES:
- COUNT is always meaningful — use it for every category
- SUM = total revenue, total sales, total cost — use for financial columns
- MEAN = average performance — use when comparing across groups
- NEVER use same value_col + agg combination for every category
- Mix chart types: pie, bar, histogram, line, scatter
- Each numeric column must appear in at least one analysis

IMPORTANT RULES FOR DASHBOARD:
- Mix aggregations: count, mean, sum — do not use only sum
- Mix chart types: pie, bar, histogram, line, scatter
- Every category should appear in at least one chart
- Use different numeric columns across different steps
- Do not repeat same group_by + value_col + agg combination

SINGLE QUESTION MODE:
SINGLE QUESTION MODE:
Create a focused 2-4 step plan that directly answers the question.

SINGLE MODE RULES:
- Read question carefully — understand what is being asked
- If question asks RELATION between 2 categorical columns:
  → aggregate(group_by=[col1, col2], value_col=col1, agg="count")
  → This gives count of records for each combination
- If question asks DISTRIBUTION of one column:
  → aggregate(group_by=col, value_col=col, agg="count") + chart
- If question asks HIGHEST/LOWEST:
  → aggregate(group_by=col, value_col=numeric_col, agg="sum" or "mean")
- If question asks TREND over time:
  → trend_over_time(date_col, value_col)
- If question asks CORRELATION between 2 numeric columns:
  → correlate(col_a=num1, col_b=num2) + scatter chart
- Always add make_chart after aggregate
- NEVER answer with unrelated columns

Respond ONLY in valid JSON, no extra text:
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

    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return {
        "steps": [{
            "step_number": 1,
            "description": "Get overview of the dataset",
            "tool_to_use": "describe_data",
            "tool_args":   {}
        }]
    }


def _validate_args(step_args: dict, actual_columns: list, detected: dict) -> dict:
    column_args = ["group_by", "value_col", "date_col", "col",
                   "col_a", "col_b", "x", "y"]
    validated = {}

    for arg_key, arg_val in step_args.items():
        if isinstance(arg_val, str) and arg_key in column_args:
            if arg_val in actual_columns:
                validated[arg_key] = arg_val
            else:
                if arg_key in ["value_col", "col", "col_a", "y"]:
                    validated[arg_key] = detected.get("value_col", arg_val)
                elif arg_key == "date_col":
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

    # ── build schema context ───────────────────────────────────────────────
    date_col          = detected_columns.get("date_col",          "none")
    value_col         = detected_columns.get("value_col",         "none")
    category_col      = detected_columns.get("category_col",      "none")
    id_col            = detected_columns.get("id_col",            "none")
    all_category_cols = detected_columns.get("all_category_cols", [])
    all_value_cols    = detected_columns.get("all_value_cols",    [])
    domain            = detected_columns.get("domain",            "unknown")
    skip_cols         = detected_columns.get("skip_cols",         [])

    # filter skip_cols from categories
    clean_cats = [c for c in all_category_cols if c not in skip_cols]

    categories_str = "\n".join(f"  - {col}" for col in clean_cats) \
                     if clean_cats else f"  - {category_col}"

    numeric_str = "\n".join(f"  - {col}" for col in all_value_cols) \
                  if all_value_cols else f"  - {value_col}"

    schema_info = f"""Dataset domain: {domain}

    Detected columns:
    - date_col:     {date_col}
    - value_col:    {value_col}
    - category_col: {category_col}
    - id_col:       {id_col}

    ALL category columns — use ALL in dashboard mode:
    {categories_str}

    Numeric columns available — use ALL of these:
    {numeric_str}

    DO NOT USE these columns — no analytical value:
    {skip_cols}

    All columns in dataset: {list(quality_report['null_counts'].keys()) if quality_report else []}"""


    user_message = f"""Question: {question}
    Mode: {mode}

    Dataset schema:
    {schema_info}

    REMINDER:
    - Only use column names listed above
    - Dashboard mode: COUNT + MEAN for EACH category, use ALL numeric columns
    - Single mode: focused 2-4 steps to answer the question
    - Mix chart types — pie, bar, histogram, line, scatter
    - NEVER use DO NOT USE columns

    Create the analysis plan."""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ])

    parsed      = _parse_json_safely(response.content)
    actual_cols = list(quality_report['null_counts'].keys()) if quality_report else []
    plan        = []

    for step in parsed.get("steps", []):
        tool_name = step.get("tool_to_use", "describe_data")
        tool_args = step.get("tool_args", {})

        tool_args = _validate_args(tool_args, actual_cols, detected_columns)

        if tool_name in ["trend_over_time", "forecast"] and \
                not detected_columns.get("date_col"):
            continue

        plan_step = PlanStep(
            step_number = step.get("step_number", len(plan) + 1),
            description = step.get("description", tool_name),
            tool_to_use = tool_name,
            tool_args   = tool_args,
        )
        plan.append(plan_step)

    if not plan:
        plan.append(PlanStep(
            step_number = 1,
            description = "Get overview of the dataset",
            tool_to_use = "describe_data",
            tool_args   = {},
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

    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    mode      = sys.argv[2] if len(sys.argv) > 2 else "dashboard"

    print(f"Loading: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print(f"Shape: {df.shape}")

    quality  = run_quality_report(df)
    detected = detect_columns(df)

    print(f"Detected categories: {detected.get('all_category_cols')}")
    print(f"Detected numerics:   {detected.get('all_value_cols')}")
    print(f"Skip cols:           {detected.get('skip_cols')}")

    test_state = {
        "question":         "Generate a complete dashboard analysis",
        "dataset_path":     file_path,
        "detected_columns": detected,
        "quality_report":   quality,
        "mode":             mode,
        "source_type":      "csv" if file_path.endswith(".csv") else "excel",
    }

    result = planner_node(test_state)

    print("\n=== Planner output ===\n")
    for step in result["plan"]:
        print(f"  Step {step.step_number}: {step.description}")
        print(f"  Tool: {step.tool_to_use} | Args: {step.tool_args}")
        print()
    print(f"Trace: {result['trace'][0]}")