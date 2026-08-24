import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert data analyst.
Given a dataset schema — identify every column's analytical role.

Return ONLY this JSON — no markdown, no backticks, no extra text:
{
  "date_col":              "column_name or null",
  "value_col":             "column_name or null",
  "all_value_cols":        ["col1", "col2", "col3"],
  "category_col":          "column_name or null",
  "all_category_cols":     ["col1", "col2", "col3"],
  "high_cardinality_cols": ["col1"],
  "id_col":                "column_name or null",
  "skip_cols":             ["col1"],
  "domain":                "sales/hr/hospital/finance/logistics/ecommerce/other"
}

STRICT RULES:

1. date_col
   → column with actual dates or timestamps
   → NOT year-only integers like 2020, 2021

2. value_col
   → PRIMARY numeric column most worth analyzing
   → e.g. revenue, price, salary, score, billing amount

3. all_value_cols
   → ALL numeric columns worth analyzing — include as many as meaningful
   → INCLUDE: any numeric column that reveals performance, size, cost, quantity
   → EXCLUDE: id numbers, room numbers, phone numbers, zip codes, row index columns

4. category_col
   → PRIMARY grouping column — most analytically useful for GROUP BY

5. all_category_cols
   → Columns with 2-100 unique values that make meaningful GROUP BY groups
   → INCLUDE: any column where grouping gives business insight
     (type, status, level, grade, region, department, condition, gender)
   → EXCLUDE:
     * Derived quality labels that describe a numeric metric
       e.g. if a column contains POOR/GOOD/EXCELLENT as labels for another
       numeric column → skip it, it adds no GROUP BY value
     * Full person names, email addresses, street addresses, URLs

6. high_cardinality_cols
   → Columns where unique count > 50% of total rows
   → Too many unique values to GROUP BY — but important for exact lookup queries
   → e.g. product names, model names, hospital names, drug names, job titles
   → These will be used when user asks "show me all X" or "list each X"

7. id_col
   → True unique row identifier — unique count equals total rows
   → e.g. patient ID, order ID, employee ID

8. skip_cols
   → ONLY: email addresses, street addresses, URLs, phone numbers
   → Do NOT skip product names, model names, or entity names

9. domain
   → What kind of dataset is this?

DECISION RULES:
- unique_count > 50% of total rows AND column is a name/label → high_cardinality_cols
- unique_count = 100% of total rows → id_col
- unique_count between 2-100 AND meaningful grouping → all_category_cols
- numeric AND analytically meaningful → all_value_cols
- POOR/GOOD/EXCELLENT as quality label for another column → skip_cols
- dates/timestamps → date_col
"""


def _build_schema(df: pd.DataFrame) -> str:
    """Schema with unique value samples for low-cardinality columns."""
    lines = []
    total_rows = len(df)
    for col in df.columns:
        dtype    = str(df[col].dtype)
        n_unique = df[col].nunique()
        pct      = round(n_unique / total_rows * 100, 1)
        samples  = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(s) for s in samples)

        # for low cardinality columns — show ALL unique values
        if n_unique <= 20:
            all_vals = df[col].dropna().unique().tolist()
            unique_str = f"ALL values: {all_vals}"
        else:
            unique_str = f"samples: {sample_str}"

        lines.append(
            f"{col!r} | {dtype} | {n_unique} unique ({pct}%) | {unique_str}"
        )
    return f"Total rows: {total_rows}\n\n" + "\n".join(lines)


def _call_llm(schema_text: str) -> dict:
    llm = ChatOpenAI(
        model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Dataset schema:\n{schema_text}")
    ])
    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def _validate_against_df(result: dict, df: pd.DataFrame) -> dict:
    actual_cols = list(df.columns)
    for key in ["date_col", "value_col", "category_col", "id_col"]:
        if result.get(key) not in actual_cols:
            result[key] = None
    for key in ["all_value_cols", "all_category_cols", "skip_cols", "high_cardinality_cols"]:
        result[key] = [
            col for col in result.get(key, [])
            if col in actual_cols
        ]
    return result


def _calculate_confidence(result: dict) -> float:
    base = sum(
        1 for k in ["date_col", "value_col", "category_col"]
        if result.get(k)
    ) / 3
    category_count = len(result.get("all_category_cols", []))
    category_bonus = min(category_count / 5, 1.0) * 0.2
    return round(min(base + category_bonus, 1.0), 2)


def _fallback_result(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(include="object").columns.tolist()

    value_col = None
    if numeric_cols:
        skip_kw = ["id", "code", "key", "ref", "number", "phone", "zip", "room"]
        clean   = [c for c in numeric_cols
                   if not any(kw in c.lower() for kw in skip_kw)]
        pool    = clean if clean else numeric_cols
        means   = {c: df[c].mean() for c in pool}
        value_col = max(means, key=means.get)

    all_cats = sorted(
        [c for c in text_cols if 2 <= df[c].nunique() <= 100],
        key=lambda c: df[c].nunique()
    )
    category_col = all_cats[0] if all_cats else None

    all_high_card = [
        c for c in text_cols
        if df[c].nunique() > len(df) * 0.5
    ]

    return {
        "date_col":              None,
        "value_col":             value_col,
        "all_value_cols":        [value_col] if value_col else [],
        "category_col":          category_col,
        "all_category_cols":     all_cats[:8],
        "high_cardinality_cols": all_high_card,
        "id_col":                None,
        "skip_cols":             [],
        "domain":                "other",
        "confidence":            0.40,
        "source":                "fallback",
        "category_unique_values": (
            int(df[category_col].nunique()) if category_col else None
        ),
    }


def detect_columns(df: pd.DataFrame, use_llm_fallback: bool = True) -> dict:
    schema_text = _build_schema(df)

    try:
        result = _call_llm(schema_text)
        result = _validate_against_df(result, df)

        if result.get("category_col") and \
                result["category_col"] not in result.get("all_category_cols", []):
            result.setdefault("all_category_cols", []).insert(
                0, result["category_col"]
            )

        if result.get("value_col") and \
                result["value_col"] not in result.get("all_value_cols", []):
            result.setdefault("all_value_cols", []).insert(
                0, result["value_col"]
            )

        result["source"]     = "llm"
        result["confidence"] = _calculate_confidence(result)
        result["category_unique_values"] = (
            int(df[result["category_col"]].nunique())
            if result.get("category_col") else None
        )

        # post-process: remove derived quality labels from categories
        # e.g. Retention Value (POOR/GOOD), HP Level might be ok but
        # columns with only 2-3 values like POOR/GOOD/EXCELLENT are quality labels
        clean_cats = []
        for col in result.get("all_category_cols", []):
            if col not in df.columns:
                continue
            unique_vals = df[col].dropna().unique().tolist()
            unique_str  = [str(v).upper().strip() for v in unique_vals]
            # skip if all values are quality labels
            quality_labels = {"POOR", "GOOD", "EXCELLENT", "FAIR",
                              "BAD", "AVERAGE", "LOW", "HIGH", "MEDIUM"}
            if all(v in quality_labels for v in unique_str):
                if col not in result.get("skip_cols", []):
                    result.setdefault("skip_cols", []).append(col)
                continue
            clean_cats.append(col)
        result["all_category_cols"] = clean_cats

        # update category_col if it was removed
        if result.get("category_col") not in clean_cats:
            result["category_col"] = clean_cats[0] if clean_cats else None

        return result

    except Exception as e:
        if use_llm_fallback:
            fallback = _fallback_result(df)
            fallback["llm_error"] = str(e)
            return fallback
        raise


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print(f"\nDataset: {file_path}")
    print(f"Shape:   {df.shape[0]} rows x {df.shape[1]} cols\n")

    result = detect_columns(df)

    print(f"Domain:            {result.get('domain')}")
    print(f"Source:            {result.get('source')}")
    print(f"Confidence:        {result.get('confidence', 0) * 100:.0f}%")
    print(f"\ndate_col:          {result['date_col']}")
    print(f"value_col:         {result['value_col']}")
    print(f"all_value_cols:    {result['all_value_cols']}")
    print(f"category_col:      {result['category_col']}")
    print(f"all_category_cols: {result['all_category_cols']}")
    print(f"id_col:                {result['id_col']}")
    print(f"high_cardinality_cols: {result.get('high_cardinality_cols', [])}")
    print(f"skip_cols:             {result['skip_cols']}")
    if result.get("llm_error"):
        print(f"\nLLM Error: {result['llm_error']}")