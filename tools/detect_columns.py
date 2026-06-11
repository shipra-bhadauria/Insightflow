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
  "date_col":          "column_name or null",
  "value_col":         "column_name or null",
  "all_value_cols":    ["col1", "col2"],
  "category_col":      "column_name or null",
  "all_category_cols": ["col1", "col2", "col3", "col4", "col5"],
  "id_col":            "column_name or null",
  "skip_cols":         ["col1"],
  "domain":            "sales/hr/hospital/finance/logistics/ecommerce/other"
}

STRICT RULES:

1. date_col
   → column with actual dates or timestamps
   → NOT year-only integers like 2020, 2021

2. value_col
   → PRIMARY numeric column worth summing or averaging
   → examples: revenue, billing amount, salary, score, price

3. all_value_cols
   → ALL numeric columns worth analyzing — aim for 2-4
   → EXCLUDE: id numbers, room numbers, phone numbers, zip codes
   → INCLUDE: age, score, rating, count, quantity — even if integers

4. category_col
   → PRIMARY grouping column — most analytically useful

5. all_category_cols
   → MOST IMPORTANT FIELD — minimum 4-6 columns, aim for 5-8
   → INCLUDE any column where:
      * unique values are between 2 and 100
      * OR unique values list shows short labels/words/codes
      * examples: Gender, Status, Type, Result, Grade, Level,
        Region, Department, Category, Blood Type, Condition,
        Outcome, Rating, Priority, Stage — anything for GROUP BY
   → EXCLUDE only:
     * full person names, email addresses, street addresses, URLs
     * Quality/rating labels applied to products: POOR/GOOD/EXCELLENT, 
       LOW/MEDIUM/HIGH when describing product quality scores
     * Derived metric categories that don't make business sense for GROUP BY
       e.g. "Retention Value: POOR/GOOD" — this is a quality label, not a category
   → INCLUDE:
     * Institution names, company names, hospital names
     * Geographic groupings: city, country, region
     * Business categories: product type, department, status
     * Medical/HR categories: condition, blood type, admission type
   → RULE: Ask yourself "Does grouping by this column give meaningful business insight?"
     If YES → include. If NO → skip_cols.

6. id_col
   → unique row identifier — unique count close to total rows

7. skip_cols
   → ONLY these: full names, email addresses, street addresses, URLs
   → DO NOT skip columns with 2-100 unique categorical values
   → DO NOT skip columns just because they seem less important

8. domain → what kind of dataset is this?

KEY INSIGHT: Look at the "unique values" list provided for each column.
If you see short repeating labels like Normal/Abnormal/Inconclusive,
Yes/No, Male/Female, Active/Inactive — that column is a CATEGORY,
not a skip column."""


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
    for key in ["all_value_cols", "all_category_cols", "skip_cols"]:
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

    return {
        "date_col":          None,
        "value_col":         value_col,
        "all_value_cols":    [value_col] if value_col else [],
        "category_col":      category_col,
        "all_category_cols": all_cats[:8],
        "id_col":            None,
        "skip_cols":         [],
        "domain":            "other",
        "confidence":        0.40,
        "source":            "fallback",
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
    print(f"id_col:            {result['id_col']}")
    print(f"skip_cols:         {result['skip_cols']}")
    if result.get("llm_error"):
        print(f"\nLLM Error: {result['llm_error']}")