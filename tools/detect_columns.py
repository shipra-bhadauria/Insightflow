import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

DATE_KEYWORDS     = ["date", "time", "day", "month", "year", "period", "week"]
VALUE_KEYWORDS    = [
    "revenue", "profit", "sales", "income", "salary", "wage",
    "amount", "total", "price", "cost", "value", "fee",
    "score", "rate", "units", "quantity", "qty", "count"
]
CATEGORY_KEYWORDS = ["region", "segment", "category", "type", "status", "channel",
                     "department", "team", "group", "class", "tier", "level",
                     "country", "city", "area", "zone", "division", "product"]



SYSTEM_PROMPT = """You are a data analyst. Given a list of column names and their
data types, identify which column is most likely:
- date_col: a date or time column
- value_col: the main numeric value column (revenue, salary, score etc)
- category_col: the main grouping/category column (region, segment, department etc)
- id_col: a unique identifier column

Respond ONLY in this JSON format, no extra text:
{
  "date_col": "column_name or null",
  "value_col": "column_name or null",
  "category_col": "column_name or null",
  "id_col": "column_name or null"
}"""


def _is_text_col(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _keyword_detect(df: pd.DataFrame) -> dict:
    detected = {
        "date_col":     None,
        "value_col":    None,
        "category_col": None,
        "id_col":       None,
    }

    col_lower = {col: col.lower() for col in df.columns}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # also include string columns that look numeric
    for col in df.columns:
        if col not in numeric_cols:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > len(df) * 0.5:
                    numeric_cols.append(col)
            except Exception:
                pass
    text_cols = [col for col in df.columns if _is_text_col(df[col])]

    # date — keyword match first
    # date — keyword match first
    # require keyword to be a standalone word not part of another word
    # e.g. "year resale value" should NOT match but "order date" should
    STRICT_DATE_KEYWORDS = ["date", "time", "period", "week", "timestamp"]
    LOOSE_DATE_KEYWORDS  = ["day", "month"]

    for col, lower in col_lower.items():
        words = lower.split()
        if any(kw in words for kw in STRICT_DATE_KEYWORDS):
            detected["date_col"] = col
            break

    if not detected["date_col"]:
        for col, lower in col_lower.items():
            words = lower.split()
            if any(kw in words for kw in LOOSE_DATE_KEYWORDS):
                detected["date_col"] = col
                break

    # date — try parsing as datetime if keyword missed
    if not detected["date_col"]:
        for col in text_cols:
            try:
                pd.to_datetime(df[col].dropna().head(10))
                detected["date_col"] = col
                break
            except Exception:
                pass
    
    # validate detected date_col actually contains real dates
    if detected["date_col"]:
        try:
            sample = pd.to_numeric(df[detected["date_col"]].dropna().head(10), errors="coerce")
            if sample.notna().all():
                # all values are plain numbers — not a real date column
                detected["date_col"] = None
        except Exception:
            pass

    # value — keyword match on numeric columns
    # iterate keywords in priority order so "revenue" beats "units"
    for kw in VALUE_KEYWORDS:
        for col, lower in col_lower.items():
            if col in numeric_cols and kw in lower:
                if any(id_kw in lower for id_kw in ["id", "code", "key", "ref", "number"]):
                    continue
                detected["value_col"] = col
                break
        if detected["value_col"]:
            break

    if not detected["value_col"] and numeric_cols:
        non_id_cols = [
            col for col in numeric_cols
            if not any(kw in col.lower() for kw in ["id", "code", "key", "ref"])
        ]
        cols_to_check = non_id_cols if non_id_cols else numeric_cols
        means = {col: df[col].mean() for col in cols_to_check}
        detected["value_col"] = max(means, key=means.get)

    # value — fallback to numeric col with highest mean
    if not detected["value_col"] and numeric_cols:
        means = {col: df[col].mean() for col in numeric_cols}
        detected["value_col"] = max(means, key=means.get)

    # category — keyword match on text columns
    for col, lower in col_lower.items():
        if _is_text_col(df[col]) and any(kw in lower for kw in CATEGORY_KEYWORDS):
            detected["category_col"] = col
            break

    # category — fallback to text col with lowest unique count
    if not detected["category_col"] and text_cols:
        cardinality = {col: df[col].nunique() for col in text_cols}
        detected["category_col"] = min(cardinality, key=cardinality.get)

    # id column
    for col, lower in col_lower.items():
        if any(kw in lower for kw in ["id", "code", "key", "ref", "number"]):
            detected["id_col"] = col
            break

    return detected


def _llm_detect(df: pd.DataFrame) -> dict:
    llm = ChatOpenAI(
        model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "null"
        schema.append(f"{col} ({dtype}) — sample: {sample}")

    schema_text = "\n".join(schema)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Columns:\n{schema_text}")
    ])

    content = response.content.strip()
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    result = json.loads(content.strip())

    # only keep columns that actually exist in the df
    for key in ["date_col", "value_col", "category_col", "id_col"]:
        if result.get(key) not in df.columns:
            result[key] = None

    return result


def detect_columns(df: pd.DataFrame, use_llm_fallback: bool = True) -> dict:

    # layer 1 — fast keyword heuristic
    detected = _keyword_detect(df)

    # which slots are still empty
    missing = [k for k in ["date_col", "value_col", "category_col", "id_col"]
               if not detected.get(k)]

    # layer 2 — LLM fallback for missing slots only
    if missing and use_llm_fallback:
        llm_result = _llm_detect(df)
        for key in missing:
            if llm_result.get(key):
                detected[key] = llm_result[key]
                detected[f"{key}_source"] = "llm"

    # mark keyword detections
    for key in ["date_col", "value_col", "category_col", "id_col"]:
        if detected.get(key) and f"{key}_source" not in detected:
            detected[f"{key}_source"] = "keyword"

    # confidence score
    found = sum(1 for k in ["date_col", "value_col", "category_col", "id_col"]
                if detected.get(k))
    detected["confidence"] = round(found / 4, 2)

    detected["category_unique_values"] = (
        int(df[detected["category_col"]].nunique())
        if detected.get("category_col") else None
    )

    return detected


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"

    df = pd.read_csv(file_path)
    result = detect_columns(df)

    print("=== Smart Column Detection ===\n")
    print(f"Date column:     {result['date_col']} ({result.get('date_col_source', '?')})")
    print(f"Value column:    {result['value_col']} ({result.get('value_col_source', '?')})")
    print(f"Category column: {result['category_col']} ({result.get('category_col_source', '?')})")
    print(f"ID column:       {result['id_col']} ({result.get('id_col_source', '?')})")
    print(f"Confidence:      {result['confidence'] * 100:.0f}%")
    if result.get("category_col"):
        print(f"Category unique: {result['category_unique_values']} values")