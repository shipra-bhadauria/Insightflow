import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_file_overview(df: pd.DataFrame, detected_columns: dict) -> dict:
    """
    File upload ke baad call karo.
    Returns:
        about       : 2-3 lines about the dataset
        suggestions : 3 specific questions using actual column names
    """

    columns      = list(df.columns)
    shape        = f"{df.shape[0]} rows × {df.shape[1]} columns"
    sample_rows  = df.head(3).to_string(index=False, max_colwidth=20)

    date_col     = detected_columns.get("date_col",          "none detected")
    value_col    = detected_columns.get("value_col",         "none detected")
    category_col = detected_columns.get("category_col",      "none detected")
    all_cats     = detected_columns.get("all_category_cols", [])
    domain       = detected_columns.get("domain",            "unknown")

    prompt = f"""You are analyzing a dataset for a non-technical user.

Dataset shape  : {shape}
Domain         : {domain}
Columns        : {columns}
Date column    : {date_col}
Value column   : {value_col}
Primary category: {category_col}
All categories : {all_cats}

Sample data:
{sample_rows}

Return ONLY a JSON object — no markdown, no backticks:
{{
  "about": "2-3 sentence plain English description. Mention domain, row count, and what kind of analysis is possible.",
  "suggestions": [
    "Specific question 1 using ACTUAL column names from the data",
    "Specific question 2 using ACTUAL column names from the data",
    "Specific question 3 using ACTUAL column names from the data"
  ]
}}

Rules:
- Use ACTUAL column names in suggestions — not generic ones
- Suggestions must be questions a business user would realistically ask
- About must be under 50 words
- Do not mention technical terms like DataFrame or dtype"""

    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model       = os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 300,
            temperature = 0,
        )

        raw    = response.choices[0].message.content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        return {
            "about":       result.get("about", ""),
            "suggestions": result.get("suggestions", []),
            "shape":       shape,
            "columns":     columns,
        }

    except Exception as e:
        # fallback — no LLM
        numeric_cols  = df.select_dtypes(include="number").columns.tolist()

        fallback_suggestions = []
        if value_col != "none detected" and category_col != "none detected":
            fallback_suggestions.append(
                f"Which {category_col} has the highest {value_col}?"
            )
        if date_col != "none detected" and value_col != "none detected":
            fallback_suggestions.append(
                f"Show the {value_col} trend over {date_col}"
            )
        if len(numeric_cols) >= 2:
            fallback_suggestions.append(
                f"Is there a correlation between {numeric_cols[0]} and {numeric_cols[1]}?"
            )

        return {
            "about":       f"Dataset with {shape}. Columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}.",
            "suggestions": fallback_suggestions or ["Describe the data", "Show summary statistics"],
            "shape":       shape,
            "columns":     columns,
        }