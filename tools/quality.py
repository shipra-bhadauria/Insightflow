import pandas as pd
import numpy as np


def run_quality_report(df: pd.DataFrame) -> dict:

    total_rows, total_cols = df.shape

    # null count per column
    null_counts = df.isnull().sum().to_dict()

    # total null cells across entire file
    total_nulls = sum(null_counts.values())

    # duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # column types — categorise each column
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif df[col].nunique() / len(df) < 0.05:
            # fewer than 5% unique values = likely a category
            column_types[col] = "category"
        else:
            column_types[col] = "text"

    # outlier detection per numeric column using IQR
    outlier_columns = []
    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        Q1  = clean.quantile(0.25)
        Q3  = clean.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((clean < lower) | (clean > upper)).sum())
        if n_outliers > 0:
            outlier_columns.append({
                "column":    col,
                "n_outliers": n_outliers,
                "pct":       round(n_outliers / len(clean) * 100, 2),
            })

    # completeness score — 1.0 = perfect, 0.0 = all nulls
    total_cells = total_rows * total_cols
    completeness_score = round(1 - (total_nulls / total_cells), 4) if total_cells > 0 else 0.0

    # overall health label
    if completeness_score >= 0.98:
        health_label = "excellent"
    elif completeness_score >= 0.90:
        health_label = "good"
    elif completeness_score >= 0.75:
        health_label = "fair"
    else:
        health_label = "poor"

    return {
        "total_rows":        total_rows,
        "total_columns":     total_cols,
        "null_counts":       null_counts,
        "total_nulls":       total_nulls,
        "duplicate_rows":    duplicate_rows,
        "column_types":      column_types,
        "outlier_columns":   outlier_columns,
        "completeness_score": completeness_score,
        "health_label":      health_label,
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"

    df = pd.read_csv(file_path)
    result = run_quality_report(df)

    print(f"=== Data Quality Report ===\n")
    print(f"Rows:              {result['total_rows']}")
    print(f"Columns:           {result['total_columns']}")
    print(f"Total nulls:       {result['total_nulls']}")
    print(f"Duplicate rows:    {result['duplicate_rows']}")
    print(f"Completeness:      {result['completeness_score'] * 100:.1f}%")
    print(f"Health:            {result['health_label'].upper()}")

    print(f"\nColumn types:")
    for col, dtype in result["column_types"].items():
        nulls = result["null_counts"].get(col, 0)
        print(f"  {col:30s} {dtype:10s} nulls: {nulls}")

    print(f"\nOutlier columns:")
    if result["outlier_columns"]:
        for item in result["outlier_columns"]:
            print(f"  {item['column']:30s} {item['n_outliers']} outliers ({item['pct']}%)")
    else:
        print("  none found")