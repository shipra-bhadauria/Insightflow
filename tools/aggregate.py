import pandas as pd


def aggregate(
    df: pd.DataFrame,
    group_by: str,
    value_col: str,
    agg: str = "sum",
    dropna: bool = True
) -> dict:

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if dropna:
        df = df.dropna(subset=[value_col])

    grouped = df.groupby(group_by)[value_col].agg(agg)
    grouped = grouped.sort_values(ascending=False)

    return {
        "group_by":   group_by,
        "value_col":  value_col,
        "agg":        agg,
        "dropna_used": dropna,
        "rows_used":  len(df),
        "result":     {str(k): round(float(v), 2) for k, v in grouped.items()},
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    df = pd.read_csv(file_path)
    result = aggregate(df, group_by="Region", value_col="Total Revenue", agg="mean")
    print(f"Rows used: {result['rows_used']}")
    for k, v in result["result"].items():
        print(f"  {k}: {v:,.2f}")
