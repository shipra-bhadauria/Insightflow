import pandas as pd


def aggregate(
    df: pd.DataFrame,
    group_by,           # str ya list[str] dono accept karta hai
    value_col: str,
    agg: str = "sum",
    dropna: bool = True
) -> dict:

    df = df.copy()

    # group_by — string ya list dono handle karo
    if isinstance(group_by, str):
        group_by_cols = [group_by]
    else:
        group_by_cols = list(group_by)

    # value_col same as group_by — count analysis
    # numeric conversion skip karo
    if value_col not in group_by_cols:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        if dropna:
            df = df.dropna(subset=[value_col])

    # groupby + aggregate
    grouped = df.groupby(group_by_cols)[value_col].agg(agg)

    # single column groupby — sort karo
    if len(group_by_cols) == 1:
        grouped = grouped.sort_values(ascending=False)
        result = {str(k): round(float(v), 2) for k, v in grouped.items()}
    else:
        # multi-column groupby — reset index, records format
        grouped_df = grouped.reset_index()
        grouped_df.columns = group_by_cols + [value_col]
        result = grouped_df.to_dict(orient="records")

    return {
        "group_by":    group_by,
        "value_col":   value_col,
        "agg":         agg,
        "dropna_used": dropna,
        "rows_used":   len(df),
        "result":      result,
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # single groupby test
    result = aggregate(df, group_by="Region", value_col="Total Revenue", agg="mean")
    print("Single groupby:")
    print(f"  Rows used: {result['rows_used']}")
    for k, v in list(result["result"].items())[:3]:
        print(f"  {k}: {v:,.2f}")

    # multi groupby test
    print("\nMulti groupby:")
    result2 = aggregate(df, group_by=["Region", "Item Type"],
                        value_col="Total Revenue", agg="sum")
    print(f"  Rows used: {result2['rows_used']}")
    print(f"  Records: {len(result2['result'])}")
    print(f"  First: {result2['result'][0]}")