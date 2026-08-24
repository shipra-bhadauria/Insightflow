import pandas as pd


def aggregate(
    df: pd.DataFrame,
    group_by,           # str ya list[str] dono accept karta hai
    value_col: str,
    agg: str = "sum",
    dropna: bool = True,
    raw: bool = False,
    value_cols: list = None,
) -> dict:

    df = df.copy()

    # large dataset sampling — token limit avoid karo
    MAX_ROWS = 10000
    if len(df) > MAX_ROWS and not raw:
        df = df.sample(n=MAX_ROWS, random_state=42)

    # value_col list aaye toh — pehla element primary, baaki value_cols me
    if isinstance(value_col, list):
        if not value_cols:
            value_cols = value_col
        value_col = value_col[0] if value_col else value_col

    # group_by — string ya list dono handle karo
    if isinstance(group_by, str):
        group_by_cols = [group_by]
    else:
        group_by_cols = list(group_by)

    if raw: 
        cols_to_show = group_by_cols.copy()

        if value_cols:
            for vc in value_cols:
                if vc in df.columns and vc not in cols_to_show:
                    cols_to_show.append(vc)
        elif value_col and value_col not in cols_to_show:
            cols_to_show.append(value_col)

        cols_to_show = [c for c in cols_to_show if c in df.columns]

        for vc in cols_to_show:
            if vc not in group_by_cols:
                df[vc] = pd.to_numeric(df[vc], errors="coerce")

        if dropna and value_col and value_col in df.columns:
            df = df.dropna(subset=[value_col])

        result_df = df[cols_to_show].sort_values(by=group_by_cols[0])

        for vc in cols_to_show:
            if vc not in group_by_cols and pd.api.types.is_numeric_dtype(result_df[vc]):
                result_df[vc] = result_df[vc].round(2)

        return {
            "group_by" : group_by,
            "value_col" : value_col,
            "agg" : "raw",
            "raw" : True,
            "rows_used": len(result_df),
            "result":result_df.to_dict(orient="records"),
        }

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
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not file_path:
        print("Usage: python aggregate.py <file_path>")
        sys.exit(1)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    print("Testing with file:", file_path)
    print("Columns:", list(df.columns))