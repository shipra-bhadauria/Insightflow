import pandas as pd

def describe_data(df: pd.DataFrame) -> dict:
    
    # basic shape
    total_rows, total_cols = df.shape
    
    # numeric columns only — mean, min, max, std
    numeric_summary = {}
    for col in df.select_dtypes(include="number").columns:
        numeric_summary[col] = {
            "mean":  round(df[col].mean(), 2),
            "min":   round(df[col].min(), 2),
            "max":   round(df[col].max(), 2),
            "std":   round(df[col].std(), 2),
            "nulls": int(df[col].isnull().sum()),
        }
    
    # column names and their data types
    column_types = {col: str(df[col].dtype) for col in df.columns}
    
    # how many rows have at least one null
    rows_with_nulls = int(df.isnull().any(axis=1).sum())

    return {
        "total_rows":      total_rows,
        "total_columns":   total_cols,
        "column_types":    column_types,
        "numeric_summary": numeric_summary,
        "rows_with_nulls": rows_with_nulls,
    }


if __name__ == "__main__":
    import sys
    
    # use file path from command line, or default to data/sales.csv
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    
    df = pd.read_csv(file_path)
    result = describe_data(df)
    
    print(f"Rows: {result['total_rows']}")
    print(f"Columns: {result['total_columns']}")
    print(f"Rows with nulls: {result['rows_with_nulls']}")
    print(f"\nColumn types:")
    for col, dtype in result['column_types'].items():
        print(f"  {col:20s} → {dtype}")
    print(f"\nNumeric summary:")
    for col, stats in result['numeric_summary'].items():
        print(f"  {col:20s} → mean: {stats['mean']}, nulls: {stats['nulls']}")