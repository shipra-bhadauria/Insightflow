import pandas as pd

def correlate(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    
    df = df.copy()
    
    # keep only rows where both columns have values
    df[col_a] = pd.to_numeric(df[col_a], errors="coerce")
    df[col_b] = pd.to_numeric(df[col_b], errors="coerce")
    df = df.dropna(subset=[col_a, col_b])
    
    # pearson correlation — standard measure of linear relationship
    # returns a value between -1 and +1
    correlation = df[col_a].corr(df[col_b])
    correlation = round(correlation, 4)
    
    # translate the number into plain English
    abs_corr = abs(correlation)
    if abs_corr >= 0.8:
        strength = "strong"
    elif abs_corr >= 0.5:
        strength = "moderate"
    elif abs_corr >= 0.3:
        strength = "weak"
    else:
        strength = "negligible"
    
    direction = "positive" if correlation > 0 else "negative"
    
    return {
        "col_a":       col_a,
        "col_b":       col_b,
        "correlation": correlation,
        "strength":    strength,
        "direction":   direction,
        "rows_used":   len(df),
        "interpretation": f"{strength} {direction} relationship between {col_a} and {col_b}",
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    
    df = pd.read_csv(file_path)
    
    result = correlate(df, col_a="Units Sold", col_b="Total Revenue")
    
    print(f"Columns: {result['col_a']} vs {result['col_b']}")
    print(f"Correlation: {result['correlation']}")
    print(f"Strength: {result['strength']}")
    print(f"Direction: {result['direction']}")
    print(f"Rows used: {result['rows_used']}")
    print(f"Interpretation: {result['interpretation']}")