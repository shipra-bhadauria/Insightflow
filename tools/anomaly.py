import pandas as pd

def detect_anomaly(df: pd.DataFrame, col: str) -> dict:
    
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    
    # IQR method — standard way to detect outliers
    # IQR = the middle 50% of data (Q3 - Q1)
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # anything below Q1-1.5*IQR or above Q3+1.5*IQR is an outlier
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # find the outlier rows
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    return {
        "col":          col,
        "rows_used":    len(df),
        "Q1":           round(Q1, 2),
        "Q3":           round(Q3, 2),
        "IQR":          round(IQR, 2),
        "lower_bound":  round(lower_bound, 2),
        "upper_bound":  round(upper_bound, 2),
        "outlier_count": len(outliers),
        "outlier_pct":  round(len(outliers) / len(df) * 100, 2),
        "outliers":     outliers[col].round(2).tolist(),
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    
    df = pd.read_csv(file_path)
    result = detect_anomaly(df, col="Total Revenue")
    
    print(f"Column: {result['col']}")
    print(f"Rows analysed: {result['rows_used']}")
    print(f"Normal range: {result['lower_bound']:,.2f} → {result['upper_bound']:,.2f}")
    print(f"Outliers found: {result['outlier_count']} ({result['outlier_pct']}%)")
    print(f"\nOutlier values:")
    for val in result["outliers"]:
        print(f"  → {val:,.2f}")