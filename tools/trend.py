import pandas as pd


def trend_over_time(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "ME",
    dropna: bool = True
) -> dict:

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    resampled = df[value_col].resample(freq).sum()

    pct_change = resampled.pct_change() * 100
    pct_change = pct_change.replace([float('inf'), float('-inf')], float('nan'))
    pct_change = pct_change.round(2)
    pct_change = pct_change.where(pct_change.notna(), None)

    first_val = resampled.iloc[0] if len(resampled) > 0 else 0
    last_val  = resampled.iloc[-1] if len(resampled) > 0 else 0

    if first_val == 0:
        overall_change_pct = 0.0
    else:
        overall_change_pct = round(((last_val - first_val) / first_val) * 100, 2)

    direction = "up" if overall_change_pct > 0 else "down" if overall_change_pct < 0 else "flat"

    periods = {}
    for ts, val in resampled.items():
        key = str(ts)
        pct = pct_change.get(ts)
        periods[key] = {
            "value":      round(float(val), 2) if pd.notna(val) else 0,
            "pct_change": round(float(pct), 2) if pct is not None and pd.notna(pct) else None,
        }

    return {
        "date_col":          date_col,
        "value_col":         value_col,
        "freq":              freq,
        "rows_used":         len(df),
        "periods":           periods,
        "overall_change_pct": overall_change_pct,
        "direction":         direction,
        "first_value":       round(float(first_val), 2),
        "last_value":        round(float(last_val), 2),
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    df = pd.read_csv(file_path)
    result = trend_over_time(df, date_col="Order Date", value_col="Total Revenue")
    print(f"Direction:      {result['direction']}")
    print(f"Overall change: {result['overall_change_pct']}%")
    print(f"Rows used:      {result['rows_used']}")
