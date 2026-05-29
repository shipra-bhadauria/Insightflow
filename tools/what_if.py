import pandas as pd


def what_if(
    df: pd.DataFrame,
    col: str,
    change_pct: float,
    group_by: str = None
) -> dict:

    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])

    # original total before any change
    original_total = df[col].sum()
    original_mean  = df[col].mean()

    # apply the percentage change to the column
    multiplier = 1 + (change_pct / 100)
    df[col] = df[col] * multiplier

    # new totals after change
    new_total = df[col].sum()
    new_mean  = df[col].mean()

    # absolute impact
    impact = new_total - original_total

    # plain English summary
    direction = "increase" if change_pct > 0 else "decrease"
    impact_summary = (
        f"A {abs(change_pct)}% {direction} in {col} would "
        f"{'add' if change_pct > 0 else 'remove'} "
        f"{abs(impact):,.0f} — "
        f"total moves from {original_total:,.0f} to {new_total:,.0f}"
    )

    # optional — breakdown by group
    group_breakdown = {}
    if group_by and group_by in df.columns:
        original_groups = df.copy()
        original_groups[col] = original_groups[col] / multiplier
        for group, group_df in df.groupby(group_by):
            orig = original_groups[original_groups[group_by] == group][col].sum()
            new  = group_df[col].sum()
            group_breakdown[str(group)] = {
                "original": round(orig, 2),
                "projected": round(new, 2),
                "impact": round(new - orig, 2),
            }

    return {
        "col":              col,
        "change_pct":       change_pct,
        "rows_used":        len(df),
        "original_total":   round(original_total, 2),
        "original_mean":    round(original_mean, 2),
        "projected_total":  round(new_total, 2),
        "projected_mean":   round(new_mean, 2),
        "impact":           round(impact, 2),
        "impact_summary":   impact_summary,
        "group_breakdown":  group_breakdown,
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"

    df = pd.read_csv(file_path)

    # test 1 — what if revenue increases by 10%?
    result = what_if(df, col="Total Revenue", change_pct=10)
    print("=== What if Total Revenue increases by 10%? ===")
    print(f"Original total:  £{result['original_total']:>15,.2f}")
    print(f"Projected total: £{result['projected_total']:>15,.2f}")
    print(f"Impact:          £{result['impact']:>15,.2f}")
    print(f"\n{result['impact_summary']}")

    # test 2 — what if revenue drops by 15% broken down by region?
    print("\n=== What if Total Revenue drops by 15% — by Region? ===")
    result2 = what_if(df, col="Total Revenue", change_pct=-15, group_by="Region")
    for region, data in result2["group_breakdown"].items():
        print(f"  {region:45s} £{data['original']:>12,.0f} → £{data['projected']:>12,.0f}  ({data['impact']:>+12,.0f})")