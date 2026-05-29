import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file instead of opening a window
import matplotlib.pyplot as plt
import os
import time


def make_chart(
    df: pd.DataFrame,
    kind: str,
    x: str,
    y: str,
    title: str = "",
    output_dir: str = "outputs"
) -> dict:

    df = df.copy()
    # only drop nulls for columns that actually exist in the df
    existing_cols = [c for c in [x, y] if c in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols)

    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[y])

    if len(df) == 0:
        raise ValueError(f"No numeric data in column '{y}'")

    os.makedirs(output_dir, exist_ok=True)

    # dark theme to match the UI
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 5))

    if kind == "bar":
        # group and sort before plotting
        df[y] = pd.to_numeric(df[y], errors="coerce")
        plot_data = df.groupby(x)[y].sum().sort_values(ascending=False)
        bars = ax.bar(plot_data.index, plot_data.values, color="#c5f432")
        ax.bar_label(bars, fmt="{:,.0f}", padding=4, color="#c5f432", fontsize=8)

    elif kind == "line":
        if x in df.columns:
            df = df.sort_values(x)
            df[x] = pd.to_datetime(df[x], errors="coerce")
            df = df.dropna(subset=[x])
            df = df.sort_values(x)
            ax.plot(df[x], df[y], color="#c5f432", linewidth=2, marker="o", markersize=4)
            ax.fill_between(df[x], df[y], alpha=0.1, color="#c5f432")
        else:
            # x column doesn't exist — just plot y values in order
            ax.plot(df[y], color="#c5f432", linewidth=2, marker="o", markersize=4)
            ax.fill_between(range(len(df[y])), df[y], alpha=0.1, color="#c5f432")

    elif kind == "scatter":
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df = df.dropna(subset=[x])
        ax.scatter(df[x], df[y], color="#c5f432", alpha=0.6, s=60)

    elif kind == "histogram":
        ax.hist(df[y], bins=20, color="#c5f432", edgecolor="#0a0e0a")

    # styling
    ax.set_xlabel(x, color="#8aaa8a", fontsize=10)
    ax.set_ylabel(y, color="#8aaa8a", fontsize=10)
    ax.set_title(title or f"{kind} — {y} by {x}", color="#e8f0e8", fontsize=12)
    ax.tick_params(colors="#6a8a6a", labelsize=8)
    ax.spines["bottom"].set_color("#1e2a1e")
    ax.spines["left"].set_color("#1e2a1e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#0a0e0a")
    ax.set_facecolor("#0c110c")

    # rotate x labels if there are many categories
    if kind == "bar" and len(df[x].unique()) > 5:
        plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    # save to outputs folder
    filename = f"{kind}_{x}_{y}.png".replace(" ", "_")
    chart_path = os.path.join(output_dir, filename)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "kind":       kind,
        "x":          x,
        "y":          y,
        "chart_path": chart_path,
        "rows_used":  len(df),
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"

    df = pd.read_csv(file_path)

    # test bar chart
    result = make_chart(df, kind="bar", x="Region", y="Total Revenue",
                        title="Total Revenue by Region")
    print(f"Chart saved: {result['chart_path']}")
    print(f"Rows used:   {result['rows_used']}")

    # test scatter
    result2 = make_chart(df, kind="scatter", x="Units Sold", y="Total Profit",
                         title="Units Sold vs Total Profit")
    print(f"Chart saved: {result2['chart_path']}")