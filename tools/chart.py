import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def make_chart(
    df: pd.DataFrame,
    kind: str,
    x: str,
    y: str,
    title: str = "",
    output_dir: str = "outputs"
) -> dict:

    df = df.copy()

    # drop nulls for existing columns
    existing_cols = [c for c in [x, y] if c in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols)

    # convert y to numeric
    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[y])

    if len(df) == 0:
        return {
            "chart_path": None,
            "kind":       kind,
            "x":          x,
            "y":          y,
            "rows_used":  0,
            "note":       f"Skipped — no numeric data in '{y}'"
        }

    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e0a")
    ax.set_facecolor("#0c110c")

    colors_list = [
        "#c5f432", "#6ab46a", "#4a9a6a", "#2a7a4a",
        "#8acc8a", "#aae44a", "#e8f432", "#f4b432",
        "#f47832", "#f43232"
    ]

    # ── chart types ───────────────────────────────────────────────────────────

    if kind == "bar":
        plot_data = df.groupby(x)[y].sum().sort_values(ascending=False)
        if len(plot_data) > 10:
            plot_data = plot_data.head(10)
        bars = ax.bar(plot_data.index, plot_data.values, color="#c5f432")
        ax.bar_label(bars, fmt="{:,.0f}", padding=4, color="#c5f432", fontsize=8)
        if len(plot_data) > 5:
            plt.xticks(rotation=30, ha="right")

    elif kind == "line":
        if x in df.columns:
            df[x] = pd.to_datetime(df[x], errors="coerce")
            df = df.dropna(subset=[x]).sort_values(x)
            ax.plot(df[x], df[y], color="#c5f432", linewidth=2, marker="o", markersize=4)
            ax.fill_between(df[x], df[y], alpha=0.1, color="#c5f432")
        else:
            ax.plot(df[y], color="#c5f432", linewidth=2, marker="o", markersize=4)

    elif kind == "area":
        if x in df.columns:
            try:
                df[x] = pd.to_datetime(df[x], errors="coerce")
                df = df.dropna(subset=[x]).sort_values(x)
                ax.fill_between(df[x], df[y], alpha=0.4, color="#c5f432")
                ax.plot(df[x], df[y], color="#c5f432", linewidth=2)
            except Exception:
                ax.fill_between(range(len(df)), df[y], alpha=0.4, color="#c5f432")
                ax.plot(range(len(df)), df[y], color="#c5f432", linewidth=2)
        else:
            ax.fill_between(range(len(df)), df[y], alpha=0.4, color="#c5f432")
            ax.plot(range(len(df)), df[y], color="#c5f432", linewidth=2)

    elif kind == "scatter":
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df = df.dropna(subset=[x])
        ax.scatter(df[x], df[y], color="#c5f432", alpha=0.6, s=60)

    elif kind == "histogram":
        ax.hist(df[y], bins=20, color="#c5f432", edgecolor="#0a0e0a")

    elif kind == "pie":
        plot_data = df.groupby(x)[y].sum().sort_values(ascending=False).head(8)
        wedge_colors = colors_list[:len(plot_data)]
        wedges, texts, autotexts = ax.pie(
            plot_data.values,
            labels=plot_data.index,
            autopct='%1.1f%%',
            colors=wedge_colors,
            textprops={'color': '#e8f0e8', 'fontsize': 8},
            pctdistance=0.85,
        )
        for autotext in autotexts:
            autotext.set_color('#0a0e0a')
            autotext.set_fontsize(7)

    else:
        # fallback to bar
        plot_data = df.groupby(x)[y].sum().sort_values(ascending=False)
        bars = ax.bar(plot_data.index, plot_data.values, color="#c5f432")
        ax.bar_label(bars, fmt="{:,.0f}", padding=4, color="#c5f432", fontsize=8)

    # ── styling ───────────────────────────────────────────────────────────────

    chart_title = title or f"{kind} — {y} by {x}"
    ax.set_title(chart_title, color="#e8f0e8", fontsize=12, pad=12)

    if kind != "pie":
        ax.set_xlabel(x, color="#8aaa8a", fontsize=10)
        ax.set_ylabel(y, color="#8aaa8a", fontsize=10)
        ax.tick_params(colors="#6a8a6a", labelsize=8)
        ax.spines["bottom"].set_color("#1e2a1e")
        ax.spines["left"].set_color("#1e2a1e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────────

    filename   = f"{kind}_{x}_{y}.png".replace(" ", "_")
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

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # test all chart types
    print("Testing bar chart...")
    r = make_chart(df, "bar", x=df.columns[0], y=df.select_dtypes("number").columns[0])
    print(f"  → {r['chart_path']}")

    print("Testing pie chart...")
    r = make_chart(df, "pie", x=df.columns[0], y=df.select_dtypes("number").columns[0])
    print(f"  → {r['chart_path']}")

    print("Done.")