import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

try:
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def make_chart(
    df: pd.DataFrame,
    kind: str,
    x: str,
    y: str,
    title: str = "",
    output_dir: str = "outputs",
    agg: str = "sum",   # sum / mean / count
) -> dict:

    df = df.copy()

    # drop nulls for existing columns
    existing_cols = [c for c in [x, y] if c in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols)

    # count agg mein y numeric hone ki zaroorat nahi (x == y ho sakta hai, category column)
    if agg != "count":
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
        if agg == "mean":
            plot_data = df.groupby(x)[y].mean().sort_values(ascending=False)
        elif agg == "count":
            plot_data = df.groupby(x)[y].count().sort_values(ascending=False)
        else:
            plot_data = df.groupby(x)[y].sum().sort_values(ascending=False)
        if len(plot_data) > 10:
            plot_data = plot_data.head(10)
        bars = ax.bar(plot_data.index, plot_data.values, color="#c5f432")
        max_val = plot_data.max() if len(plot_data) > 0 else 1
        fmt = "{:,.2f}" if max_val < 100 else "{:,.0f}"
        ax.bar_label(bars, fmt=fmt, padding=4, color="#c5f432", fontsize=8)
        if len(plot_data) > 5:
            plt.xticks(rotation=30, ha="right")

        elif kind == "line":
            if x in df.columns:
                df[x] = pd.to_datetime(df[x], errors="coerce")
                df = df.dropna(subset=[x]).sort_values(x)
                df = df.groupby(x)[y].mean().reset_index()
                ax.plot(df[x], df[y], color="#c5f432", linewidth=2, marker="o", markersize=4)
                ax.fill_between(df[x], df[y], alpha=0.1, color="#c5f432")
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.xticks(rotation=30, ha="right")
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
        if agg == "count":
            plot_data = df.groupby(x)[y].count().sort_values(ascending=False).head(8)
        elif agg == "mean":
            plot_data = df.groupby(x)[y].mean().sort_values(ascending=False).head(8)
        else:
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
        #plot_data = df.groupby(x)[y].sum().sort_values(ascending=False)
        if agg == "mean":
            plot_data = df.groupby(x)[y].mean().sort_values(ascending=False).head(8)
        elif agg == "count":
            plot_data = df.groupby(x)[y].count().sort_values(ascending=False).head(8)
        else:
            plot_data = df.groupby(x)[y].sum().sort_values(ascending=False).head(8)
        bars = ax.bar(plot_data.index, plot_data.values, color="#c5f432")
        max_val = plot_data.max() if len(plot_data) > 0 else 1
        fmt = "{:,.2f}" if max_val < 100 else "{:,.0f}"
        ax.bar_label(bars, fmt=fmt, padding=4, color="#c5f432", fontsize=8)
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
    plt.savefig(chart_path, dpi=90)
    plt.close(fig)

    return {
        "kind":       kind,
        "x":          x,
        "y":          y,
        "chart_path": chart_path,
        "rows_used":  len(df),
    }


def make_plotly_chart(
    df: pd.DataFrame,
    kind: str,
    x: str,
    y: str,
    title: str = "",
    agg: str = "sum",
):
    """Interactive Plotly chart — returns fig for st.plotly_chart()"""
    if not _PLOTLY_AVAILABLE:
        return None

    df = df.copy()
    existing = [c for c in [x, y] if c in df.columns]
    if existing:
        df = df.dropna(subset=existing)

    # count agg mein y numeric hone ki zaroorat nahi (x == y ho sakta hai, category column)
    if agg != "count" and y in df.columns:
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna(subset=[y])

    if len(df) == 0:
        return None

    chart_title = title or f"{y} by {x}"

    try:
        if kind == "bar":
            if agg == "mean":
                plot_data = df.groupby(x)[y].mean().sort_values(ascending=False).head(20).reset_index()
            elif agg == "count":
                plot_data = df.groupby(x)[y].count().sort_values(ascending=False).head(20).reset_index()
            else:
                plot_data = df.groupby(x)[y].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(plot_data, x=x, y=y, title=chart_title,
                         color_discrete_sequence=["#c5f432"],
                         template="plotly_dark")

        elif kind == "line":
            df[x] = pd.to_datetime(df[x], errors="coerce")
            df = df.dropna(subset=[x]).sort_values(x)
            plot_data = df.groupby(x)[y].mean().reset_index()
            fig = px.line(plot_data, x=x, y=y, title=chart_title,
                          color_discrete_sequence=["#c5f432"],
                          template="plotly_dark")

        elif kind == "pie":
            if agg == "count":
                plot_data = df.groupby(x)[y].count().reset_index()
            else:
                plot_data = df.groupby(x)[y].sum().reset_index()
            plot_data = plot_data.nlargest(8, y)
            fig = px.pie(plot_data, names=x, values=y, title=chart_title,
                         template="plotly_dark")

        elif kind == "scatter":
            fig = px.scatter(df, x=x, y=y, title=chart_title,
                             color_discrete_sequence=["#c5f432"],
                             template="plotly_dark")

        elif kind == "histogram":
            fig = px.histogram(df, x=x if x in df.columns else y,
                               title=chart_title,
                               color_discrete_sequence=["#c5f432"],
                               template="plotly_dark")

        else:
            plot_data = df.groupby(x)[y].sum().reset_index()
            fig = px.bar(plot_data, x=x, y=y, title=chart_title,
                         template="plotly_dark")

        fig.update_layout(
            paper_bgcolor="#0a0e0a",
            plot_bgcolor="#0a0e0a",
            font_color="#c5f432",
            title_font_color="#c5f432",
        )
        return fig

    except Exception as e:
        print(f"[plotly] chart error: {e}")
        return None


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