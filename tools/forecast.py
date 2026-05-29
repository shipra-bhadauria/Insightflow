import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from prophet import Prophet


def forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 30,
    freq: str = "D",
    output_dir: str = "outputs"
) -> dict:

    df = df.copy()

    # prophet requires exactly two columns named ds and y
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(df[date_col])
    prophet_df["y"]  = pd.to_numeric(df[value_col], errors="coerce")

    # drop nulls — prophet cannot handle them
    prophet_df = prophet_df.dropna()

    # fit the model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80   # 80% confidence interval on forecast
    )
    model.fit(prophet_df)

    # create future dates dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq)

    # generate forecast
    forecast_df = model.predict(future)

    # split into historical and future portions
    last_historical_date = prophet_df["ds"].max()
    future_only = forecast_df[forecast_df["ds"] > last_historical_date]

    # build the chart
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5))

    # plot historical values as solid line
    ax.plot(
        prophet_df["ds"], prophet_df["y"],
        color="#c5f432", linewidth=2, label="Historical"
    )

    # plot forecast as dotted line
    ax.plot(
        future_only["ds"], future_only["yhat"],
        color="#c5f432", linewidth=2, linestyle="--", label="Forecast"
    )

    # confidence interval shading
    ax.fill_between(
        future_only["ds"],
        future_only["yhat_lower"],
        future_only["yhat_upper"],
        alpha=0.15, color="#c5f432", label="80% confidence"
    )

    # vertical line separating history from forecast
    ax.axvline(
        x=last_historical_date,
        color="#3a5a3a", linewidth=1,
        linestyle=":", label="Forecast start"
    )

    # styling
    ax.set_xlabel(date_col, color="#8aaa8a", fontsize=10)
    ax.set_ylabel(value_col, color="#8aaa8a", fontsize=10)
    ax.set_title(f"{value_col} — {periods} period forecast", color="#e8f0e8", fontsize=12)
    ax.tick_params(colors="#6a8a6a", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#1e2a1e")
    ax.spines["left"].set_color("#1e2a1e")
    ax.legend(fontsize=8, labelcolor="#6a8a6a", facecolor="#0c110c", edgecolor="#1e2a1e")
    fig.patch.set_facecolor("#0a0e0a")
    ax.set_facecolor("#0c110c")
    plt.tight_layout()

    filename = f"forecast_{value_col}_{periods}periods.png".replace(" ", "_")
    chart_path = os.path.join(output_dir, filename)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "date_col":        date_col,
        "value_col":       value_col,
        "periods":         periods,
        "freq":            freq,
        "rows_used":       len(prophet_df),
        "forecast_dates":  future_only["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "forecast_values": future_only["yhat"].round(2).tolist(),
        "forecast_lower":  future_only["yhat_lower"].round(2).tolist(),
        "forecast_upper":  future_only["yhat_upper"].round(2).tolist(),
        "chart_path":      chart_path,
    }


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"

    df = pd.read_csv(file_path)

    result = forecast(
        df,
        date_col="Order Date",
        value_col="Total Revenue",
        periods=30,
        freq="D"
    )

    print(f"Rows used:     {result['rows_used']}")
    print(f"Periods ahead: {result['periods']}")
    print(f"Chart saved:   {result['chart_path']}")
    print(f"\nFirst 5 forecast values:")
    for date, val in zip(result["forecast_dates"][:5], result["forecast_values"][:5]):
        print(f"  {date} → {val:,.2f}")