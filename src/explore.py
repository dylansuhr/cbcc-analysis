import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_monthly_data():
    """Load CSV and build three monthly time series (2023-01 to 2025-12)."""
    df = pd.read_csv("data/booking-report-23-24-25.csv")

    # parse dates and clean revenue
    df["date"] = pd.to_datetime(df["Start Date"], format="%Y-%m-%d")
    df["Total_num"] = df["Total"].replace(r"[\$,]", "", regex=True).astype(float)
    df["month"] = df["date"].dt.to_period("M")

    # complete 36-month index
    full_idx = pd.period_range("2023-01", "2025-12", freq="M")

    # 1) total revenue per month
    revenue = df.groupby("month")["Total_num"].sum().reindex(full_idx, fill_value=0)
    revenue.name = "revenue"

    # 2) Marie L charter count (any Item containing "MARIE L")
    marie = df[df["Item"].str.contains("MARIE L", case=False, na=False)]
    marie_l_charters = marie.groupby("month").size().reindex(full_idx, fill_value=0)
    marie_l_charters.name = "marie_l_charters"

    # 3) Monhegan charter count (any Item containing "MONHEGAN")
    mon = df[df["Item"].str.contains("MONHEGAN", case=False, na=False)]
    monhegan_charters = mon.groupby("month").size().reindex(full_idx, fill_value=0)
    monhegan_charters.name = "monhegan_charters"

    return revenue, marie_l_charters, monhegan_charters


def forecast_holt_winters(series, name):
    """Fit Holt-Winters and forecast 2026, then save a plot."""
    model = ExponentialSmoothing(
        series.values,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    )
    fit = model.fit()

    forecast = np.clip(fit.forecast(12), 0, None)
    forecast_idx = pd.period_range("2026-01", periods=12, freq="M")
    forecast_series = pd.Series(forecast, index=forecast_idx, name=name)

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index.to_timestamp(), series.values, marker="o", label="Historical")
    ax.plot(
        forecast_idx.to_timestamp(),
        forecast_series.values,
        marker="s",
        linestyle="--",
        color="tomato",
        label="2026 Forecast",
    )
    ax.set_title(f"Holt-Winters Forecast — {name}")
    ax.set_xlabel("Month")
    ax.set_ylabel(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"output/figures/{name}_forecast.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved output/figures/{name}_forecast.png")

    # print table
    print(f"\n2026 Forecast — {name}")
    print("-" * 30)
    for period, val in forecast_series.items():
        print(f"  {period}  {val:>12,.2f}")

    return forecast_series


def _yearly_table(series, forecast, is_dollar=False, fmt=",.0f"):
    """Build a markdown table pivoted by year with monthly rows, including forecast."""
    df = series.to_frame("value")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="month", columns="year", values="value").fillna(0)

    # append forecast as a column
    fc_vals = [max(0, v) for v in forecast.values]
    pivot["2026 (F)"] = fc_vals

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    cols = list(pivot.columns)
    header = "| Month | " + " | ".join(str(c) for c in cols) + " |"
    sep = "|-------|" + "|".join("--------:" for _ in cols) + "|"
    rows = [header, sep]
    for m in range(1, 13):
        vals = []
        for c in cols:
            v = pivot.loc[m, c] if m in pivot.index else 0
            if is_dollar:
                vals.append(f"${v:{fmt}}")
            else:
                vals.append(f"{v:{fmt}}")
        rows.append(f"| {month_names[m-1]}   | " + " | ".join(vals) + " |")
    # totals row
    totals = []
    for c in cols:
        t = pivot[c].sum()
        if is_dollar:
            totals.append(f"**${t:{fmt}}**")
        else:
            totals.append(f"**{t:{fmt}}**")
    rows.append(f"| **Total** | " + " | ".join(totals) + " |")
    return rows


def write_report(revenue, marie_l, monhegan, rev_fc, marie_fc, mon_fc):
    """Write forecast_report.md, replacing only generated content between markers."""
    BEGIN = "<!-- BEGIN GENERATED -->"
    END = "<!-- END GENERATED -->"

    generated = []
    generated += [
        "",
        "## Monthly Revenue",
        "",
        "![Revenue Forecast](figures/revenue_forecast.png)",
        "",
    ]
    generated += _yearly_table(revenue, rev_fc, is_dollar=True, fmt=",.0f")
    generated += [
        "",
        "---",
        "",
        "## Charter Count — Marie L",
        "",
        "![Marie L Charters Forecast](figures/marie_l_charters_forecast.png)",
        "",
    ]
    generated += _yearly_table(marie_l, marie_fc, is_dollar=False, fmt=".0f")
    generated += [
        "",
        "---",
        "",
        "## Charter Count — Monhegan",
        "",
        "![Monhegan Charters Forecast](figures/monhegan_charters_forecast.png)",
        "",
    ]
    generated += _yearly_table(monhegan, mon_fc, is_dollar=False, fmt=".0f")

    generated += [
        "",
        "---",
        "",
        "## Method",
        "",
        "- **Model:** `ExponentialSmoothing` from `statsmodels.tsa.holtwinters`",
        "- **Trend:** Additive",
        "- **Seasonality:** Additive (multiplicative cannot handle zero-valued off-season months)",
        "- **Seasonal periods:** 12",
        "- **Training data:** Jan 2023 – Dec 2025 (36 months, off-season months filled with 0)",
        "- **Forecast horizon:** Jan – Dec 2026",
        "- **Post-processing:** Negative forecasts clipped to 0",
        "",
    ]

    report_path = "output/forecast_report.md"
    generated_block = "\n".join(generated)

    # try to preserve user-edited content outside the markers
    try:
        with open(report_path) as f:
            existing = f.read()
        if BEGIN in existing and END in existing:
            before = existing[: existing.index(BEGIN) + len(BEGIN)]
            after = existing[existing.index(END) :]
            content = before + "\n" + generated_block + "\n" + after
        else:
            # markers missing — rebuild with default header
            content = _default_header() + BEGIN + "\n" + generated_block + "\n" + END + "\n"
    except FileNotFoundError:
        content = _default_header() + BEGIN + "\n" + generated_block + "\n" + END + "\n"

    with open(report_path, "w") as f:
        f.write(content)
    print("\nSaved output/forecast_report.md")


def _default_header():
    return (
        "# Holt-Winters 2026 Forecast Report\n"
        "\n"
        "> **Note:** All data is aggregated by **availability date** "
        "(`Start Date`), not booking creation date. Each month reflects "
        "revenue and charter counts for trips that **operated** in that "
        "month, regardless of when the reservation was made.\n"
        "\n"
        "Forecasts generated using Holt-Winters exponential smoothing "
        "(additive trend, additive seasonality, 12-month period) trained "
        "on 36 months of booking data (Jan 2023 – Dec 2025).\n"
        "\n"
        "---\n"
        "\n"
    )


def main():
    revenue, marie_l, monhegan = load_monthly_data()

    rev_fc = forecast_holt_winters(revenue, "revenue")
    marie_fc = forecast_holt_winters(marie_l, "marie_l_charters")
    mon_fc = forecast_holt_winters(monhegan, "monhegan_charters")

    write_report(revenue, marie_l, monhegan, rev_fc, marie_fc, mon_fc)


if __name__ == "__main__":
    main()
