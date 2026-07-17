"""
Precompute the dashboard's forecast artifacts.

The underlying sales data is static, so the AutoARIMA forecasts and holdout
metrics never change between deploys. Computing them at app runtime made every
Cloud Run cold start retrain ~30 ARIMA models before the first page render.
This script trains everything once and writes the results to
03_outputs/precomputed/; the Streamlit app loads the artifacts from disk and
only falls back to training if an artifact is missing.

Re-run after the underlying data or forecasting code changes:

    python 02_SRC/precompute_forecasts.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parent.parent
PRECOMPUTED_DIR = ROOT / "03_outputs" / "precomputed"

GROUPS = ("category_1", "category_2", "bikeshop_name")


def compute_forward_forecast(raw: pd.DataFrame, group: str, h: int = 3, year_offset: int = 9):
    """
    Train on ALL available data (no holdout) and forecast h steps ahead.
    Dates are shifted by year_offset so the data displays as ending Dec 2024
    and the forecast covers Q1 2025.

    Returns dict with keys:
      history   : DataFrame(order_date, series, value)
      arima     : DataFrame(order_date, series, prediction, ci_lower, ci_upper)
      naive     : DataFrame(order_date, series, naive_prediction)
    """
    from database import summarize_by_time
    from forecasting import arima_forecast, _naive_forecast

    wide = summarize_by_time(
        data=raw, date_column="order_date", value_column="total_price",
        groups=group, rule="ME", kind="period", agg_func=np.sum, wide_format=True,
    )

    offset = pd.DateOffset(years=year_offset)

    # ARIMA forecast on full series (all groups stacked)
    arima_raw = arima_forecast(
        data=wide, h=h, sp=1, coverage=0.95, suppress_warnings=True
    )
    # arima_forecast names the series column after the group (e.g. "category_1")
    series_col = [c for c in arima_raw.columns if c not in ("order_date", "value", "prediction", "ci_lower", "ci_upper")][0]
    arima_raw = arima_raw.rename(columns={series_col: "series"})
    arima_raw["order_date"] = pd.to_datetime(arima_raw["order_date"].astype(str)) + offset
    # Keep only forecast rows (where value is NaN = future months)
    arima_df = arima_raw[arima_raw["value"].isna()].copy()

    # History from wide (long format)
    hist_long = wide.reset_index().melt(
        id_vars="order_date", var_name="series", value_name="value"
    )
    hist_long["order_date"] = pd.to_datetime(hist_long["order_date"].astype(str)) + offset

    # Naive forecast per series
    naive_rows = []
    for col in wide.columns:
        y = wide[col].dropna()
        if y.empty:
            continue
        preds = _naive_forecast(y, h=h, seasonal_period=1)
        last = pd.to_datetime(str(y.index[-1])) + offset
        fc_dates = pd.date_range(last + pd.offsets.MonthEnd(1), periods=h, freq="ME")
        for dt, val in zip(fc_dates, preds):
            naive_rows.append({"series": str(col), "order_date": dt, "naive_prediction": val})
    naive_df = pd.DataFrame(naive_rows)

    return {"history": hist_long, "arima": arima_df, "naive": naive_df}


def compute_arima_metrics(raw: pd.DataFrame, group: str, h: int = 3) -> pd.DataFrame:
    """AutoARIMA holdout evaluation → tidy metrics DataFrame (series, model, scores)."""
    from database import summarize_by_time
    from forecasting import evaluate_arima_holdout

    wide = summarize_by_time(
        data=raw, date_column="order_date", value_column="total_price",
        groups=group, rule="ME", kind="period", agg_func=np.sum, wide_format=True,
    )
    _, metrics = evaluate_arima_holdout(wide=wide, h=h, sp=1, suppress_warnings=True)
    metric_rows = []
    for series, by_model in metrics.items():
        for model_name, scores in by_model.items():
            metric_rows.append({"series": series, "model": model_name, **scores})
    return pd.DataFrame(metric_rows)


def main() -> None:
    from database import collect_data

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting data…")
    raw = collect_data()
    raw.to_pickle(PRECOMPUTED_DIR / "data.pkl")
    print(f"  saved data.pkl ({len(raw):,} rows)")

    for group in GROUPS:
        print(f"Forward forecast: {group}…")
        fwd = compute_forward_forecast(raw, group, h=3, year_offset=9)
        pd.to_pickle(fwd, PRECOMPUTED_DIR / f"forward_{group}.pkl")
        print(f"  saved forward_{group}.pkl ({fwd['arima']['series'].nunique()} series)")

    # The Model Performance section evaluates the bikeshop-level holdout
    print("Holdout evaluation: bikeshop_name…")
    metrics_df = compute_arima_metrics(raw, "bikeshop_name", h=3)
    metrics_df.to_pickle(PRECOMPUTED_DIR / "arima_holdout_metrics_bikeshop_name.pkl")
    print(f"  saved arima_holdout_metrics_bikeshop_name.pkl ({len(metrics_df)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
