"""
Forecasting logic shared by Arima_forecasting.py and Multivariate_forecasting.py:
AutoARIMA modeling, LSTM/ARIMA comparison, and evaluation metrics.

Naive baseline
--------------
Every evaluation path here (evaluate_arima_holdout, compare_arima_lstm) scores
models against a seasonal-naive forecast, not just AutoARIMA/LSTM in isolation.
The naive forecast simply repeats the last `seasonal_period` observed values
from the training data for the length of the holdout window (seasonal_period=1
is the plain "last value carried forward" naive forecast). See _naive_forecast().

It exists for two reasons:
1. A model's error number is meaningless on its own — "MAE = 34,688" tells you
   nothing about whether that's good. Comparing it to naive's MAE answers
   "is this model actually better than doing nothing clever?"
2. MASE (Mean Absolute Scaled Error, see _mase()) scales a model's MAE by the
   naive forecast's MAE on the training data. MASE < 1 means the model beats
   naive; MASE > 1 means naive would have been better. Unlike MAPE/MSPE, MASE
   doesn't explode when actuals are near zero (a real issue in this dataset at
   the bikeshop_name grouping — see outputs/forecasting.txt), so it's the
   metric to trust when MAPE numbers look absurd.

Both evaluate_arima_holdout() and compare_arima_lstm() return the naive
forecast's own MAE/MSE/RMSE/MAPE alongside AutoARIMA/LSTM (not just MASE),
so the baseline itself is visible, not just implied by a ratio.
"""

import pandas as pd
import numpy as np
import pandas_flavor as pf

from database import summarize_by_time

from sktime.forecasting.arima import AutoARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


@pf.register_dataframe_method
def arima_forecast(data, h, sp, coverage, suppress_warnings, *args, **kwargs):
    """Generates ARIMA forecasts for one or more time series.

    Args:
        data (DataFrame): Wide-format data with a pandas period time-series index.
        h (int): The forecast horizon.
        sp (int): The seasonal period.
        coverage (float): Confidence interval coverage (e.g. 0.95).
        suppress_warnings (bool): Suppresses ARIMA feedback during automated model training.
        args, kwargs: Passed to sktime.forecasting.arima.AutoARIMA.

    Returns:
        DataFrame: Columns are value, prediction, ci_lower, and ci_upper.
            Multiple time series are returned stacked by group.
    """
    df = data
    model_results_dict = {}

    for col in tqdm(df.columns, mininterval=0):
        y = df[col]

        forecaster = AutoARIMA(
            sp=sp,
            suppress_warnings=suppress_warnings,
            *args,
            **kwargs)
        forecaster.fit(y)

        y_pred_ints = forecaster.predict_interval(
            fh=np.arange(1, h + 1),
            coverage=coverage)
        y_pred = forecaster.predict()

        ret = pd.concat([y, y_pred, y_pred_ints], axis=1)
        ret.columns = ["value", "prediction", "ci_lower", "ci_upper"]

        model_results_dict[col] = ret

    model_results_df = pd.concat(model_results_dict, axis=0)

    nms = [*df.columns.names, *df.index.names]
    model_results_df.index.names = nms

    ret = model_results_df.reset_index()
    cols_to_keep = ~ret.columns.str.contains("level")
    ret = ret.loc[:, cols_to_keep]
    return ret


@pf.register_dataframe_method
def data_prep(data, group, h, LSTM_df):
    """
    Combines sales actuals, AutoARIMA, and LSTM forecasts by category,
    sub-category, and location into a single DataFrame used for visualization.

    Args:
        data (DataFrame): Original sales data.
        group (str): Grouping criteria (e.g., "category_1", "category_2", "bikeshop_name").
        h (int): Forecast horizon.
        LSTM_df (DataFrame): DataFrame containing LSTM forecast results.

    Returns:
        DataFrame: Combined DataFrame with actuals, ARIMA, and LSTM forecasts.
    """
    df1 = summarize_by_time(
        data=data,
        date_column="order_date",
        value_column="total_price",
        groups=group,
        rule="ME",
        kind="period",
        agg_func=np.sum,
        wide_format=True)

    df2 = arima_forecast(
        data=df1,
        h=h,
        sp=1,
        coverage=0.95,
        suppress_warnings=True)

    df2['order_date'] = df2['order_date'].dt.to_timestamp()

    columns_to_convert = ['value', 'prediction', 'ci_lower', 'ci_upper']
    for column in columns_to_convert:
        df2[column] = df2[column].apply(lambda x: int(x) if not pd.isna(x) else x)

    if group == "category_1":
        cat_cols = [('total_price', 'Mountain'), ('total_price', 'Road')]
        cat_df = LSTM_df[cat_cols]
    elif group == "category_2":
        excluded_cols = [('total_price', 'Mountain'), ('total_price', 'Road')]
        cat_cols = [col for col in LSTM_df.columns if col not in excluded_cols]
        cat_df = LSTM_df[cat_cols]
    else:
        cat_df = LSTM_df

    cat_df.index = df2['order_date'].iloc[-3:]

    if isinstance(cat_df.columns, pd.MultiIndex):
        value_vars = [col[1] for col in cat_df.columns]
    else:
        value_vars = list(cat_df.columns)

    lstm_df = cat_df.reset_index().melt(
        id_vars='order_date',
        value_vars=value_vars,
        var_name=group,
        value_name='LSTM prediction'
    )

    if group is not None:
        merged_df = pd.merge(df2, lstm_df, on=['order_date', group], how='outer')
        id_vars = ["order_date", "ci_lower", "ci_upper", group]
    else:
        merged_df = pd.merge(df2, lstm_df, on='order_date', how='outer')
        id_vars = ["order_date", "ci_lower", "ci_upper"]

    merged_df.rename(columns={
        'prediction': 'Arima_prediction',
        'LSTM prediction': 'LSTM_prediction',
        'value': 'Actuals'
    }, inplace=True)

    prediction_df = merged_df \
        .melt(
            value_vars=["Actuals", "Arima_prediction", "LSTM_prediction"],
            id_vars=id_vars,
            var_name="variable",
            value_name=".value") \
        .assign(order_date=lambda x: pd.to_datetime(x["order_date"].astype(str))) \
        .rename({".value": "Sales"}, axis=1)

    return prediction_df


@pf.register_dataframe_method
def extract_and_evaluate(arima_results):
    """
    Extract true values and predictions from ARIMA forecast results
    and calculate evaluation metrics.

    Args:
        arima_results (DataFrame): Results from arima_forecast function.

    Returns:
        dict: Evaluation metrics for each column/series.
    """
    metric_cols = {"value", "prediction", "ci_lower", "ci_upper"}
    group_cols = [
        c for c in arima_results.columns
        if c not in metric_cols and c != "order_date"
    ]
    group_col = group_cols[-1] if group_cols else arima_results.columns[0]

    grouped = arima_results.groupby(group_col)
    metrics_dict = {}

    for series, group in grouped:
        valid = group[["value", "prediction"]].dropna()
        if valid.empty:
            continue
        y_true = valid["value"]
        y_pred = valid["prediction"]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
        mspe = np.mean(((y_true - y_pred) / (y_true + 1e-10)) ** 2)

        metrics_dict[series] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "MSPE": mspe,
        }

    return metrics_dict


def _mase(y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series, seasonal_period: int = 1) -> float:
    """Mean Absolute Scaled Error: MAE of the forecast scaled by the MAE of a
    seasonal-naive forecast on the training data. <1 means the model beats naive."""
    naive_errors = np.abs(y_train.diff(seasonal_period).dropna())
    scale = naive_errors.mean()
    if scale == 0 or np.isnan(scale):
        return np.nan
    return mean_absolute_error(y_true, y_pred) / scale


def _naive_forecast(y_train: pd.Series, h: int, seasonal_period: int = 1) -> np.ndarray:
    """Seasonal-naive forecast: repeats the last `seasonal_period` observed
    values cyclically for h steps. With seasonal_period=1 this is the
    standard naive forecast (last value carried forward)."""
    last_season = y_train.values[-seasonal_period:]
    reps = int(np.ceil(h / seasonal_period))
    return np.tile(last_season, reps)[:h]


def _score(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE, MAPE for a single true/predicted pair."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100,
    }


def compare_arima_lstm(prediction_df: pd.DataFrame, train_actuals: pd.DataFrame, group: str, seasonal_period: int = 1) -> pd.DataFrame:
    """
    Score AutoARIMA and LSTM forecasts against the same actuals on the same
    holdout window, using the long-format output of data_prep().

    Args:
        prediction_df (DataFrame): Output of data_prep() — long format with
            columns order_date, group, variable (Actuals/Arima_prediction/LSTM_prediction), Sales.
        train_actuals (DataFrame): Wide-format historical actuals (pre-holdout) used
            to scale MASE, indexed by the same group's series.
        group (str): The grouping column used to produce prediction_df (e.g. "bikeshop_name").
        seasonal_period (int): Seasonal lag for the naive baseline used in MASE.

    Returns:
        DataFrame: One row per (series, model) with MAE, RMSE, MAPE, MASE.
    """
    wide = prediction_df.pivot_table(
        index=["order_date", group],
        columns="variable",
        values="Sales"
    ).dropna(subset=["Actuals"])

    rows = []
    for series, series_df in wide.groupby(level=group):
        y_true = series_df["Actuals"]

        if series in train_actuals.columns:
            y_train = train_actuals[series]
        else:
            y_train = y_true

        for model_col, model_name in [
            ("Arima_prediction", "AutoARIMA"),
            ("LSTM_prediction", "LSTM"),
        ]:
            if model_col not in series_df.columns:
                continue
            paired = series_df[["Actuals", model_col]].dropna()
            if paired.empty:
                continue

            y_t, y_p = paired["Actuals"], paired[model_col]
            scores = _score(y_t, y_p)
            scores["MASE"] = _mase(y_t, y_p, y_train, seasonal_period)

            rows.append({"series": series, "model": model_name, **scores})

        naive_pred = _naive_forecast(y_train, h=len(y_true), seasonal_period=seasonal_period)
        naive_scores = _score(y_true, naive_pred)
        naive_scores["MASE"] = 1.0

        rows.append({"series": series, "model": "Naive", **naive_scores})

    return pd.DataFrame(rows)


def format_metrics(metrics_dict: dict) -> str:
    """Pretty-print AutoARIMA vs. naive-baseline metrics for file or console output."""
    lines = ["AutoARIMA vs. naive baseline", "=" * 40]
    for series, by_model in metrics_dict.items():
        lines.append(f"\n[{series}]")
        for model_name, metrics in by_model.items():
            lines.append(f"  {model_name}:")
            for name, value in metrics.items():
                lines.append(f"    {name}: {value:.6f}")
    return "\n".join(lines)


def evaluate_arima_holdout(
    wide: pd.DataFrame,
    h: int = 3,
    sp: int = 1,
    suppress_warnings: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Train AutoARIMA on all but the last h periods, forecast h steps,
    and compare predictions to held-out actuals.
    """
    result_rows = []
    metrics_dict = {}

    for col in tqdm(wide.columns, mininterval=0):
        y = wide[col].dropna()
        if len(y) <= h:
            continue

        y_train = y.iloc[:-h]
        y_test = y.iloc[-h:]

        forecaster = AutoARIMA(sp=sp, suppress_warnings=suppress_warnings)
        forecaster.fit(y_train)
        fh = np.arange(1, h + 1)
        y_pred = forecaster.predict(fh=fh)
        y_pred.index = y_test.index
        try:
            y_pred_ints = forecaster.predict_interval(fh=fh, coverage=0.95)
            ci_lower = y_pred_ints.iloc[:, 0].values
            ci_upper = y_pred_ints.iloc[:, 1].values
        except Exception:
            ci_lower = np.full(h, np.nan)
            ci_upper = np.full(h, np.nan)

        naive_pred = _naive_forecast(y_train, h=h, seasonal_period=sp)
        naive_std = np.std(y_train.diff(sp).dropna())

        for i, (dt, actual, pred, naive) in enumerate(zip(y_test.index, y_test.values, y_pred.values, naive_pred)):
            result_rows.append({
                "series": str(col),
                "order_date": dt,
                "value": actual,
                "prediction": pred,
                "ci_lower": ci_lower[i],
                "ci_upper": ci_upper[i],
                "naive_prediction": naive,
                "naive_ci_lower": naive - 1.96 * naive_std,
                "naive_ci_upper": naive + 1.96 * naive_std,
            })

        arima_scores = _score(y_test, y_pred)
        arima_scores["MSPE"] = np.mean(((y_test.values - y_pred.values) / (y_test.values + 1e-10)) ** 2)
        arima_scores["MASE"] = _mase(y_test, y_pred, y_train, seasonal_period=sp)

        naive_scores = _score(y_test, naive_pred)
        naive_scores["MSPE"] = np.mean(((y_test.values - naive_pred) / (y_test.values + 1e-10)) ** 2)
        naive_scores["MASE"] = 1.0  # naive is the scale itself, so MASE is 1.0 by construction

        metrics_dict[str(col)] = {
            "AutoARIMA": arima_scores,
            "Naive": naive_scores,
        }

    return pd.DataFrame(result_rows), metrics_dict


def run_arima_evaluation(group: str, h: int = 3) -> dict:
    """Load data, forecast with AutoARIMA hold-out, and return evaluation metrics."""
    from database import collect_data, summarize_by_time

    df = collect_data()
    wide = summarize_by_time(
        data=df,
        date_column="order_date",
        value_column="total_price",
        groups=group,
        rule="ME",
        kind="period",
        agg_func=np.sum,
        wide_format=True,
    )
    _, metrics = evaluate_arima_holdout(
        wide=wide,
        h=h,
        sp=1,
        suppress_warnings=True,
    )
    return metrics


def main() -> None:
    all_metrics = {}
    for group in ("category_1", "category_2", "bikeshop_name"):
        print(f"\nRunning AutoARIMA for {group}…", flush=True)
        all_metrics[group] = run_arima_evaluation(group=group, h=3)
        print(format_metrics(all_metrics[group]), flush=True)

    print("\n" + "=" * 40, flush=True)
    print("Summary by grouping: AutoARIMA vs. naive baseline", flush=True)
    print("(MASE < 1 beats naive; MAPE/MSPE are unreliable when actuals are near zero, e.g. at the bikeshop level)", flush=True)
    for group, metrics in all_metrics.items():
        arima = [m["AutoARIMA"] for m in metrics.values()]
        naive = [m["Naive"] for m in metrics.values()]

        arima_mase = np.nanmean([m["MASE"] for m in arima])
        arima_mape = np.nanmean([m["MAPE"] for m in arima])
        arima_mae = np.nanmean([m["MAE"] for m in arima])

        naive_mape = np.nanmean([m["MAPE"] for m in naive])
        naive_mae = np.nanmean([m["MAE"] for m in naive])

        print(f"  {group}:")
        print(f"    AutoARIMA: MAE={arima_mae:.4f}, MAPE={arima_mape:.4f}, MASE={arima_mase:.4f}", flush=True)
        print(f"    Naive:     MAE={naive_mae:.4f}, MAPE={naive_mape:.4f}, MASE=1.0000", flush=True)


if __name__ == "__main__":
    main()
