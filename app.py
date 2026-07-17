"""
Streamlit dashboard for the Bikes Sales Forecast project.

Replaces the static Tableau dashboard with an interactive app that reads
directly from the project's SQLite database and forecasting pipeline
(02_SRC/database.py, 02_SRC/forecasting.py) — no separate BI tool needed.

Run with: streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_SRC"))

st.set_page_config(page_title="Bikes Sales Forecast", layout="wide")

# Precomputed artifacts (written by 02_SRC/precompute_forecasts.py). Loading
# them avoids retraining ARIMA models — and importing sktime/pmdarima at all —
# on every cold start. The heavy imports below happen only in the fallbacks.
PRECOMPUTED_DIR = os.path.join("03_outputs", "precomputed")


@st.cache_data(show_spinner="Loading sales data…")
def load_data() -> pd.DataFrame:
    cached = os.path.join(PRECOMPUTED_DIR, "data.pkl")
    if os.path.exists(cached):
        return pd.read_pickle(cached)
    from database import collect_data  # slow: rebuilds the DB from Excel
    return collect_data()


@st.cache_data(show_spinner="Loading AutoARIMA holdout metrics…")
def load_arima_metrics(group: str, h: int = 3) -> pd.DataFrame:
    """Holdout metrics (series, model, MAE/RMSE/MASE) for the given group."""
    cached = os.path.join(PRECOMPUTED_DIR, f"arima_holdout_metrics_{group}.pkl")
    if os.path.exists(cached):
        return pd.read_pickle(cached)
    # Fallback: train from scratch (slow) — run precompute_forecasts.py to avoid
    from precompute_forecasts import compute_arima_metrics
    return compute_arima_metrics(load_data(), group, h=h)


@st.cache_data(show_spinner=False)
def load_lstm_predictions():
    """
    Load LSTM per-period predictions (3 holdout months × 30 bikeshops).
    Returns a DataFrame indexed 0–2 with bikeshop columns, or None if not yet generated.
    """
    pred_path = os.path.join("03_outputs", "Multivariate_time_series_predictions")
    if not os.path.exists(pred_path):
        return None
    try:
        return pd.read_pickle(pred_path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def lstm_predictions_for_group(group: str, h: int = 3):
    """
    Map bikeshop-level LSTM predictions to the requested grouping.

    - bikeshop_name : return pred_df as-is (columns = shop names)
    - category_1/2  : weight-sum bikeshop predictions by each shop's historical
                      share of that category's revenue, giving approximate
                      category-level LSTM forecasts.

    Returns a dict {series_name: np.array of length h} or empty dict.
    """
    pred_df = load_lstm_predictions()
    if pred_df is None or pred_df.empty:
        return {}

    raw = load_data()

    if group == "bikeshop_name":
        return {col: pred_df[col].values for col in pred_df.columns if col in pred_df.columns}

    # Build a shop → category revenue share table
    shop_cat = (
        raw.groupby(["bikeshop_name", group])["total_price"]
        .sum()
        .reset_index()
    )
    total_per_cat = shop_cat.groupby(group)["total_price"].transform("sum")
    shop_cat["share"] = shop_cat["total_price"] / total_per_cat

    result = {}
    for cat in shop_cat[group].unique():
        cat_shops = shop_cat[shop_cat[group] == cat]
        lstm_cat = np.zeros(len(pred_df))
        for _, row in cat_shops.iterrows():
            shop = row["bikeshop_name"]
            if shop in pred_df.columns:
                lstm_cat += pred_df[shop].values * row["share"]
        result[cat] = lstm_cat
    return result


@st.cache_data(show_spinner="Loading Q1 2025 forecasts…")
def load_forward_forecast(group: str, h: int = 3, year_offset: int = 9):
    """
    Forward forecast for the trend chart. Loads the precomputed artifact when
    available; otherwise trains on all data (see precompute_forecasts.py).

    Returns dict with keys:
      history   : DataFrame(order_date, series, value)
      arima     : DataFrame(order_date, series, prediction, ci_lower, ci_upper)
      naive     : DataFrame(order_date, series, naive_prediction)
    """
    cached = os.path.join(PRECOMPUTED_DIR, f"forward_{group}.pkl")
    if os.path.exists(cached):
        return pd.read_pickle(cached)
    # Fallback: train from scratch (slow) — run precompute_forecasts.py to avoid
    from precompute_forecasts import compute_forward_forecast
    return compute_forward_forecast(load_data(), group, h=h, year_offset=year_offset)


df = load_data()

st.title("Bikes Sales Forecast")
st.caption("Sales performance analysis & AutoARIMA / LSTM forecast evaluation")

# Pre-compute year/month columns used across KPI sections
df["_year"]  = df["order_date"].dt.year
df["_month"] = df["order_date"].dt.month
max_yr    = int(df["_year"].max())   # 2015 → displayed as 2024
prior_yr  = max_yr - 1               # 2014 → displayed as 2023
prev2_yr  = max_yr - 2               # 2013 → displayed as 2022

# Sparkline window: Dec of prior year through Dec of current year (13 points)
# Labels: "Dec '23", "Jan", "Feb", ..., "Nov", "Dec '24"
spark_labels = ["Dec '23"] + ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"] + ["Dec '24"]
prior_labels = ["Dec '22"] + ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"] + ["Dec '23"]

def _dec_to_dec(df_in, year_start, year_end, col, agg):
    """Return 13 values: Dec of year_start, then Jan–Dec of year_end."""
    dec_prev = df_in[(df_in["_year"] == year_start) & (df_in["_month"] == 12)]
    full_cur  = df_in[df_in["_year"] == year_end]
    if agg == "sum":
        v0  = dec_prev[col].sum()
        rest = full_cur.groupby("_month")[col].sum().reindex(range(1, 13), fill_value=0).values
    elif agg == "nunique":
        v0  = dec_prev[col].nunique()
        rest = full_cur.groupby("_month")[col].nunique().reindex(range(1, 13), fill_value=0).values
    else:  # count
        v0  = len(dec_prev)
        rest = full_cur.groupby("_month").size().reindex(range(1, 13), fill_value=0).values
    return np.concatenate([[v0], rest])


def _sparkline(cur_vals, prev_vals, labels, color, fmt=",.0f", prefix="", suffix="", height=120,
               cur_range=None, prev_range=None):
    """
    Return a minimal Plotly sparkline with start/end date labels on each line.
    cur_range  : (start_label, end_label) for the solid line, e.g. ("Dec '23", "Dec '24")
    prev_range : (start_label, end_label) for the dotted line, e.g. ("Dec '22", "Dec '23")
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=prev_vals, mode="lines",
        line=dict(color="rgba(150,150,150,0.45)", width=1.5, dash="dot"),
        showlegend=False,
        hovertemplate=f"%{{x}}<br>Prior: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=cur_vals, mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5),
        showlegend=False,
        hovertemplate=f"%{{x}}<br>Current: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
    ))

    annotations = []
    label_font_cur  = dict(size=9, color=color)
    label_font_prev = dict(size=9, color="rgba(130,130,130,0.85)")

    if cur_range:
        annotations += [
            dict(x=0, y=0, xref="paper", yref="paper",
                 text=cur_range[0], showarrow=False,
                 xanchor="left", yanchor="bottom", yshift=-2,
                 font=label_font_cur),
            dict(x=1, y=0, xref="paper", yref="paper",
                 text=cur_range[1], showarrow=False,
                 xanchor="right", yanchor="bottom", yshift=-2,
                 font=label_font_cur),
        ]
    if prev_range:
        annotations += [
            dict(x=0, y=1, xref="paper", yref="paper",
                 text=prev_range[0], showarrow=False,
                 xanchor="left", yanchor="top", yshift=2,
                 font=label_font_prev),
            dict(x=1, y=1, xref="paper", yref="paper",
                 text=prev_range[1], showarrow=False,
                 xanchor="right", yanchor="top", yshift=2,
                 font=label_font_prev),
        ]

    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=14, b=14),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        annotations=annotations,
    )
    return fig


def _yoy(cur_total, prev_total):
    if not prev_total:
        return 0.0
    return (cur_total - prev_total) / prev_total * 100


# ── Revenue ───────────────────────────────────────────────────────────────
rev_cur  = _dec_to_dec(df, prior_yr,  max_yr,   "total_price", "sum")
rev_prev = _dec_to_dec(df, prev2_yr,  prior_yr,  "total_price", "sum")
rev_ytd_cur  = rev_cur.sum()
rev_ytd_prev = rev_prev.sum()

# ── Units sold ────────────────────────────────────────────────────────────
units_cur  = _dec_to_dec(df, prior_yr, max_yr,  "quantity", "sum")
units_prev = _dec_to_dec(df, prev2_yr, prior_yr, "quantity", "sum")
units_ytd_cur  = int(units_cur.sum())
units_ytd_prev = int(units_prev.sum())

# ── Active shops ──────────────────────────────────────────────────────────
shops_cur  = _dec_to_dec(df, prior_yr, max_yr,  "bikeshop_name", "nunique")
shops_prev = _dec_to_dec(df, prev2_yr, prior_yr, "bikeshop_name", "nunique")
shops_ytd_cur  = int(shops_cur.max())   # peak active shops in window
shops_ytd_prev = int(shops_prev.max())

# ── Order lines ───────────────────────────────────────────────────────────
orders_cur  = _dec_to_dec(df, prior_yr, max_yr,  "_month", "count")
orders_prev = _dec_to_dec(df, prev2_yr, prior_yr, "_month", "count")
orders_ytd_cur  = int(orders_cur.sum())
orders_ytd_prev = int(orders_prev.sum())

# ── KPI card layout ───────────────────────────────────────────────────────

kc1, kc2, kc3, kc4 = st.columns(4)

with kc1:
    pct = _yoy(rev_ytd_cur, rev_ytd_prev)
    st.metric(
        "Revenue  Dec '23 – Dec '24",
        f"${rev_ytd_cur:,.0f}",
        f"{'▲' if pct>=0 else '▼'} {abs(pct):.1f}% vs PY",
        delta_color="normal" if pct >= 0 else "inverse",
    )
    st.plotly_chart(
        _sparkline(rev_cur, rev_prev, spark_labels, "#1f77b4", prefix="$",
                   cur_range=("Dec '23", "Dec '24"), prev_range=("Dec '22", "Dec '23")),
        use_container_width=True, key="spark_rev",
    )

with kc2:
    pct = _yoy(units_ytd_cur, units_ytd_prev)
    st.metric(
        "Units Sold  Dec '23 – Dec '24",
        f"{units_ytd_cur:,}",
        f"{'▲' if pct>=0 else '▼'} {abs(pct):.1f}% vs PY",
        delta_color="normal" if pct >= 0 else "inverse",
    )
    st.plotly_chart(
        _sparkline(units_cur, units_prev, spark_labels, "#ff7f0e",
                   cur_range=("Dec '23", "Dec '24"), prev_range=("Dec '22", "Dec '23")),
        use_container_width=True, key="spark_units",
    )

with kc3:
    pct = _yoy(shops_ytd_cur, shops_ytd_prev)
    st.metric(
        "Active Shops  Dec '23 – Dec '24",
        f"{shops_ytd_cur:,}",
        f"{'▲' if pct>=0 else '▼'} {abs(pct):.1f}% vs PY",
        delta_color="normal" if pct >= 0 else "inverse",
    )
    st.plotly_chart(
        _sparkline(shops_cur, shops_prev, spark_labels, "#2ca02c",
                   cur_range=("Dec '23", "Dec '24"), prev_range=("Dec '22", "Dec '23")),
        use_container_width=True, key="spark_shops",
    )

with kc4:
    pct = _yoy(orders_ytd_cur, orders_ytd_prev)
    st.metric(
        "Order Lines  Dec '23 – Dec '24",
        f"{orders_ytd_cur:,}",
        f"{'▲' if pct>=0 else '▼'} {abs(pct):.1f}% vs PY",
        delta_color="normal" if pct >= 0 else "inverse",
    )
    st.plotly_chart(
        _sparkline(orders_cur, orders_prev, spark_labels, "#9467bd",
                   cur_range=("Dec '23", "Dec '24"), prev_range=("Dec '22", "Dec '23")),
        use_container_width=True, key="spark_orders",
    )

df.drop(columns=["_year", "_month"], inplace=True)

st.divider()

# --- Sales Breakdown: Treemap (hierarchy + sub-category share via color) --
st.header("Sales Breakdown")

total_revenue = df["total_price"].sum()

cat1_sales = df.groupby("category_1")["total_price"].sum().reset_index()
cat1_sales["pct_total"] = cat1_sales["total_price"] / total_revenue * 100

cat2_sales = df.groupby(["category_1", "category_2"])["total_price"].sum().reset_index()
cat2_sales = cat2_sales.merge(
    cat1_sales[["category_1", "total_price"]].rename(columns={"total_price": "cat1_total"}),
    on="category_1",
)
cat2_sales["pct_total"]    = cat2_sales["total_price"] / total_revenue * 100
cat2_sales["pct_category"] = cat2_sales["total_price"] / cat2_sales["cat1_total"] * 100

# Color scheme:
#   Mountain sub-categories → blue family, shade = pct_category (darker = larger share)
#   Road sub-categories     → orange family, shade = pct_category
#   Parent nodes            → solid anchor colors
def _shade(hex_dark: str, hex_light: str, pct: float) -> str:
    """Interpolate between light and dark hex based on 0–100 pct."""
    t = max(0.25, min(1.0, pct / 100))
    def _parse(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    d, l = _parse(hex_dark), _parse(hex_light)
    r = tuple(int(l[i] + t * (d[i] - l[i])) for i in range(3))
    return f"rgb({r[0]},{r[1]},{r[2]})"

MOUNTAIN_DARK  = "#1a4f8a"   # deep blue
MOUNTAIN_LIGHT = "#bdd7f0"   # pale blue
ROAD_DARK      = "#b84c00"   # deep orange
ROAD_LIGHT     = "#fdd5a8"   # pale orange
PARENT_COLORS  = {"Mountain": "#2166ac", "Road": "#e07020"}

node_ids      = []
node_labels   = []
node_parents  = []
node_values   = []
node_colors   = []
node_customdata = []

# Root node (invisible, required for branchvalues="total")
root_total = total_revenue
node_ids.append("All")
node_labels.append("All")
node_parents.append("")
node_values.append(root_total)
node_colors.append("rgba(0,0,0,0)")
node_customdata.append([100.0, 100.0])

for _, row in cat1_sales.iterrows():
    cat = row["category_1"]
    node_ids.append(cat)
    node_labels.append(cat)
    node_parents.append("All")
    node_values.append(row["total_price"])
    node_colors.append(PARENT_COLORS.get(cat, "#888888"))
    node_customdata.append([row["pct_total"], 100.0])

for _, row in cat2_sales.iterrows():
    cat1 = row["category_1"]
    cat2 = row["category_2"]
    pct  = row["pct_category"]
    if cat1 == "Mountain":
        color = _shade(MOUNTAIN_DARK, MOUNTAIN_LIGHT, pct)
    else:
        color = _shade(ROAD_DARK, ROAD_LIGHT, pct)
    node_ids.append(f"{cat1}|{cat2}")
    node_labels.append(cat2)
    node_parents.append(cat1)
    node_values.append(row["total_price"])
    node_colors.append(color)
    node_customdata.append([row["pct_total"], pct])

treemap_fig = go.Figure(go.Treemap(
    ids=node_ids,
    labels=node_labels,
    parents=node_parents,
    values=node_values,
    branchvalues="total",
    marker=dict(colors=node_colors),
    customdata=node_customdata,
    texttemplate=(
        "<b>%{label}</b><br>"
        "$%{value:,.0f}<br>"
        "%{customdata[0]:.1f}% of total<br>"
        "%{customdata[1]:.1f}% of category"
    ),
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Sales: $%{value:,.0f}<br>"
        "% of Total: %{customdata[0]:.1f}%<br>"
        "% of Category: %{customdata[1]:.1f}%"
        "<extra></extra>"
    ),
    root_color="white",
))
treemap_fig.update_layout(
    title=(
        "Sales Hierarchy — Mountain (blue) · Road (orange) · "
        "Darker shade = larger share of parent category"
    ),
    margin=dict(t=50, l=0, r=0, b=0),
    height=480,
)
st.plotly_chart(treemap_fig, use_container_width=True)

st.divider()

# --- Sales Trend & Forecast (Jan 2020 – Dec 2024 actuals + Q1 2025 forecast) -
st.header("Sales Trend & Forecast")

trend_group_map = {"Category": "category_1", "Sub-Category": "category_2", "Bikeshop": "bikeshop_name"}
trend_group_label = st.selectbox("Group by", list(trend_group_map.keys()), key="trend_group")
trend_group = trend_group_map[trend_group_label]

# Load forward forecasts (cached)
fwd = load_forward_forecast(trend_group, h=3, year_offset=9)
hist_long = fwd["history"]
arima_df  = fwd["arima"]
naive_df  = fwd["naive"]

# Series picker
all_series = sorted(hist_long["series"].unique())
trend_series = st.selectbox("Select series", all_series, key="trend_series")

# Filter to chosen series
h_ser = hist_long[hist_long["series"] == trend_series].sort_values("order_date")
a_ser = arima_df[arima_df["series"] == trend_series].sort_values("order_date")
n_ser = naive_df[naive_df["series"] == trend_series].sort_values("order_date")

# LSTM forward forecast
lstm_preds_map = lstm_predictions_for_group(trend_group, h=3)
lstm_vals = lstm_preds_map.get(trend_series, None)

# Build forecast dates: 3 months after last historical date
last_hist_date = h_ser["order_date"].max()
fc_dates = pd.date_range(last_hist_date + pd.offsets.MonthEnd(1), periods=3, freq="ME")

trend_fig = go.Figure()

# Actuals (solid line)
trend_fig.add_trace(go.Scatter(
    x=h_ser["order_date"], y=h_ser["value"],
    mode="lines", name="Actuals",
    line=dict(color="#1f77b4", width=2.5),
    hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Actuals</extra>",
))

# AutoARIMA forecast + CI band
if not a_ser.empty:
    trend_fig.add_trace(go.Scatter(
        x=pd.concat([a_ser["order_date"], a_ser["order_date"].iloc[::-1]]),
        y=pd.concat([a_ser["ci_upper"], a_ser["ci_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    trend_fig.add_trace(go.Scatter(
        x=a_ser["order_date"], y=a_ser["prediction"],
        mode="lines+markers", name="AutoARIMA",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=7),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>AutoARIMA</extra>",
    ))

# Naive forecast + band
if not n_ser.empty:
    trend_fig.add_trace(go.Scatter(
        x=n_ser["order_date"], y=n_ser["naive_prediction"],
        mode="lines+markers", name="Naive",
        line=dict(color="#7f7f7f", width=1.5, dash="dot"),
        marker=dict(size=6),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Naive</extra>",
    ))

# LSTM forecast
if lstm_vals is not None and len(lstm_vals) >= 3:
    trend_fig.add_trace(go.Scatter(
        x=fc_dates, y=lstm_vals[-3:],
        mode="lines+markers", name="LSTM",
        line=dict(color="#9467bd", width=2, dash="dashdot"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>LSTM</extra>",
    ))

# Vertical divider at forecast start
trend_fig.add_vline(
    x=last_hist_date + pd.offsets.MonthEnd(1),
    line=dict(color="rgba(100,100,100,0.5)", width=1.5, dash="dot"),
    annotation_text="Forecast →", annotation_position="top right",
)

trend_fig.update_layout(
    xaxis_title=None,
    yaxis=dict(title="Revenue ($)", tickprefix="$", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=420,
    margin=dict(t=50, b=20),
)
st.plotly_chart(trend_fig, use_container_width=True)

st.divider()

# --- Model Performance Comparison -----------------------------------------
comparison_path = os.path.join("03_outputs", "arima_vs_lstm_comparison.csv")
st.header("Model Performance Comparison")
st.markdown(
    "Holdout evaluation across all bikeshops (last 3 months held out). "
    "**MASE < 1** means the model beats the naïve baseline. "
    "**Lower MAE / RMSE = better.** MAPE is excluded — near-zero monthly sales cause division-by-zero distortion."
)

# Pull AutoARIMA + Naive metrics from the precomputed holdout evaluation
arima_metrics_df = load_arima_metrics(group="bikeshop_name", h=3)

# If LSTM rows exist in the comparison CSV, merge them in
if os.path.exists(comparison_path):
    csv_df = pd.read_csv(comparison_path)
    lstm_rows = csv_df[csv_df["model"] == "LSTM"][["series", "model", "MAE", "RMSE", "MASE"]]
    base_cols = ["series", "model", "MAE", "RMSE", "MASE"]
    arima_sub = arima_metrics_df[base_cols] if all(c in arima_metrics_df.columns for c in base_cols) else arima_metrics_df[["series", "model", "MAE", "RMSE", "MASE"]]
    all_metrics = pd.concat([arima_sub, lstm_rows], ignore_index=True) if not lstm_rows.empty else arima_sub
else:
    all_metrics = arima_metrics_df[["series", "model", "MAE", "RMSE", "MASE"]]

# ── Overall summary (mean across all series) ──────────────────────────────
st.subheader("Overall Mean Metrics by Model")

summary = (
    all_metrics.groupby("model")[["MAE", "RMSE", "MASE"]]
    .mean()
    .sort_values("MASE")
    .reset_index()
)

# Rename columns for display
summary_display = summary.copy()
summary_display["MAE"]  = summary_display["MAE"].map("${:,.0f}".format)
summary_display["RMSE"] = summary_display["RMSE"].map("${:,.0f}".format)
summary_display["MASE"] = summary_display["MASE"].map("{:.3f}".format)
summary_display = summary_display.rename(columns={"model": "Model", "MAE": "MAE (avg)", "RMSE": "RMSE (avg)", "MASE": "MASE (avg)"})

def _highlight_model_row(row):
    model = row["Model"]
    if model == "AutoARIMA":
        return ["background-color: #fff3e0; color: #7a3e00"] * len(row)
    elif model == "LSTM":
        return ["background-color: #f3e5f5; color: #4a148c"] * len(row)
    else:
        return ["background-color: #f5f5f5; color: #333333"] * len(row)

st.dataframe(
    summary_display.style.apply(_highlight_model_row, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ── MASE comparison bar chart ─────────────────────────────────────────────
st.subheader("MASE by Model (lower = better, < 1 beats naïve)")

mase_bar = go.Figure()
model_colors = {"AutoARIMA": "#ff7f0e", "Naive": "#7f7f7f", "LSTM": "#9467bd"}
for _, row in summary.iterrows():
    mase_bar.add_trace(go.Bar(
        x=[row["model"]], y=[row["MASE"]],
        name=row["model"],
        marker_color=model_colors.get(row["model"], "#1f77b4"),
        text=[f"{row['MASE']:.3f}"],
        textposition="outside",
    ))
mase_bar.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.6,
                   annotation_text="Naïve baseline (MASE = 1)", annotation_position="top right")
mase_bar.update_layout(
    height=320, margin=dict(t=30, b=20),
    yaxis=dict(title="Mean MASE", range=[0, summary["MASE"].max() * 1.25]),
    showlegend=False, bargap=0.4,
)
st.plotly_chart(mase_bar, use_container_width=True)
