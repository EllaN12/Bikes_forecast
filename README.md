# Bikes Sales Forecast

## Overview

A fictitious bicycle manufacturer distributes bikes to bike shops nationwide. This project analyzes sales performance across the full 2020–2024 period and forecasts Q1 2025 revenue using two machine learning approaches: AutoARIMA and a TensorFlow LSTM deep learning model. Results are surfaced through an interactive Streamlit dashboard deployed on Google Cloud Run.

**Key results:** Mountain Bikes are the top-selling category nationwide. Both models beat a naïve baseline — LSTM achieves a mean MASE of 0.703 and AutoARIMA 0.778 on a 3-month holdout across all 30 bikeshops. MASE, not MAPE, is the headline metric — see [Model Evaluation](#model-evaluation) for why.

---

## Live App

The dashboard is deployed on Google Cloud Run:

> **https://bikes-forecast-6trxjbynbq-uc.a.run.app**  
---

## Data Sources

A synthetic sales dataset generated using Python, stored in a SQLite database (`01_database/`) with three interconnected tables:

- **Bikes** (4,753 records): Catalog of bike products — names, descriptions, categories.
- **Bikeshops** (93 records): Registry of bike stores across the continental US.
- **Orderlines** (109,514 records): Complete order history over 5 years (2020–2024 display).

Raw source files live in `00_data_raw/`.

---

## Streamlit Dashboard

`app.py` is a fully self-contained Streamlit app with four sections:

| Section | What it shows |
|---------|--------------|
| **KPI Cards** | Revenue, Units Sold, Active Shops, Order Lines — Dec '23–Dec '24 vs Dec '22–Dec '23 (prior year dotted sparkline) |
| **Sales Breakdown** | Treemap of revenue by Category and Sub-Category. Mountain = blue shades, Road = orange shades; shade intensity reflects share of parent category |
| **Sales Trend & Forecast** | Full historical actuals (Jan 2020–Dec 2024) + Q1 2025 forecasts for AutoARIMA, Naive, and LSTM. Dropdown for Category / Sub-Category / Bikeshop |
| **Model Performance Comparison** | Mean MAE, RMSE, MASE per model; MASE bar chart vs naïve baseline |

Run locally:

```bash
streamlit run app.py
```

---

## Methods and Tools

### Methodology

The CRISP-DM framework was applied end-to-end:

**Data Understanding:** Exploratory data analysis using pandas-profiling.

**Data Preparation:**
- Date parsing, category/location splitting, and feature engineering automated via `collect_data()` in `database.py`.
- Time-series aggregation (monthly, by shop/category) via `summarize_by_time()`.

**Feature Engineering:**
- AutoARIMA: Aggregated monthly series per grouping using `summarize_by_time()`.
- LSTM: Min-Max scaled, windowed (3-month look-back) multivariate input across all 30 bikeshops simultaneously. Train/val/test split applied before scaling to prevent data leakage.

### Machine Learning Models

**AutoARIMA (Univariate Time Series)**
- Automatically selects optimal ARIMA (p, d, q) order via sktime
- Independently fit per series at bikeshop, category, and sub-category level
- 95% confidence intervals via `predict_interval()`
- Holdout MASE: **0.778** (mean across 30 bikeshops)

**TensorFlow LSTM (Multivariate Deep Learning)**
- Architecture: `Input(3, 30) → LSTM(32) → Dense(30)`
- Trained jointly across all 30 bikeshop revenue series
- 3-month sliding window; trained for up to 30 epochs with early stopping (patience=5)
- Pre-computed predictions committed to `04_outputs/` — TensorFlow is **not** a runtime dependency of the deployed app
- Holdout MASE: **0.703** (mean across 30 bikeshops)

---

## Model Evaluation

Every model is scored against a **naïve baseline** (last value carried forward) on a true holdout — the last 3 months of data — not in-sample.

- **MASE** is the headline metric: model MAE ÷ naïve MAE on training data. **MASE < 1 beats naïve; MASE > 1 means naïve would have won.**
- **MAPE is excluded** from the final dashboard — near-zero bikeshop months cause percentage errors in the quadrillions, making the metric meaningless for this dataset.
- `evaluate_arima_holdout()` scores AutoARIMA and Naive side-by-side per series.
- `compare_arima_lstm()` aligns LSTM test predictions to the same holdout window as AutoARIMA (via `evaluate_arima_holdout`), writing a combined report to `04_outputs/arima_vs_lstm_comparison.csv`.

| Model | MAE (avg) | RMSE (avg) | MASE (avg) |
|-------|-----------|------------|------------|
| **LSTM** | $31,981 | $38,165 | **0.703** |
| AutoARIMA | $34,688 | $39,689 | 0.778 |
| Naïve | $26,612 | $36,551 | 1.000 |

---

## Key Findings

- **Top Category:** Mountain Bikes are the #1 selling category nationwide across all shops.
- **Both models beat naïve:** LSTM (MASE 0.703) and AutoARIMA (MASE 0.778) both outperform the naïve baseline on a true holdout.
- **LSTM wins overall:** Lower MAE, RMSE, and MASE than AutoARIMA on the bikeshop-level holdout.
- **AutoARIMA advantage:** Produces confidence intervals and per-series forecasts without a GPU; easier to interpret and retrain.
- **MAPE excluded:** Unreliable for sparse bikeshop data — MASE is the correct metric here.

---

## Project Structure

```
Bikes_forecast/
├── 00_data_raw/                        # Raw Excel source files
├── 01_database/                        # SQLite database
├── 02_SRC/                             # Python source
│   ├── database.py                     # collect_data(), summarize_by_time()
│   ├── forecasting.py                  # arima_forecast, evaluate_arima_holdout, compare_arima_lstm, _naive_forecast
│   ├── Arima_forecasting.py            # AutoARIMA holdout evaluation script
│   └── Multivariate_forecasting.py     # LSTM training & ARIMA/LSTM/naive comparison
├── 03_outputs/                         # Trained model (.h5), predictions (.pkl), comparison CSV
├── app.py                              # Streamlit dashboard
├── requirements.txt                    # App runtime dependencies (no TensorFlow)
├── requirements-train.txt              # Full deps for local LSTM training (includes TensorFlow)
├── Dockerfile                          # Cloud Run container (Python 3.12-slim, 512Mi)
├── .dockerignore                       # Excludes .venv, logs, hyperband artifacts
├── cloudbuild.yaml                     # CI/CD: Cloud Build → Artifact Registry → Cloud Run
├── deploy.sh                           # One-shot manual deploy script
└── README.md
```

---

## Installation and Setup

### Local development

```bash
git clone https://github.com/EllaN12/Bikes_forecast.git
cd Bikes_forecast
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt   # includes TensorFlow for retraining
streamlit run app.py
```

### Retrain the LSTM model

```bash
python 02_SRC/Multivariate_forecasting.py
```

Outputs saved to `04_outputs/`: `time_series_model.h5`, `Multivariate_time_series_predictions`, `arima_vs_lstm_comparison.csv`. Commit these files before redeploying so the live app picks up updated predictions.

---

## Deployment (Google Cloud Run)

The app runs in a lightweight Docker container (~400 MB) with no TensorFlow — LSTM predictions are pre-computed and bundled in `04_outputs/`.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- GCP project with billing enabled

### One-shot deploy

```bash
./deploy.sh YOUR_PROJECT_ID us-central1
```

The script:
1. Enables Artifact Registry and Cloud Run APIs
2. Creates an Artifact Registry Docker repo (idempotent)
3. Builds the image for `linux/amd64` (required for Apple Silicon)
4. Pushes to Artifact Registry
5. Deploys to Cloud Run (`512Mi`, `min-instances=0`, scales to zero when idle)
6. Prints the live URL

### Automated CI/CD

Connect the GitHub repo to **Cloud Build** (GCP Console → Cloud Build → Triggers) and point to `cloudbuild.yaml`. Every push to `main` triggers a build and redeploy automatically.

### Cost

With `min-instances=0` and low traffic the app stays within Cloud Run's permanent free tier (2M requests + 360K GB-seconds/month).

---

## Acknowledgements

- pandas, NumPy, sktime, pmdarima — data manipulation and AutoARIMA
- TensorFlow / Keras — LSTM model
- Plotly / Streamlit — interactive dashboard
- scikit-learn — preprocessing
- Google Cloud Run — deployment
