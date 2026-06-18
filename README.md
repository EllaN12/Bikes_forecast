# Bikes Sales Forecast

## Overview

A fictitious bicycle manufacturer distributes bikes to bike shops nationwide. This project analyzes sales performance for Q1–Q3 2024 and forecasts sales for Q4 2024 using two machine learning approaches: AutoARIMA and a TensorFlow LSTM deep learning model.

**Key results:** Mountain Bikes are the top-selling category nationwide. Kansas City 29ers is the top-performing shop. AutoARIMA beats a naive (last-value-carried-forward) baseline at every level of aggregation — MASE of 0.45 (category), 0.57 (sub-category), and 0.79 (bikeshop) — while TensorFlow LSTM produces more conservative, lower-error estimates on the same forecasts. See [Model Evaluation](#model-evaluation) for why MASE, not MAPE, is the metric to trust here.

---

## Data Sources

A synthetic sales dataset generated using Python, stored in a SQLite database with three interconnected tables:

- **Bikes** (4,753 records, 4 attributes): Catalog of bike products including names, descriptions, and category information.
- **Bikeshops** (93 records, 3 attributes): Registry of bike stores across the continental United States with location and business details.
- **Orderlines** (109,514 records, 6 attributes): Complete order history documenting bike purchases by bike shops over 5 years.

---

## Methods and Tools

### Methodology

The CRISP-DM framework was applied end-to-end:

**Data Understanding:** Exploratory data analysis (EDA) using pandas-profiling reports to understand dataset characteristics and temporal patterns.

**Data Preparation:**
- Cleaning: Fields such as date, location, and description were split into new features (bike main categories, sub-categories, frame materials).
- ETL Automation: Data collection and cleaning is fully automated via the `collect_data()` function in `database.py`, handling date parsing, feature engineering, and SQLite queries.

**Feature Engineering:**
- AutoARIMA: Aggregated and summarized time-series data per shop and category using the `summarize_by_time()` function in `database.py`.
- LSTM: Scaled and split data into training/testing sets using TensorFlow Datasets; model architecture optimized with Keras Tuner.

### Machine Learning Models

**AutoARIMA (Univariate Time Series)**
- Automatically selects optimal ARIMA order (p, d, q)
- Produces smooth, conservative forecast lines well-suited for interpretable results
- Evaluated on a true holdout (last 3 months), not in-sample: MASE = 0.45 (category_1), 0.57 (category_2), 0.79 (bikeshop_name)

**TensorFlow LSTM (Multivariate Deep Learning)**
- Architecture: 1 LSTM layer (256 units) + 3 Dense layers (80, 16, 11 units)
- Total parameters: 889,428 (296,475 trainable, 592,953 optimizer states)
- Hyperparameters tuned with Keras Tuner to minimize validation loss
- Scored against the same holdout actuals and the same naive baseline as AutoARIMA via `compare_arima_lstm()`

**Performance Summary:** AutoARIMA beats a naive forecast at every grouping level (MASE < 1 throughout). LSTM tends to produce more conservative, lower-error point estimates on the same series, while AutoARIMA gives smoother, more interpretable forecast lines with confidence intervals.

---

## Model Evaluation

Every model is scored against a **naive baseline** (last value carried forward, see `_naive_forecast()` in `forecasting.py`) on a true train/holdout split — not just against each other. This answers the question a raw error number can't: *is the model actually better than doing nothing clever?*

- **MASE (Mean Absolute Scaled Error)** is the headline metric: a model's MAE divided by the naive baseline's MAE on the training data. **MASE < 1 means the model beats naive; MASE > 1 means naive would have won.**
- **MAPE/MSPE are reported too, but are unreliable at the `bikeshop_name` grouping** — some bikeshop-month actuals are at or near zero, which makes percentage-based metrics explode (literal quadrillions in `outputs/forecasting.txt`). MASE doesn't have this failure mode, since it scales by the naive forecast's *average* error rather than dividing by each individual actual.
- `evaluate_arima_holdout()` (in `forecasting.py`) reports AutoARIMA's and Naive's metrics side by side per series and per grouping.
- `compare_arima_lstm()` (in `forecasting.py`) scores AutoARIMA, LSTM, and Naive against the *same* holdout actuals using the merged output of `data_prep()`, writing a combined report to `outputs/arima_vs_lstm_comparison.csv`.

---

## Key Findings

- **Top Category:** Mountain Bikes are the #1 selling category nationwide across all shops.
- **Top Shop:** Kansas City 29ers is the highest-grossing bike shop in the network.
- **Beats Naive:** AutoARIMA outperforms a naive forecast at every grouping level (MASE 0.45–0.79), confirming the model adds real value over a "no model" baseline.
- **Forecast Smoothness:** AutoARIMA produced smoother forecast lines; LSTM produced more conservative, lower-error estimates.
- **MAPE Caveat:** Reported MAPE/MSPE numbers are unreliable at the bikeshop level due to near-zero actuals — MASE is the trustworthy metric there.
- **Visualization:** Interactive Tableau dashboard — [View on Tableau Public](https://public.tableau.com/app/profile/ella.claude/viz/BikesslaesForecast/Story1)

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Docker (optional)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/EllaN12/Bikes_forecast.git
   cd Bikes_forecast
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Scripts

All Python source lives together in `03_SRC/`, so `Arima_forecasting.py` and `Multivariate_forecasting.py` can import `database.py` and `forecasting.py` as plain sibling modules — no `PYTHONPATH` setup needed:

```bash
python 03_SRC/Arima_forecasting.py
python 03_SRC/Multivariate_forecasting.py
```

---

## Project Structure

```
Bikes_forecast/
├── 00_data_raw/                          # Raw synthetic data files
├── 03_SRC/                               # All Python source, run as-is
│   ├── database.py                       # collect_data(), summarize_by_time()
│   ├── forecasting.py                    # arima_forecast, data_prep, naive baseline, MASE, compare_arima_lstm
│   ├── Arima_forecasting.py              # AutoARIMA forecast & holdout evaluation
│   └── Multivariate_forecasting.py       # LSTM training & ARIMA/LSTM/naive comparison
├── outputs/                              # All file outputs: trained models (.h5), predictions (.pkl, .csv), EDA report (.html)
├── database/                             # SQLite database
├── requirements.txt                      # Python dependencies
├── Bikes_Sales_Forecast_Presentation.pptx   # Project slide deck
└── README.md
```

---

## Key Components

- **`database.py`**: Automates data collection, cleaning, and SQLite queries via the `collect_data()` function, including date parsing, category splitting, and feature engineering. Also defines `summarize_by_time()`, which aggregates sales data by shop, category, and time period for forecasting pipelines.
- **`forecasting.py`**: Shared `arima_forecast`, `data_prep`, and `extract_and_evaluate` functions — single source of truth for ARIMA modeling and evaluation logic. Also home to the naive-baseline evaluation framework: `_naive_forecast()`, `_mase()`, `evaluate_arima_holdout()`, and `compare_arima_lstm()`. See [Model Evaluation](#model-evaluation).
- **`Arima_forecasting.py`**: Imports the shared functions from `forecasting.py` to run the AutoARIMA holdout evaluation (`forecasting.main()`), printing and saving AutoARIMA-vs-naive metrics per grouping.
- **`Multivariate_forecasting.py`**: LSTM model training with TensorFlow/Keras and Keras Tuner optimization, then scores AutoARIMA, LSTM, and the naive baseline against the same holdout actuals via `compare_arima_lstm()`. Writes all outputs to `outputs/`, including `arima_vs_lstm_comparison.csv`.

---

## Acknowledgments

- Pandas and NumPy for data manipulation
- Tableau Public for interactive visualization
- pmdarima (AutoARIMA) and TensorFlow/Keras for forecasting models
- Keras Tuner for hyperparameter optimization
- Scikit-learn for preprocessing utilities
