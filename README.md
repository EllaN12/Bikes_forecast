# Bikes Sales Forecast

## Overview

A fictitious bicycle manufacturer distributes bikes to bike shops nationwide. This project analyzes sales performance for Q1–Q3 2024 and forecasts sales for Q4 2024 using two machine learning approaches: AutoARIMA and a TensorFlow LSTM deep learning model.

**Key results:** Mountain Bikes are the top-selling category nationwide. Kansas City 29ers is the top-performing shop. TensorFlow LSTM yielded slightly more accurate results (MAE: 0.236) compared to AutoARIMA (MAPE: 0.246), while AutoARIMA produced smoother, more interpretable forecast lines.

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
- ETL Automation: Data collection and cleaning is fully automated via the `collect_data()` function in `my_pandas_extensions/database.py`, handling date parsing, feature engineering, and SQLite queries.

**Feature Engineering:**
- AutoARIMA: Aggregated and summarized time-series data per shop and category using `summarize_by_time.py`.
- LSTM: Scaled and split data into training/testing sets using TensorFlow Datasets; model architecture optimized with Keras Tuner.

### Machine Learning Models

**AutoARIMA (Univariate Time Series)**
- Automatically selects optimal ARIMA order (p, d, q)
- Produces smooth, conservative forecast lines well-suited for interpretable results
- Evaluation: MAPE = 0.2458, MSPE = 0.0902

**TensorFlow LSTM (Multivariate Deep Learning)**
- Architecture: 1 LSTM layer (256 units) + 3 Dense layers (80, 16, 11 units)
- Total parameters: 889,428 (296,475 trainable, 592,953 optimizer states)
- Hyperparameters tuned with Keras Tuner to minimize validation loss
- Evaluation: MAE = 0.2360, MSE = 0.0904

**Performance Summary:** LSTM achieves a slight edge with lower MAE, but both models perform comparably on smaller-scale (thousands-range) forecasts. AutoARIMA is preferable when smooth, interpretable lines are needed.

---

## Key Findings

- **Top Category:** Mountain Bikes are the #1 selling category nationwide across all shops.
- **Top Shop:** Kansas City 29ers is the highest-grossing bike shop in the network.
- **Forecast Smoothness:** AutoARIMA produced smoother forecast lines; LSTM produced more conservative, lower-error estimates.
- **Comparable Accuracy:** Both models performed similarly for smaller regional and category-level forecasts.
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

---

## Project Structure

```
Bikes_forecast/
├── 00_data_raw/                          # Raw synthetic data files
├── 02_reports/                           # EDA profile reports (HTML)
├── 03_SRC/                               # Source code scripts
│   ├── database.py                       # ETL automation (collect_data)
│   ├── summarize_by_time.py              # Time-series aggregation utilities
│   ├── Arima_forecasting.py              # AutoARIMA forecast & evaluation
│   └── Multivariate_forecast.py          # LSTM model training & prediction
├── 04_artifacts/                         # Trained models (.h5) & predictions (.pkl, .csv)
├── 05_images/                            # Charts and figures
├── my_pandas_extensions/                 # Custom ETL extension library
├── database/                             # SQLite database
├── requirements.txt                      # Python dependencies
├── Bikes_Sales_Forecast_Presentation.pptx   # Project slide deck
└── README.md
```

---

## Key Components

- **`database.py`**: Automates data collection, cleaning, and SQLite queries via the `collect_data()` function, including date parsing, category splitting, and feature engineering.
- **`summarize_by_time.py`**: Aggregates sales data by shop, category, and time period for forecasting pipelines.
- **`Arima_forecasting.py`**: AutoARIMA forecast pipeline, evaluation functions (MAPE, MSPE), and forecast combination utilities.
- **`Multivariate_forecast.py`**: LSTM model training with TensorFlow/Keras, Keras Tuner optimization, and multivariate time-series predictions.

---

## Acknowledgments

- Pandas and NumPy for data manipulation
- Tableau Public for interactive visualization
- pmdarima (AutoARIMA) and TensorFlow/Keras for forecasting models
- Keras Tuner for hyperparameter optimization
- Scikit-learn for preprocessing utilities
