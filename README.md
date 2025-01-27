# Bikes_forecast

## Overview
A fictitious bicycle manufacturer distributes bikes to bikeshops nationwide. The project aims to analyze sales performance for the first three quarters of 2024 and forecast sales for the final quarter.
Mountain bikes are top selling category of bikes nationwide. Kansas city 29rs is the top selling shop. TensorFlow forecasting yielded more accurate and conservative results compared to AutoARIMA.



## Data Sources
- synthetic sales dataset generated using python
- SQLite Database with 3 Interconnected Tables:
 1. Bikes (4,753 records, 4 attributes):Comprehensive catalog of bike products, including names, descriptions, and categorization
 2. Bikeshops (93 records, 3 attributes):Detailed registry of bike stores across the continental United States, capturing location and business information
 3. Orderlines (109,514 records, 6 attributes):Comprehensive order history documenting bike purchases by bike shops over the past 5 years


## Methods and Tools
### Data Processing & Analysis
The CRISP-DM methodology was applied, systematically addressing:
- Data Understanding: Conduct exploratory data analysis (EDA) on dataset characteristics and initial insights 
- Data Preparation: 
  . Cleaning: Fields such as date, location, and description were split into new features like bike main categories, sub-categories, and frame materials.
  . ETL Automation: data cleaning is automated within the ETL process using a Python function (collect_data) located in my_pandas_extension/database.py.
- Feature engineering approaches: Auto ARIMA: Aggregated and summarized time-series data for forecasting.
- LSTM: Scaled and split data into training and testing sets using TensorFlow Datasets. Optimized model architecture with Keras Tuner to minimize validation loss. The model generated is composed of
 . 1 LSTM layer containinng 256 units,
 . 3 dense layers of 80, 16 and 11 units respectively. 
 . The model generated 889,428 parameters of which, 296,475 were trainaible and 592,953 were optimzers 


### Machine Learning (if applicable)
- Models tested: 
 . AutoArima Forecaster 
 . TensorFlow's LSTM: 
- Evaluation metrics:
 . Auto-Arima: Mean Absolute Percentage Error (MAPE) 0.2458, Mean Squared Percentage Error (MSPE) 0.09022
 . Long-short Memory (LSTM): Mean Absolute Error: 0.2360, Mean squared error (MSE) 0.0904
- Performance summary: The LSTM model shows a slight performance edge over AutoArima, achieving lower error rates and more conservative results
- Model limitations: 

## Key Findings
- Smoother forecasts: AutoARIMA produced smoother forecasted lines compared to LSTM
- Comparable Performance for Smaller Forecasts: Both methods demonstrated comparable performance when forecasting smaller figures (e.g., in the thousands)
- Accuracy: LSTM acheived lower error rates. 
 refer to viz in Tableu public https://public.tableau.com/app/profile/ella.claude/viz/BikesslaesForecast/Story1



## Deliverables
List of what's included in the repository:
- Notebooks (with descriptions)
- Scripts
- Documentation
- Presentation materials
- Models
- Datasets (if public)

## Installation and Setup
```bash
## Installation Instructions

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)

### Setup
1. Clone the repository:
   ```
   git clone <https://github.com/EllaN12/Bikes_forecast.git>
   cd <Bikes_forecast>
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```



## Project Structure
```
project/
│
├── 00_data_raw/               # Raw Data files
├── 02_reports/          # EDA reports
├── 03_src/               # Source code
├── 04_artifacts/            # Trained models
├── 05_images/              # Figure, tables etcs
├── requirements.txt   # Dependencies
└── README.md
```

## Key Components
- **database.py**: ETL function to automate dta collection and cleaning.
- **summarize_by_time.py**: automated function to summarize data by group and time period
- **Arima_forecast.py**: Contains AutoArima forecarst and evaluation functions and other functions to combine forecasting results,.
- **Multi_variate_forecast.py**: Performs Forecasting using LSTM model.



## Acknowledgments
- Pandas and NumPy for data manipulation
- Tableau Public for data visualization
- AutoArima and LSTM for machine learning and deep learning models
- Scikit-learn for machine learning utilities






