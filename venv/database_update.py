import sqlalchemy as sql
from sqlalchemy.sql.schema import MetaData
from sqlalchemy.types import String, Numeric, DateTime


import pandas as pd
import numpy as np

#Data Collection
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time
from my_pandas_extensions.forecasting import arima_forecast, LSTM_forecast, data_prep



df = collect_data()

# LSTM Forecasting tables
LSTM_total_price_df = LSTM_forecast(
    group  = None,
    WINDOW_SIZE = 3, 
    BATCH_SIZE  = 9, 
    shuffle_buffer_size = 90, 
    train_rate = .80, 
    dev_rate = .10)


LSTM_category_1_df = LSTM_forecast(
    group = 'category_1',
    WINDOW_SIZE = 3,
    BATCH_SIZE = 9,
    shuffle_buffer_size = 90,
    train_rate = .80,
    dev_rate = .10)

LSTM_category_2_df = LSTM_forecast(
    group = 'category_2',
    WINDOW_SIZE = 3,
    BATCH_SIZE = 9,
    shuffle_buffer_size = 90,
    train_rate = .80,
    dev_rate = .10)


LSTM_bikeshop_df = LSTM_forecast(
    group = 'bikeshop_name',
    WINDOW_SIZE = 3,
    BATCH_SIZE = 9,
    shuffle_buffer_size = 90,
    train_rate = .80,
    dev_rate = .10)


cat_2_prediction_df = data_prep(
    data = df,
    group = 'category_2',
    h = 3,
    LSTM_df = LSTM_category_2_df
)

df.to_pickle('/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/cat_2_prediction_df.pkl')


cat_1_prediction_df = data_prep(
    data = df,
    group = 'category_1',
    h = 3,
    LSTM_df = LSTM_category_1_df
)

df.to_pickle('/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/cat_1_prediction_df.pkl')


bikeshop_prediction_df = data_prep(
    data = df,
    group = 'bikeshop_name',
    h = 3,
    LSTM_df = LSTM_bikeshop_df
)

bikeshop_prediction_df.to_pickle('/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/bikeshop_prediction_df.pkl')



total_sales_prediction_df = data_prep(
    data = df,
    group = None,
    h = 3,
    LSTM_df = LSTM_total_price_df
)
total_sales_prediction_df.dtypes()


cat_1_df  =data_prep(
    data = df,
    group = 'category_1',
    h = 3,
    LSTM_df = LSTM_category_1_df
)

cat_1_df.to_pickle('/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/cat_1_prediction.pkl')


cat_2_df  =data_prep(
    data = df,
    group = 'category_2',
    h = 3,
    LSTM_df = LSTM_category_2_df
)

cat_2_df.to_pickle('/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/cat_2_prediction.pkl')




df1 = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    groups = None,
    rule = "ME",
    kind = "period",
    agg_func = np.sum,
    wide_format = True)
    
df2 = arima_forecast(
data = df1,
h = 3,
sp = 1,
coverage= 0.95,
suppress_warnings = True)

#convert order_date to timestamp
df2['order_date'].dt.to_timestamp()

columns_to_convert = ['value', 'prediction', 'ci_lower', 'ci_upper']

for column in columns_to_convert:
    df2[column] = df2[column].apply(lambda x: int(x) if not pd.isna(x) else x)


cat_df = LSTM_total_price_df

    
cat_df.index = df2['order_date'].iloc[-3:]

    #cat_df.rename(columns = {'total_price': "LSTM_prediction"}, inplace = True)
columns = cat_df.columns


lstm_df = cat_df\
.melt(
#id_vars = 'order_date',
#col_level = 1,
value_vars = columns,
#var_name = 'variable',
value_name = 'LSTM prediction',

ignore_index= False)

merged_df = pd.merge(df2, lstm_df, on = 'order_date', how = 'outer')
        
    # rename columns
merged_df.rename(columns = {'prediction': 'Arima_prediction', 'LSTM prediction': 'LSTM_prediction', 'value': 'Actuals' }, inplace = True)

merged_df

prediction_df = merged_df\
        .melt(
        value_vars = ["Actuals", "Arima_prediction", "LSTM_prediction"],
        id_vars = [
            "order_date",
            "ci_lower", "ci_upper"
        ],
        var_name = "variable",
        value_name = ".value")\
        .assign(
            order_date = lambda x: x["order_date"].dt.to_timestamp())\
        .rename({".value": "Sales"}, axis = 1)



prediction_df.to_pickle('/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/total_prediction.pkl')







arima_forecast_df \
    .rename(
        {'category_2': 'id',
         'order_date': 'date',
    },
    axis = 1,
    )
data = arima_forecast_df
id_column = "category_2"
date_column = "order_date"

def prep_forecast_data_for_update(
    data, id_column, date_column):
    
    #Format the columns names
    df = data\
        .rename(
            {id_column: 'id',
             date_column: 'date',
            },
            axis = 1,
        )
    # Validate correct columns
    required_col_names = ['id', 'date', 'value', 'prediction', 'ci_lower', 'ci_upper']
    if not all(pd.Series(col in df.columns for col in required_col_names)):
        col_text = ', '.join(required_col_names)
        raise Exception(f"Data must contain columns: {col_text}")
    return df


prep_forecast_data_for_update(
    data = arima_forecast_df,
    id_column = "category_2",
    date_column = "order_date")



arima_forecast_df, lstm_df  = data_prep(
    data = df,
    group = 'category_2',
    h = 3,
    LSTM_df = LSTM_category_2_df)


arima_forecast_df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m')
    
    

    
