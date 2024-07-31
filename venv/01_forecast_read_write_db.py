# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 8 (SQL Database Update): Forecast Write and Read Functions ----
#%%
# IMPORTS ----
import sqlalchemy as sql
from sqlalchemy.sql.schema import MetaData
from sqlalchemy.types import String, Numeric, DateTime

import pandas as pd
import numpy as np

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.timeseries import summarize_by_time
from my_pandas_extensions.forecasting import arima_forecast, plot_forecast

df = collect_data()
# WORKFLOW ----
#%%
# - Until Module 07: Visualization
arima_forecast_df = df\
    .summarize_by_time(
        date_column="order_date",
        value_column="total_price",
        groups = "category_2",
        rule = "M",
        agg_func = np.sum,
        kind = "period",
        wide_format = True,
        fillna = 0)\
    .arima_forecast(
        h = 12,
        sp = 1,
        coverage = 0.95,
        suppress_warnings = True)

arima_forecast_df\
    .plot_forecast(
        id_column = "category_2",
        date_column = "order_date",
        facet_ncol = 3)
    
# DATABASE UPDATE FUNCTIONS ----

#%%
# 1.0 PREPARATION FUNCTIONS ----
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

#%%
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

#%% 
#function testing 
prep_forecast_data_for_update(
    data = arima_forecast_df.drop("category_2", axis = 1),
    id_column = "category_2",
    date_column = "order_date")

#%%
# 2.0 WRITE TO DATABASE ----
def write_forecast_to_database(
    data, id_column, date_column, 
    table_name = "forecast", 
    conn_str = "sqlite:///00_database/bike_orders_database.sqlite",
    if_exists = 'fail',
    **kwargs):
    
    # Prepare the data
    df = prep_forecast_data_for_update(data, id_column, date_column)
    
    #Check format for SQL Database
    df['date'] = df['date'].dt.to_timestamp()
    
    df.info()
    
    SQL_dtype = {
        'id': String(255),
        'date': String(255),
        'value': Numeric(),
        'prediction': Numeric(),
        'ci_lower': Numeric(),
        'ci_upper': Numeric()
    }
    
    # Connect to the database
    engine = sql.create_engine(conn_str)
    
    conn = engine.connect()
    
    # Make Table
    
    # Write to the database
    df.to_sql(
        name =table_name,
        con = engine,
        if_exists = if_exists,
        index = False,
        dtype = SQL_dtype,
        **kwargs
    )
     # Close the connection
    conn.close()

#%%
write_forecast_to_database(
    data = arima_forecast_df,
    id_column = "category_2",
    date_column = "order_date",
    table_name = "forecast",
    if_exists = 'replace'
)


# 3.0 READ FROM DATABASE ----

# %%
def read_forecast_from_database(
    table_name = "forecast",
    conn_str = "sqlite:///00_database/bike_orders_database.sqlite",
    **kwargs):
    
    # Connect to the database
    engine = sql.create_engine(conn_str)
    
    conn = engine.connect()
    
    # Read from the database
    df = pd.read_sql(
        sql = f"SELECT * FROM {table_name}",
        con = engine,
        parse_dates=['date']
    )
    
    # Close the connection
    conn.close()
    
    return df

read_forecast_from_database(table_name="forecast")
    
