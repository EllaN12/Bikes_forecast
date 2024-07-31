
#Analysis
import pandas as pd
import numpy as np
import sqlalchemy as sql
import os 
import matplotlib.pyplot as plt


# Importing Data
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.forecasting import arima_forecast
from ydata_profiling import ProfileReport



# create a database and connection
os.mkdir("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/database")

engine = sql.create_engine("sqlite:////Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/database/bike_orders_database.sqlite")

conn = engine.connect()

# Read Excel Files

bikes_df = pd.read_excel("./00_data_raw/bikes.xlsx")
bikeshops_df = pd.read_excel("./00_data_raw/bikeshops.xlsx")
orderlines_df = pd.read_excel("./00_data_raw/orderlines.xlsx")


# Create Tables
bikes_df.to_sql("bikes", con=conn, if_exists="replace")

pd.read_sql("SELECT * FROM bikes", con=conn)

bikeshops_df.to_sql("bikeshops", con=conn, if_exists="replace")
pd.read_sql("SELECT * FROM bikeshops", con=conn)

orderlines_df \
    .iloc[: , 1:] \
    .to_sql("orderlines", con=conn, if_exists="replace")

pd.read_sql("SELECT * FROM orderlines", con = conn)


# Close Connection
conn.close()

# RECONNECTING TO THE DATABASE 

# Connecting is the same as creating
engine = sql.create_engine("sqlite:///./database/bike_orders_database.sqlite")

conn = engine.connect()

# GETTING DATA FROM THE DATABASE

# Get the table names

inspector = sql.inspect(conn)

inspector.get_schema_names()

inspector.get_table_names()

inspector.get_table_names()

# Read the data
table = inspector.get_table_names()
pd.read_sql(f"SELECT * FROM {table[0]}", con=conn)

# Close connection
conn.close()


df = collect_data()

df['city'].unique()

# Pandas Profiling 
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)


# Sampling - Big Datasets

df.profile_report()

df.sample(frac=0.5).profile_report()

df.profile_report(dark_mode = True)

df.profile_report().to_file("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/02_reports/profile_report.html")


# Reshaping the dataframe to get the total price of each category_2 per month
df

rule = "D"

df[['category_2', 'order_date', 'total_price']] \
    .set_index('order_date') \
    .groupby([ 'category_2']) \
    .resample("M", kind = 'timestamp') \
    .agg({'total_price':np.sum}) \
    .unstack("category_2") \
    .reset_index() \
    .assign(order_date = lambda x: x['order_date'].dt.to_period()) \
    .set_index('order_date') 


# Building a function to summarize data by time
#%%
data = df
# %%
def summarize_by_time(
    data, date_column, value_column,
    groups = None,
    rule  = "D",
    agg_func = np.sum,
    kind = "timestamp",
    wide_format = True,
    fillna = 0,
    *args,
    **kwargs):
    
    """_summary_
    A function to smumarize data based on time intervals 
    data = data input in pandas dataframe
    date_column = column with date information
    value_column = column with value information to be aggregated or summarized
    groups = column to be used as groups ( i.e. category, product, etc)
    rule = pandaas resample methodology (i.e. "D" for day, "M" for month, "Y" for year)
    agg_func : function used to aggregate the data (i.e. np.sum, no.mean etc)
    kind: specifies the type of data to be used (i.e. "timestamp" or "period")
    wide_format: if True, will return a wide format dataframe
    fillna: This parameter specifies the value to fill NaN (missing) values with. It defaults to 0.
    *args and kwargs: 

    Returns:
        _type_: _description_
    """
    ##Checks
    
    """for col in value_column:
        if col not in data.columns:
            raise ValueError(f"value_column '{col}' does not exist in the DataFrame.")"""
    
    if (type(data) is not pd.DataFrame):
        raise TypeError("data must be a pandas dataframe.")
    
    if type(value_column) is not list:
        value_column = [value_column]
        
    # Body
    
    #Handle date column
    data = data.set_index(date_column)
    if date_column not in data.columns:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("date_column does not exist in the DataFrame.")
    
    
    # handle groupby
    if groups is not None and groups not in data.columns:
        raise ValueError("groups column does not exist in the DataFrame.")
    if groups is not None:
        data = data.groupby(groups)
    
    # Handle resample 
    data = data.resample(
        rule = rule,
        kind= kind
    )
    
    # Handle aggregation
    function_list = [agg_func] * len ( value_column) #make sure the function list is repeated for each value column
    agg_dict = dict(zip(value_column, function_list))
    data = data \
        .agg(
            func = agg_dict,
            *args,
            **kwargs
        )
#Handle Pivot Wider 
    if wide_format:
         if groups is not None:
             data = data.unstack(groups)  
             if kind == "period":
                 if not isinstance(data.index, pd.PeriodIndex):
                     data.index = data.index.to_period()    
    data = data.fillna(value = fillna)
    
    return data 


summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    groups = 'category_1',
    rule = "D",
    kind = "period",
    agg_func = np.sum,
    wide_format = True,
    #fillna = np.nan
)







#Forecasting using Auto ARIMA
# %%

#Total Revenue
df = collect_data()
bike_total_sales = df\
    .summarize_by_time(
        date_column = "order_date",
        value_column = "total_price",
        rule = "M",
        kind = "period",
        agg_func = np.sum,
        wide_format = True,
        
    )
    
    
    
    
arima_forecast(
    data = bike_total_sales,
    h = 12,
    sp = 12,
    coverage = 0.95,
    suppress_warnings = True
)



# Deep Learning Forecasting

# %%
