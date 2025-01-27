
import sqlalchemy as sql
from sqlalchemy import create_engine
from sqlalchemy.types import String, Numeric, DateTime
import pandas_flavor as pf

import pandas as pd
import numpy as np

import pandas_flavor as pf
import os



print("Current working directory:", os.getcwd())
db_path = 'database/bike_orders_database.sqlite'
resolved_db_path = os.path.abspath(db_path)
print("Resolved path to database file:", resolved_db_path)




#COLLECT DATA ---

def collect_data(conn_string = f'sqlite:///{resolved_db_path}'):  
    """
    Collects and combines the bike orders data. 

    Args:
        conn_string (str, optional): A SQLAlchemy connection string to find the database. Defaults to "sqlite:///00_database/bike_orders_database.sqlite".

    Returns:
        DataFrame: A pandas data frame that combines data from tables:
            - orderlines: Transactions data
            - bikes: Products data
            - bikeshops: Customers data
    """

    # Body

    # Cread and create database
    bikes_df = pd.read_excel("./00_data_raw/bikes.xlsx")
    bikeshops_df = pd.read_excel("./00_data_raw/bikeshops.xlsx")
    orderlines_df = pd.read_excel("./00_data_raw/orderlines.xlsx")

    # 1.0 Connect to database

    engine = create_engine(conn_string)

    with engine.connect() as conn:
    # Create Tables and Write DataFrames to SQL
        bikes_df.to_sql("bikes", con=conn, if_exists="replace", index=False)
        bikeshops_df.to_sql("bikeshops", con=conn, if_exists="replace", index=False)
        orderlines_df.iloc[:, 1:].to_sql("orderlines", con=conn, if_exists="replace", index=False)
    
    # Read the tables back into dataframes (optional)
        bikes_data = pd.read_sql("SELECT * FROM bikes", con=conn)
        bikeshops_data = pd.read_sql("SELECT * FROM bikeshops", con=conn)
        orderlines_data = pd.read_sql("SELECT * FROM orderlines", con=conn)

    # Combine tables/data into a dictionary (optional)
        table_names = ['bikes', 'bikeshops', 'orderlines']
        data_dict = {}

        for table in table_names:
        # Drop 'index' if it's part of the dataframe after reading (just in case)
            data_dict[table] = pd.read_sql(f"SELECT * FROM {table}", con=conn).drop(columns="index", errors="ignore")
    
    # At this point, data_dict contains the DataFrames for each table
        #print(data_dict)


    # 2.0 Combining Data

        joined_df = pd.DataFrame(data_dict['orderlines']) \
            .merge(
                right    = data_dict['bikes'],
                how      = 'left',
                left_on  = 'product.id',
                right_on = 'bike.id'
            ) \
            .merge(
                right    = data_dict['bikeshops'],
                how      = "left",
                left_on  = "customer.id",
                right_on = 'bikeshop.id'
            )
        
        # 3.0 Cleaning Data 

        df = joined_df
        

        df['order.date'] = pd.to_datetime(df['order.date'])

        temp_df = df['description'].str.split(" - ", expand = True)
        df['category.1'] = temp_df[0]
        df['category.2'] = temp_df[1]
        df['frame.material'] = temp_df[2]

        temp_df = df['location'].str.split(", ", expand = True)
        df['city'] = temp_df[0]
        df['state'] = temp_df[1]

        df['total.price'] = df['quantity'] * df['price']

        df.columns

        cols_to_keep_list = [
            'order.id', 'order.line', 'order.date',    
            'quantity', 'price', 'total.price', 
            'model', 'category.1', 'category.2', 'frame.material', 
            'bikeshop.name', 'city', 'state'
        ]

        df = df[cols_to_keep_list]

        df.columns = df.columns.str.replace(".", "_")

        df.info()

        return df
# %%


df = collect_data()

print(df)
df.info()
print (df)

@pf.register_dataframe_method
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




df = collect_data()