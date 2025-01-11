#%%
import os
import pandas as pd
import numpy as np
import pandas_flavor as pf
import pickle

#Data Collection
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time
import matplotlib.pyplot as plt



#Auto ARIMA
from sktime.forecasting.arima import AutoARIMA
from tqdm import tqdm


#deep learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.optimizers import Nadam
import keras_tuner as kt

@pf.register_dataframe_method
def arima_forecast(data,h,sp, coverage, suppress_warnings, *args, **kwargs):
    
    #Checks
    
    #Handle input 
    df = data
    """_summary_
    For each column in the original dataset:
The time series data is extracted.
An AutoARIMA model is instantiated and fitted to the data.
Predictions and prediction intervals are generated.
Predictions, prediction intervals, and original values are combined into a DataFrame.
The DataFrame is added to model_results_dict.
Finally, the results from each column are concatenated into a single DataFrame.

    Returns:
        _type_: _description_
        
        
  """""" Generates ARIMA forecasts for one or more time series.
    Args:
        data (Pandas Data Frame): 
            Data must be in wide format. 
            Data must have a time-series index 
            that is a pandas period.
        h (int): 
            The forecast horizon
        sp (int): 
            The seasonal period
        alpha (float, optional): 
            Contols the confidence interval. 
            alpha = 1 - 95% (CI).
            Defaults to 0.05.
        suppress_warnings (bool, optional): 
            Suppresses ARIMA feedback during automated model training. 
            Defaults to True.
        args: Passed to sktime.forecasting.arima.AutoARIMA
        kwargs: Passed to sktime.forecasting.arima.AutoARIMA
    Returns:
        Pandas Data Frame:
            - A single time series contains columns: value, prediction, ci_lo, and ci_hi
            - Multiple time series will be returned stacked by group
    """
    """ """
    # For Loop
    model_results_dict = {}
    for col in tqdm(df.columns, mininterval=0):
        #series extractions
        y = df[col]
        
        #print(y[0:5]) # checkpoint 1
        
        #Modeling 
        forecaster = AutoARIMA(
            sp = sp, 
            suppress_warnings = suppress_warnings,
            *args,
            **kwargs)
        
        forecaster.fit(y)
        #print(forecaster) code check
     
        # Predictions and Intervals 
        y_pred_ints = forecaster.predict_interval(
            fh = np.arange(1, h+1),
            coverage = coverage)
        
        # Predictions 
        y_pred = forecaster.predict()
        
        #Code check 
        #print (y_pred_ints)
        #print (y_pred)
        
        # combine into dataframe
        ret = pd.concat([y, y_pred, y_pred_ints], axis = 1)
        ret.columns = ["value", "prediction", "ci_lower", "ci_upper"]
       
        # Update dictionary
        model_results_dict[col] = ret
        
        #code check
         #code check
        #print(ret)
        #Stack Each Dict Element on Top of Each Other
    model_results_df = pd.concat(
        model_results_dict,
        axis = 0
        )
    #return model_results_df

#print (arima_forecast(data, h, sp, coverage, suppress_warnings))

        # Handle Names
    nms = [*df.columns.names, *df.index.names]
    model_results_df.index.names = nms
        
        # reset index
    ret = model_results_df.reset_index()
        
        # Drop Columns Containing "level"
    cols_to_keep = ~ret.columns.str.contains("level")
    ret = ret.loc[:, cols_to_keep]
    return ret


df = collect_data()
@pf.register_series_method
def data_prep(data, group, h, LSTM_df):
    

    df1 = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    groups = group,
    rule = "ME",
    kind = "period",
    agg_func = np.sum,
    wide_format = True)
    
    df2 = arima_forecast(
    data = df1,
    h = h,
    sp = 1,
    coverage= 0.95,
    suppress_warnings = True)
    
    #convert order_date to timestamp
    df2['order_date'].dt.to_timestamp()


    columns_to_convert = ['value', 'prediction', 'ci_lower', 'ci_upper']

    for column in columns_to_convert:
        df2[column] = df2[column].apply(lambda x: int(x) if not pd.isna(x) else x)

    # Merging the dataframes

    # importing Deep learning forecasting files:

    
    if group == "category_1" or group == "category_2":
        cat_1_col = [('total_price',           'Mountain'),
                ('total_price',               'Road')]
        cat_2_col = [col for col in LSTM_df.columns if col not in cat_1_col]
        
        if group == "category_1":
            cat_df = LSTM_df[cat_1_col]
        elif group == "category_2":
            cat_df = LSTM_df[cat_2_col]
        
    elif group == "bikeshop_name":
        cat_df = LSTM_df
        
    else:
        cat_df = LSTM_df
        

    cat_df.index = df2['order_date'].iloc[-3:]


    #cat_df.rename(columns = {'total_price': "LSTM_prediction"}, inplace = True)
    if group == "category_1":
        cat_1_col = [('total_price', 'Mountain'), ('total_price', 'Road')]
        cat_df = LSTM_df[cat_1_col]
    elif group == "category_2":
        cat_2_col = [col for col in LSTM_df.columns if col not in [('total_price', 'Mountain'), ('total_price', 'Road')]]
        cat_df = LSTM_df[cat_2_col]
    elif group == "bikeshop_name":
        cat_df = LSTM_df
    else:
        cat_df = LSTM_df

    # Set index for cat_df to match the last 3 dates in df2
    cat_df.index = df2['order_date'].iloc[-3:]

    # Prepare for melting
    if isinstance(cat_df.columns, pd.MultiIndex):
        # If multi-level columns, extract the second level for melting
        value_vars = [col[1] for col in cat_df.columns]
    else:
        # If single-level columns, use all column names
        value_vars = list(cat_df.columns)

    # Melt the DataFrame 
    lstm_df = cat_df.reset_index().melt(
        id_vars='order_date',
        value_vars=value_vars,
        var_name=group,
        value_name='LSTM prediction'
    )

    if group is not None:
        merged_df = pd.merge(df2, lstm_df, on = ['order_date', group], how = 'outer')
    else:
        merged_df = pd.merge(df2, lstm_df, on = 'order_date', how = 'outer')
        
    # rename columns
    merged_df.rename(columns = {'prediction': 'Arima_prediction', 'LSTM prediction': 'LSTM_prediction', 'value': 'Actuals' }, inplace = True)


    if group is not None:
            id_vars = [
            "order_date",
            "ci_lower", "ci_upper", group]
    else:
        id_vars = [
            "order_date",
            "ci_lower", "ci_upper"	]
            
            
    
    prediction_df = merged_df\
        .melt(
        value_vars = ["Actuals", "Arima_prediction", "LSTM_prediction"],
        id_vars = id_vars  ,
        var_name = "variable",
        value_name = ".value")\
        .assign(
            order_date = lambda x: x["order_date"].dt.to_timestamp())\
        .rename({".value": "Sales"}, axis = 1)

    
    return prediction_df



# %%




