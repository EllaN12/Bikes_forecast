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
from my_pandas_extensions.forecasting import arima_forecast, data_prep


#Auto ARIMA
from sktime.forecasting.arima import AutoARIMA
from tqdm import tqdm

# %%
df = collect_data()

bikeshop_predictions = '04_artifacts/bikeshop_prediction.pkl'
resolved_multi_var_path = os.path.abspath(bikeshop_predictions)


with open(resolved_multi_var_path, 'rb') as f:
    bikeshop_predictions = pickle.load(f)



lstm_df = bikeshop_predictions[bikeshop_predictions['variable'] == 'LSTM_prediction']

lstm_df = lstm_df.dropna(subset=['Sales'])

lstm_df.drop(columns=['ci_lower', 'ci_upper',	'variable' ])

lstm_df = lstm_df.pivot(
    columns = 'bikeshop_name',
    values = 'Sales',
    index = 'order_date'

)\
    .reset_index()\
        .drop(columns = ['order_date'])




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



bikeshop_predictions_df = data_prep(
    data = df,
    group = 'bikeshop_name',
    h = 3,
    LSTM_df = lstm_df
)



bikeshop_predictions_df.to_csv('04_artifacts/bikeshop_prediction.csv')


df = collect_data()

bikeshop_predictions_df.to_pickle('04_artifacts/bikeshop_predictions.pkl')

cat_2_pred = ('04_artifacts/category_2_predictions')
resolved_multi_var_path = os.path.abspath(cat_2_pred)
cat_2_predictions_df = pd.read_csv(resolved_multi_var_path)
cat_2_predictions_df.to_csv('04_artifacts/category_2_predictions.csv')

