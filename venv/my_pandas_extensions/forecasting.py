import pandas as pd
import numpy as np
import pandas_flavor as pf

#Data Collection
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time


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



@pf.register_dataframe_method
def LSTM_forecast(group, WINDOW_SIZE, BATCH_SIZE, shuffle_buffer_size, train_rate, dev_rate):
    
    """_summary_
    
        Returns:
            _type_: _description_
        """
    df = collect_data()
    
    if group == 'category_1' or group == 'category_2':
        cat_1 = summarize_by_time(
        data = df,
        date_column = "order_date",
        groups= 'category_1',
        value_column = "total_price",
        rule = "M",
        kind = "period",
        agg_func = np.sum,
        wide_format = True)
        
        cat_2 = summarize_by_time(
        data = df,
        date_column = "order_date",
        groups= 'category_2',
        value_column = "total_price",
        rule = "M",
        kind = "period",
        agg_func = np.sum,
        wide_format = True)
        
        df1 = cat_1.merge(cat_2, on = 'order_date', how = 'outer')
    elif group == 'bikeshop_name' or group is None:
        df1 = summarize_by_time(
        data = df,
        date_column = "order_date",
        groups= group,
        value_column = "total_price",
        rule = "M",
        kind = "period",
        agg_func = np.sum,
        wide_format = True)
    
    
    # Define parameters
    n_features = len(df1.columns) # number of features/category of bikes
    
   
    #scale data
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df1.values)
    
    #create a dataframe with the scaled values
    df2 = pd.DataFrame(scaled_values, index= df1.index, columns=df1.columns)
    
    
    # split the data into train and val and test

    train_size = int(len(df2) * train_rate)
    dev_size = int(len(df2) * dev_rate)

    train_df = df2.iloc[:train_size]
    dev_df = df2.iloc[train_size:train_size + dev_size]
    test_df = df2.iloc[train_size + dev_size:]
    
    
   
    # function to create X and y
    def create_X_y(df, WINDOW_SIZE):
        X = []
        y = []
        for i in range(len(df) - WINDOW_SIZE):
            X.append(df.values[i:i+WINDOW_SIZE])
            y.append(df.values[i+WINDOW_SIZE])
        return np.array(X), np.array(y)

    # create X and y for train, val and test datasets

    X_train, y_train = create_X_y(train_df, WINDOW_SIZE)
    X_val, y_val = create_X_y(dev_df, WINDOW_SIZE)
    X_test, y_test = create_X_y(test_df, WINDOW_SIZE)
    

    # Creating tensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
    dev_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    
    
    # Import the model
    if group == 'category_1' or group == 'category_2':
        model_path = ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Multivariate_time_series_model.h5")
    elif group == 'bikeshop_name':
        model_path = ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/bikeshop_time_series_model.h5")
    else:
        model_path = ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/univariate_time_series_model.h5")        
      
    model = tf.keras.models.load_model(model_path)
    
    
    # make predictions
    X_test = np.concatenate([x for x, y in test_dataset], axis=0)
    y_pred = model.predict(X_test)
        
    # create a dataframe with the predictions
    pred_inverse = scaler.inverse_transform(y_pred)
    LSTM_prediction_df = pd.DataFrame(
    pred_inverse,
    #index=test_df.index[-len(pred_inverse):],
    columns = df1.columns)

    return LSTM_prediction_df



@pf.register_series_method
def data_prep(data, group, h, LSTM_df):
    
    df = collect_data() 
    
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
    
    # importing Deep learning forecasting df
    #category_df = LSTM_df 
    
    #category_columns = category_df.columns
    
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
    columns = cat_df.columns

    if group == "category_1" or group == "category_2" or group == "bikeshop_name":
        lstm_df = cat_df\
        .melt(
        #id_vars = 'order_date',
        col_level = 1,
        value_vars = [col[1] for col in columns],
        var_name = group,
        value_name = 'LSTM prediction',
    
        ignore_index= False)
    else: 
        lstm_df = cat_df\
        .melt(
        #id_vars = 'order_date',
        #col_level = 0,
        value_vars = [col for col in columns],
        #var_name = 'total_price',
        value_name = 'LSTM prediction',
    
        ignore_index= False)

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


