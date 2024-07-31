
#data
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time

#Analysis 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Forecasting
from sktime.forecasting.arima import AutoARIMA
from tqdm import tqdm

#vizualization

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd




# Data prep


# Total Revenue
df = collect_data()
total_bikes_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = False,
    #fillna = np.nan
)

total_bikes_sales_df

# Revenue by Category

bikes_sales_by_category_1 = summarize_by_time(
    data = df,
    groups= 'category_1',
    date_column = "order_date",
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = True
    #fillna = np.nan
)

bikes_sales_by_category_1


bikes_sales_by_category_2 = summarize_by_time(
    data = df,
    groups= 'category_2',
    date_column = "order_date",
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = True
)

bikes_sales_by_category_2




#Forecasting
?AutoARIMA

h= 12 # period for forecast
sp = 1 # seasonal period. how many sets of 12 months forecast are we working with 
coverage = 0.95 # confidence interval
suppress_warnings = True

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


Cat_1_arima_pred_df = arima_forecast(
    data = bikes_sales_by_category_1,
    h = 3,
    sp = 1,
    coverage = 0.95,
    suppress_warnings = True
)

arima_pred_df.tail(30)


#Forecasting using LSTM
total_bikes_sales_df
df = collect_data()
total_bikes_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    groups= None,
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = False,
    #fillna = np.nan
)

total_bikes_sales_df

Total_arima_pred_df = arima_forecast(
    data = total_bikes_sales_df,
    h = 3,
    sp = 1,
    coverage = 0.95,
    suppress_warnings = True)


Cat_2_bikes_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    groups= 'category_2',
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = True,
    #fillna = np.nan
)
Cat_2_bikes_sales_df.tail(30)
Cat_2_arima_pred_df = arima_forecast(
    data = Cat_2_bikes_sales_df,
    h = 3,
    sp = 1,
    coverage = 0.95,
    suppress_warnings = True)

Cat_2_arima_pred_df.tail(30)



# Step 1: Data preparation for Plot
df_prepped = Cat_2_arima_pred_df\
    .melt(
        value_vars = ["value", "prediction"],
        id_vars = [
            "category_2", "order_date",
            "ci_lower", "ci_upper"
        ],
        # var_name = "variable",
        value_name = ".value"
    )\
    .rename({".value": "value"}, axis = 1)\
    .assign(
        order_date = lambda x: x["order_date"].dt.to_timestamp()
    )
df_prepped['order_date'].dtype == 'datetime64[ns]'

df_prepped.tail(30)
df_prepped['category_2'].unique()
#%%
# Step 2: Plotting

## geom_ribbon() - for confidence interval .It creates a shaded region between two y-values ( min and max)

ggplot(
    mapping = aes(
        x= "order_date", 
        y = "value", 
        color = "variable"
    ),
    data = df_prepped) \
    + geom_ribbon(
        mapping = aes(ymin = "ci_lower", ymax = "ci_upper"),
        alpha = 0.2,
        color = None) \
    + geom_line() \
    + facet_wrap("category_2", ncol = 3, scales = "free_y") \
    + scale_x_datetime(
        date_labels = "%Y",
        date_breaks = "2 Year "
        
    ) \
    + scale_y_continuous(
        labels = dollar_format( big_mark = ",", digits =0 )
    ) \
    + scale_color_manual(values = ["red", "#2C3E50"]) \
    + theme_minimal() \
    + theme(
        legend_position = "none",
        subplots_adjust= {"wspace":0.25},
        figure_size=(16, 8)
    ) \
    + labs(
        title = "Forecast Plot",
        x = "Date",
        y = "Revenue")



