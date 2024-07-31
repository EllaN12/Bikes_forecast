# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 7 (Plotnine): Plot Automation ----
 #%%
# Imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots_adjust
#from mizani.breaks import date_breaks
import pandas as pd
import numpy as np
from pandas.core.algorithms import value_counts

from plotnine import *
from plotnine.labels import labs
from plotnine.scales.scale_manual import scale_color_manual
from plotnine.scales.scale_xy import scale_x_datetime, scale_y_continuous
#from mizani.formatters import dollar_format
from plotnine.facets import facet_wrap
from plotnine.geoms import geom_line, geom_point, geom_ribbon, geom_bar
from plotnine.ggplot import ggplot
from plotnine.themes import theme, theme_minimal
from plotnine.themes.themeable import figure_size, legend_position

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.timeseries import summarize_by_time
from my_pandas_extensions.forecasting import arima_forecast 





# Workflow until now
#%%
df = collect_data()


arima_forecast_df = df\
    .summarize_by_time(
        date_column='order_date',
        value_column = "total_price",
        groups = "category_1",
        rule = "M", # aggregation rule
        kind = "period",
        wide_format = True
    )\
    .arima_forecast(
        h = 12,
        sp = 1,
        coverage= 0.95,
        suppress_warnings = True
        
    )
arima_forecast_df
# 1.0 FORECAST VISUALIZATION ----

#%%
#Melting the data = 
# Step 1: Data preparation for Plot
df_prepped = arima_forecast_df\
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

df_prepped

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

  

# 2.0 PLOTTING AUTOMATION ----
# - Make plot_forecast()

#%%
# Function Development 
data = arima_forecast_df
id_column = "category_2" 
date_column = "order_date"
     
def plot_forecast(
    data,
    id_column,
    date_column,
    facet_ncol = 1,
    facet_scales = "free_y",
    date_labels = "%Y",
    date_breaks = "1 Year",
    ribbon_alpha = 0.2,
    wspace = 0.25,
    figure_size = (16, 8),
    title = "Forecast Plot",
    xlab = "Date",
    ylab = "Revenue",
    
):
    arima_forecast_df = data
    required_columns = [id_column, date_column, 'value', 'prediction', "ci_lower", "ci_upper"]
    
    #Data Wrangling
    df_prepped = arima_forecast_df\
        .loc[:, required_columns]\
        .melt(
            value_vars = ["value", "prediction"],
            id_vars = [id_column, date_column, "ci_lower", "ci_upper"],
        # var_name = "variable",
        value_name = "forecast_value")\
        .rename(columns={"forecast_value": "value"})\
        .assign(
            order_date = lambda x: x["order_date"].dt.to_timestamp())
        
        # Handle the Categorical Conversion
    facet_order = df_prepped \
        .groupby(id_column)\
        .agg({'value': np.mean})\
        .reset_index()\
        .assign(id_column = lambda x: pd.Categorical(x[id_column],
                                                 categories = x.sort_values("value", ascending = False)[id_column], ordered = True)).id_column
                                                 
            
        
    # Check for period, convert to datetime64   
    if df_prepped[date_column].dtype != 'datetime64[ns]':
        # TRY CHANGE TO TIMESTAMP
        try:
            df_prepped[date_column] = df_prepped[date_column].dt.to_timestamp()
        except:
            try:
                df_prepped[date_column] = pd.to_datetime(df_prepped[date_column])
            except:
                raise Exception("Could not auto=convert 'date-column' to datetime64[ns]")
    
    # Plotting
    # Geometries
    g = ggplot(
        mapping = aes(
            x= date_column, 
            y = "value", 
            color = "variable"
        ),
        data = df_prepped) \
        + geom_ribbon(
            mapping = aes(ymin = "ci_lower", ymax = "ci_upper"),
            alpha = ribbon_alpha,
            color = None) \
        + geom_line() \
        + facet_wrap(id_column, ncol = facet_ncol, scales = facet_scales) 
        
        #
    g = g \
        + scale_x_datetime(
            date_labels = date_labels,
            date_breaks = date_breaks
        ) \
        + scale_y_continuous(
            labels = dollar_format( big_mark = ",", digits =0 )
        ) \
        + scale_color_manual(values = ["red", "#2C3E50"]) 
        
        # Themes and labels
    g = g \
        + theme_minimal() \
        + theme(
            legend_position = "none",
            subplots_adjust= {"wspace": 0.25},
            figure_size= figure_size
        ) \
        + labs(
            title = title,
            x = xlab,
            y = ylab)
    return g
# Testing 
#%%


arima_forecast_df = df\
    .summarize_by_time(
        date_column='order_date',
        value_column = "total_price",
        groups = "category_2",
        rule = "M", # aggregation rule
        kind = "period",
        wide_format = True
    )\
    .arima_forecast(
        h = 12,
        sp = 1,
        coverage= 0.95,
        suppress_warnings = True
        
    )\
    .assign(
        id_column = lambda x: "revenue")
plot_forecast(
    data = arima_forecast_df,
    id_column = "category_2",
    date_column = 'order_date',
    title = "Revenue Over Time",
    facet_scales= "free_y",
    date_breaks= "2 Year",
    figure_size= (16, 8),
    facet_ncol = 3
)



# %%
#arima_forecast_df
#df
from my_pandas_extensions.forecasting import plot_forecast
#from my_pandas_extensions.forecasting import plot_forecast
# %%
?plot_forecast

arima_forecast_df.plot_forecast(
    id_column = "category_1",
    date_column = "order_date",
    title = "Revenue Over Time",
    facet_scales = "free_y",
    date_breaks = "2 Year",
    figure_size = (16, 8),
    facet_ncol = 3
)
# %%
