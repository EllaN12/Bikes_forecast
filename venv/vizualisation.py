
#imports
import numpy as np
import pandas as pd

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time

#Forecasting

from my_pandas_extensions.forecasting import arima_forecast
import pickle

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Total Sales Vizualisation
#collect data
df = collect_data() 

df

#summarize data
total_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = True
)

TT_arima_df = arima_forecast(
    data = total_sales_df,
    h = 3,
    sp = 1,
    coverage= 0.95,
    suppress_warnings = True

)



TT_arima_df['order_date'].dt.to_timestamp()
# List of columns to convert
columns_to_convert = ['value', 'prediction', 'ci_lower', 'ci_upper']

for column in columns_to_convert:
    TT_arima_df[column] = TT_arima_df[column].apply(lambda x: int(x) if not pd.isna(x) else x)

TT_arima_df



# importing Deep learning forecasting
uni_df = pd.read_pickle("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Univariate_predictions.pkl")

uni_df
"""
# Function to calculate the next month's total price
def calculate_next_month(prices):
    return (np.sum(prices[-3:]) * 4) / 12

# Calculate for 9 more months
for i in range(9):
    next_month = calculate_next_month(uni_df['total_price'])
    uni_df.loc[len(uni_df)] = {'total_price': next_month}

uni_df """

uni_df.index = TT_arima_df['order_date'].iloc[-3:] 



uni_df.rename(columns = {'total_price': "deep_learning_prediction"}, inplace = True)


uni_df.reset_index(inplace = True)



merged_df = pd.merge(TT_arima_df, uni_df, on = 'order_date', how = 'outer')

merged_df.rename(columns = {'prediction': 'Arima_prediction', 'deep_learning_prediction': 'LSTM_prediction' }, inplace = True)


?df.melt

df1 = merged_df\
    .melt(
    value_vars = ["value", "Arima_prediction", "LSTM_prediction"],
    id_vars = [
        "order_date",
        "ci_lower", "ci_upper"
    ],
    var_name = "variable",
    value_name = ".value")\
    .assign(
        order_date = lambda x: x["order_date"].dt.to_timestamp())\
    .rename({".value": "Sales"}, axis = 1)




px.line(
    df1,
    x = "order_date",
    y = "Sales",
    color = "variable",
    title = "Total Sales Forecasting"
)



# Sales Vizualisation by Category

df = collect_data() 

Cat_1_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    groups = "category_1",
    rule = "ME",
    kind = "period",
    agg_func = np.sum,
    wide_format = True
)

Cat_1_arima_df = arima_forecast(
    data = Cat_1_sales_df,
    h = 3,
    sp = 1,
    coverage= 0.95,
    suppress_warnings = True

)

Cat_1_arima_df

#convert order_date to timestamp
Cat_1_arima_df['order_date'].dt.to_timestamp()

columns_to_convert = ['value', 'prediction', 'ci_lower', 'ci_upper']

for column in columns_to_convert:
    Cat_1_arima_df[column] = Cat_1_arima_df[column].apply(lambda x: int(x) if not pd.isna(x) else x)


# importing Deep learning forecasting
category_df = pd.read_pickle("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Multivariate_time_series_predictions.pkl")
category_columns = category_df.columns


cat_1_col = [('total_price',           'Mountain'),
            ('total_price',               'Road')]
category_1_df = category_df[cat_1_col]

cat_2_col = [col for col in category_columns if col not in cat_1_col]

category_2_df = category_df[cat_2_col]



# Merging the dataframes

category_1_df.index = Cat_1_arima_df['order_date'].iloc[-3:]

category_1_df.rename(columns = {'total_price': "LSTM_prediction"}, inplace = True)

"""
#Annualize 1 year forcast 
def calculate_next_month(prices):
    return (np.sum(prices[-3:]) * 4) / 12

# Calculate for 9 more months for each category
categories = category_1_df.columns.to_list()

for i in range(9):
    for category in categories[1]:
        cat_data = category_1_df[category_1_df['category'] == category]
        if len(cat_data) >= 3:
            next_month = calculate_next_month(cat_data['LSTMprediction'].tail(3))
            last_date = cat_data['order_date'].max()
            new_date = last_date + relativedelta(months=1)
            new_row = pd.DataFrame({
                'date': [new_date],
                'category': [category],
                'LSTMprediction': [next_month]
            })


"""

cat_1_df = category_1_df\
    .melt(
    #id_vars = 'order_date',
    col_level = 1,
    value_vars = ['Mountain', 'Road'],
    var_name = 'category_1',
    value_name = 'LSTMprediction',
  
    ignore_index= False)







merged_df = pd.merge(Cat_1_arima_df, cat_1_df, on = ['order_date', 'category_1'], how = 'outer')

merged_df.rename(columns = {'prediction': 'Arima_prediction', 'LSTMprediction': 'LSTM_prediction', 'value': 'Actuals' }, inplace = True)


?pd.merge

df2 = merged_df\
    .melt(
    value_vars = ["Actuals", "Arima_prediction", "LSTM_prediction"],
    id_vars = [
        "order_date",
        "ci_lower", "ci_upper", "category_1"	
    ],
    var_name = "variable",
    value_name = ".value")\
    .assign(
        order_date = lambda x: x["order_date"].dt.to_timestamp())\
    .rename({".value": "Sales"}, axis = 1)



df2.tail(10)

categories = df2['category_1'].unique()

color_map = {'Actuals': 'blue', 'LSTM_prediction': 'red', 'Arima_prediction': 'green'}


fig = make_subplots(rows=len(categories), cols=1, 
                    subplot_titles=categories,
                    shared_xaxes=True,
                    vertical_spacing=0.1)

#
for i , category in enumerate(categories, start = 1):
    for variable in df2['variable'].unique():
        category_data = df2[(df2['category_1'] == category) & (df2['variable'] == variable)]
        fig.add_trace(
            go.Scatter(
                x = category_data['order_date'],
                y = category_data['Sales'],
                mode = 'lines',
                name= f'{variable}',
                line= dict(color=color_map.get(variable, 'black')),
                legendgroup = f'{variable}_{category}',
                showlegend = True if i == 1 else False
            ),
            row = i,
            col = 1
        )
        fig.update_layout(
            legend=dict(
                x=1.05,  # Adjust legend x position
                y=0.5,   # Adjust legend y position
                traceorder="normal",  # Ensure legend items are ordered normally
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2
            )
        )

fig.update_layout(
    height=800,
    width=1000,
    title_text='Sales Forecast by Category',
    xaxis=dict(title='Order Date'),
    yaxis=dict(title='Sales'),
    showlegend=True,  # Show legend for each subplot
    template='plotly_white'  # White background theme
)

fig.show()



# Category 2 Sales Forecasting
#Lstm prediction
category_2_df

#Arima prediction
df = collect_data() 

Cat_2_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    groups = "category_2",
    rule = "ME",
    kind = "period",
    agg_func = np.sum,
    wide_format = True
)

Cat_2_arima_df = arima_forecast(
    data = Cat_2_sales_df,
    h = 3,
    sp = 1,
    coverage= 0.95,
    suppress_warnings = True
)

Cat_2_arima_df


#convert order_date to timestamp
Cat_2_arima_df['order_date'].dt.to_timestamp()

columns_to_convert = ['value', 'prediction', 'ci_lower', 'ci_upper']

for column in columns_to_convert:
    Cat_2_arima_df[column] = Cat_2_arima_df[column].apply(lambda x: int(x) if not pd.isna(x) else x)



# Merging the dataframes

category_2_df.index = Cat_2_arima_df['order_date'].iloc[-3:]

category_2_df.rename(columns = {'total_price': "LSTM_prediction"}, inplace = True)

columns = category_2_df.columns.to_list()


cat_2_df = category_2_df\
    .melt(
    #id_vars = 'order_date',
    col_level = 1,
    value_vars = [col[1] for col in columns],
    var_name = 'category_2',
    value_name = 'LSTM prediction',
  
    ignore_index= False)



merged_df = pd.merge(Cat_2_arima_df, cat_2_df, on = ['order_date', 'category_2'], how = 'outer')

merged_df.rename(columns = {'prediction': 'Arima_prediction', 'LSTM prediction': 'LSTM_prediction', 'value': 'Actuals' }, inplace = True)


?pd.merge

df2 = merged_df\
    .melt(
    value_vars = ["Actuals", "Arima_prediction", "LSTM_prediction"],
    id_vars = [
        "order_date",
        "ci_lower", "ci_upper", "category_2"	
    ],
    var_name = "variable",
    value_name = ".value")\
    .assign(
        order_date = lambda x: x["order_date"].dt.to_timestamp())\
    .rename({".value": "Sales"}, axis = 1)



df2.tail(10)

categories = df2['category_2'].unique()

color_map = {'Actuals': 'blue', 'LSTM_prediction': 'red', 'Arima_prediction': 'green'}


# Create subplots dynamically based on number of categorical columns
num_cols = len(categories)
num_rows = (num_cols -1) // 4 + 1  # Adjust rows based on columns

fig = make_subplots(rows= num_rows, cols=3, 
                    subplot_titles=categories,
                    shared_yaxes=True,
                    vertical_spacing=0.1)


# Calculate subplot indices (1-based)
row = (i // 4) + 1
col_num = (i % 4) + 1
#
for i , category in enumerate(categories, start = 1):
    for variable in df2['variable'].unique():
        category_data = df2[(df2['category_2'] == category) & (df2['variable'] == variable)]
        fig.add_trace(
            go.Scatter(
                x = category_data['order_date'],
                y = category_data['Sales'],
                mode = 'lines',
                name= f'{variable}',
                line= dict(color=color_map.get(variable, 'black')),
                legendgroup = f'{variable}_{category}',
                showlegend = True if i == 1 else False
            ),
            row = row,
            col = col_num
        )
        fig.update_layout(
            legend=dict(
                x=1.05,  # Adjust legend x position
                y=0.5,   # Adjust legend y position
                traceorder="normal",  # Ensure legend items are ordered normally
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2
            )
        )

fig.update_layout(
    height=1000,
    width=1000,
    title_text='Sales Forecast by Category',
    xaxis=dict(title='Order Date'),
    yaxis=dict(title='Sales'),
    showlegend=True,  # Show legend for each subplot
    template='plotly_white'  # White background theme
)

fig.show()




def data_prep(data, group, h):
    
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
    
    # importing Deep learning forecasting
    category_df = pd.read_pickle("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Multivariate_time_series_predictions.pkl")
    category_columns = category_df.columns
    
    cat_1_col = [('total_price',           'Mountain'),
                ('total_price',               'Road')]
    category_1_df = category_df[cat_1_col]

    cat_2_col = [col for col in category_columns if col not in cat_1_col]
    category_2_df = category_df[cat_2_col]
    
    bikeshop_df = pd.read_pickle("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/bikeshops_predictions.pkl")
    bike_cols = bikeshop_df.columns.to_list()
    

    if group == "category_1":
        cat_df = category_1_df
    elif group == "category_2":
        cat_df = category_2_df
    else:
        cat_df = bikeshop_df
        
    columns = cat_df.columns.to_list()
    
    cat_df.index = df2['order_date'].iloc[-3:]

    #cat_df.rename(columns = {'total_price': "LSTM_prediction"}, inplace = True)


    lstm_df = cat_df\
        .melt(
        #id_vars = 'order_date',
        col_level = 1,
        value_vars = [col[1] for col in columns],
        var_name = group,
        value_name = 'LSTM prediction',
    
        ignore_index= False)

    
    
    return df2, lstm_df

                                        

df2, lstm_df = data_prep(
    data = df,
    group = "bikeshop_name",
    h = 3
)

df2 
lstm_df



# Constants for styling
PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'
COMMON_STYLE ={
    "font-size": "1.25rem",
    "color": " white",
    "text-align": "left",
    "margin-top": "1rem",
    "margin-bottom": "1rem",
    "font-weight": "none",
    "font-family": "Times New Roman",
    "background-color": "transparent" 
}

def plot_prediction(df2, lstm_df, group):
    
    merged_df = pd.merge(df2, lstm_df, on = ['order_date', group], how = 'outer')

    merged_df.rename(columns = {'prediction': 'Arima_prediction', 'LSTM prediction': 'LSTM_prediction', 'value': 'Actuals' }, inplace = True)


    df3 = merged_df\
        .melt(
        value_vars = ["Actuals", "Arima_prediction", "LSTM_prediction"],
        id_vars = [
            "order_date",
            "ci_lower", "ci_upper", group	
        ],
        var_name = "variable",
        value_name = ".value")\
        .assign(
            order_date = lambda x: x["order_date"].dt.to_timestamp())\
        .rename({".value": "Sales"}, axis = 1)

 
    # Initialize categories and color mapping           
    categories = df3[group].unique()
    color_map = {'Actuals': 'blue', 'LSTM_prediction': 'red', 'Arima_prediction': 'green'}


    # Create subplots dynamically based on number of categorical columns
    num_cols = 1
    num_rows = (len(categories)) # Adjust rows based on columns

    fig = make_subplots(rows= num_rows, cols= num_cols, 
                        subplot_titles=categories,
                        shared_xaxes=True,
                        vertical_spacing=0.1)


    
    # Add traces for each category and variable
    for i , category in enumerate(categories, start = 1):
        for variable in df3['variable'].unique():
            category_data = df3[(df3[group] == category) & (df3['variable'] == variable)]
            
            # Calculate subplot indices (1-based)
            row = (i -1) // num_cols + 1
            col_num = (i -1) % num_cols + 1
            
            fig.add_trace(
                go.Bar(
                    x = category_data['order_date'],
                    y = category_data['Sales'],
                    name= f'{variable}',
                    marker= dict(color=color_map.get(variable, 'black')),
                    legendgroup = f'{variable}_{category}',
                    showlegend = True if i == 1 else False
                ),
                row = row,
                col = col_num
                order_by = 'sales'
            )
            
    fig.update_layout(
        legend=dict(
            x=1.05,  # Adjust legend x position
            y=0.5,
            font=dict(
                family="sans-serif",
                size=12,
                color= PLOT_FONT_COLOR  
            ),
            bgcolor = PLOT_BACKGROUND),
            height=1200,
            width=1200,
            title_text='Sales Forecast by Category',
            xaxis=dict(title='Order Date'),
            yaxis=dict(title='Sales'),
            )
            
    return fig# Adjust legend y position


plot_prediction(df2, lstm_df, "bikeshop_name")



cat_2_df2, cat_2_lstm_df = data_prep(
    data = df,
    group = "category_2",
    h = 3
)


plot_prediction(cat_2_df2, cat_2_lstm_df, "category_2")

cat_2_df2['Cross Country Race']



df = collect_data()

Cat_2_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    groups = "category_2",
    rule = "ME",
    kind = "period",
    agg_func = np.sum,
    wide_format = True
)

Cat_2_arima_df = arima_forecast(
    data = Cat_2_sales_df,
    h = 3,
    sp = 1,
    coverage= 0.95,
    suppress_warnings = True
)

df = Cat_2_arima_df[Cat_2_arima_df['category_2'] == 'Cross Country Race']

df= collect_data()




    
df1 = summarize_by_time(
    data = df,
    date_column = 'order_date',
    value_column = 'total_price',
    groups = 'category_2',
    rule = 'ME',
    kind = 'period',
    agg_func = np.sum,
    wide_format = True
)

def summarize_by_time2(df, date_column, value_column, groups, rule='ME', kind='period', agg_func=np.sum, wide_format=True):
    """
    Summarizes DataFrame by time periods based on specified parameters.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - date_column (str): Name of the column containing date/time information.
    - value_column (str): Name of the column containing the values to aggregate.
    - groups (list): List of column names to group by.
    - rule (str): Resampling rule ('ME' for month end, 'MS' for month start, etc.).
    - kind (str): Type of period to create ('period' or 'timestamp').
    - agg_func (function): Aggregation function to apply to grouped data.
    - wide_format (bool): If True, returns DataFrame in wide format; otherwise, returns in long format.

    Returns:
    - DataFrame: Summarized DataFrame.
    """
    # Convert date_column to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column])

    # Group by specified columns
    grouped = df.groupby(groups)

    # Resample and apply aggregation function
    summarized = grouped.resample(rule=rule, on=date_column)[value_column].agg(agg_func)

    # Reset index if wide_format is True
    if wide_format:
        summarized = summarized.unstack(level=groups)

    return summarized


summarized_df = summarize_by_time2(
    df=df,
    date_column='order_date',
    value_column='total_price',
    groups='category_2',
    rule='q',  # 'M' for month end, you can change to 'MS' for month start if needed
    kind='period',
    agg_func=np.sum,
    wide_format=True  # Returns wide format with category_2 and bikeshop_name as columns
)


summarized_df


df = collect_data()