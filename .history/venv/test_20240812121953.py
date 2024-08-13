import numpy as np
import pandas as pd
import pickle

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time
from my_pandas_extensions.Viz import forecast_data_pull

#Forecasting

from my_pandas_extensions.forecasting import arima_forecast

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


df = collect_data()

Mountain_list = df[df['category_1'] == 'Mountain']['category_2'].unique().tolist()

Road_list = df[df['category_1'] == 'Road']['category_2'].unique().tolist()
bikeshop_name = df['bikeshop_name'].unique().tolist()

return df    
def rename_variable_columns(row):
    if row['quarter'] == '2015Q1':
        return 'YoY Comparison for Q1 Forecast'
    elif row['quarter'] == '2015Q4':
        return "Actuals"
    elif row['variable'] == 'Actuals':
        return 'Older data'
    else:
        return row['variable']
    


def plot_forecast_line():
    
    df= forecast_data_pull(Bikeshop = True)
   
    fig = go.Figure()


    color_map = {
    'LSTM_prediction': 'orange',
    'Arima_prediction': 'green'
    }
     
    for variable in df['variable'].unique():
        data_subset = df[df['variable'] == variable]
        color = color_map.get(variable, 'blue')
        fig.add_trace(
            go.Scatter(
                x= data_subset['order_date'],
                y=data_subset['Sales'],
                mode='lines',
                name= variable,
                line=dict(color=color),
                #legendgroup=, 
                #hovertext= trace_name, # Assign legend group here
                showlegend=True
        )
    )

    # Set the first bikeshop traces to be visible
    for trace in fig.data:
        if 'Albuquerque Cycles' in trace.name:
            trace.visible = True


    fig.update_layout(
        title='Sales Forecast Comparison',
        xaxis_title='Order Date',
        yaxis_title='Sales',
        legend_title='Variables',
    )

    return fig

plot_forecast_line()    

d['month_number'].unique()
d = plot_forecast_line()

def rename_variable_columns(row):
    if row['quarter'] == '2015Q1':
        return 'YoY Comparison for Q1 Forecast'
    elif row['quarter'] == '2015Q4':
        return "Actuals"
    elif row['variable'] == 'Actuals':
        return 'Older data'
    else:
        return row['variable']

# Bar chart
def plot_Bar_Chart(df, category):
    
    columns_not_needed = ['ci_lower', 'ci_upper']

    df = df.drop(columns_not_needed, axis=1)
    
    
    # Create a 'quarter' column
    df['quarter'] = df['order_date'].dt.to_period('Q')
    
    # Calculate month number
    grouped = df.groupby(['quarter'])

    df['month_number'] = grouped['order_date'].rank(method = 'dense').astype(int)
  
    
    color_map = {
        ('LSTM_prediction', '2016Q1'): 'orange',
        ('Arima_prediction', '2016Q1'): 'green',
        ('YoY Comparison for Q1 Forecast', '2015Q1'): 'pink',
        ('Actuals', '2015Q4'): 'purple',
        
    }
     
     # rename variable columns for plotting
    df['variable'] = df \
    .apply(rename_variable_columns, axis = 1)

    
    fig = go.Figure()
# Iterate over categories (bikeshops)
    if category is not None:
        for cat in df[category].unique():
            for variable in df['variable'].unique():
                for quarter in df['quarter'].unique():
                    category_data = df[(df[category] == cat) & 
                                                    (df['variable'] == variable) & 
                                                    (df['quarter'] == quarter)]
                    category_data = category_data.dropna()  # Drop NaN values directly


                    trace_name = f'{variable}_{quarter}'
                    color = color_map.get((variable, str(quarter)), 'blue')
                    # Retrieve legend group from legend_map
                    #legend_group = legend_map.get((variable, quarter), 'Others')
            fig.add_trace(
                go.Bar(
                    x=category_data['month_number'],
                    y=category_data['Sales'],
                    name=trace_name,
                    marker_color=color,
                    legendgroup= f'{variable}' ,
                    showlegend=True
                )
        )
            
    else:
        for variable in df['variable'].unique():
                for quarter in df['quarter'].unique():
                    category_data = df[(df['variable'] == variable) & (df['quarter'] == quarter)]
                    trace_name = f'{variable}_{quarter}'
                    color = color_map.get((variable, str(quarter)), 'blue')
                    
                    fig.add_trace(
                                go.Scatter(
                                    x=category_data[quarter],
                                    y=category_data['Sales'],
                                    mode='lines',
                                    name=trace_name,
                                    line=dict(color=color),
                                    legendgroup= f'{variable}',
                                    showlegend=True
                                )
                            )

        fig.update_layout(
            barmode='group',
            title='Sales Forecast',
            xaxis_title='Month',
            yaxis_title='Sales',
            legend=dict(
                x=1.05,  # Adjust legend x position
                y=0.5,   # Adjust legend y position
                traceorder="normal",  # Ensure legend items are ordered normally
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                plot_bgcolor = PLOT_BACKGROUND,
                paper_bgcolor = PLOT_BACKGROUND,
                font_color =  PLOT_FONT_COLOR,
            )
        )
    fig.show() 

plot_Bar_Chart(df, 'bikeshop_name')

# bikeshops

file_path = '/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/04_artifacts/bikeshop_prediction.pkl'

with open(file_path, 'rb') as f:
    bikeshop_sales_df = pickle.load(f)
    


# remove unnecessary columns
columns_not_needed = ['ci_lower', 'ci_upper']

bikeshop_sales_df = bikeshop_sales_df.drop(columns_not_needed, axis=1)

# remove nan values
bikeshop_sales_df = bikeshop_sales_df.dropna()



q_bikeshops_df = bikeshop_sales_df\
    .reset_index()\
    .groupby(['bikeshop_name', 'variable'])\
    .resample('QE', on='order_date')\
    .agg({'Sales': 'sum'})\
    .reset_index()
# Define the color map
color_map = {
    'Q1 YoY': 'pink',
    'Actuals': 'purple',
    'Other Quarters': 'grey',
    'LSTM_prediction': 'orange',
    'Arima_prediction': 'green'
}


# Bar chart with dropdown
def plot_Bar_Chart(df):
    # Create the initial figure
    fig = go.Figure()

    # Create traces for each bikeshop
    for bikeshop in df['bikeshop_name'].unique():
        for variable in df['variable'].unique():
            bikeshop_data = df[(df['bikeshop_name'] == bikeshop) & (df['variable'] == variable)]
            trace_name = f'{bikeshop}_{variable}'
            color = color_map.get(variable, 'blue')

            fig.add_trace(
                go.Bar(
                    x=bikeshop_data['order_date'],
                    y=bikeshop_data['Sales'],
                    name=trace_name,
                    marker=dict(
                    color=color,),
                    #line=dict(width=0.5)
                    visible=False  # Start with all traces hidden
                )
            )

    # Set the first bikeshop traces to be visible
    for trace in fig.data:
        if 'Albuquerque Cycles' in trace.name:
            trace.visible = True

    # Create the dropdown menu
    buttons = []
    for bikeshop in df['bikeshop_name'].unique():
        visible = [trace.name.startswith(bikeshop) for trace in fig.data]
        buttons.append(dict(label=bikeshop,
                            method='update',
                            args=[{'visible': visible},
                                  {'title': f'Sales Data for {bikeshop}'}]))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons)],
        title='Sales Data',
        xaxis_title='Date',
        yaxis_title='Sales',
        height=1200,
        width=1200,
        barmode='group'
    )

    fig.show()


plot_Bar_Chart(q_bikeshops_df)


q_bikeshops_df['variable'].unique()





