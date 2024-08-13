import os
import numpy as np
import pandas as pd
import pickle


from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time

#Forecasting

from my_pandas_extensions.forecasting import arima_forecast

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots




file_path = ''

with open(file_path, 'rb') as f:
    cat_1_df = pickle.load(f)
    
quarter_cat_1_df = cat_1_df\
    .reset_index()\
    .groupby(['category_1','variable'])\
    .resample('Q', on='order_date')\
    .agg({'Sales': 'sum'})\
    .reset_index()

quarter_cat_1_df.tail(10)
 
annual_cat_1_df = cat_1_df\
    .reset_index()\
    .groupby(['category_1','variable'])\
    .resample('y', on='order_date')\
    .agg({'Sales': 'sum'})\
    .reset_index()\
    .melt(
        id_vars=['category_1', 'order_date'],
        value_vars='Sales',
        var_name='variable',
        value_name='.Sales'
    )

  
    
file_path = '/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/04_artifacts/cat_2_prediction.pkl'

with open(file_path, 'rb') as f:
    cat_2_df = pickle.load(f)
    
    
quarter_cat_2_df = cat_2_df\
    .reset_index()\
    .groupby(['category_2','variable'])\
    .resample('Q', on='order_date')\
    .agg({'Sales': 'sum'})\
    .reset_index()
    
quarter_cat_2_df.head(10)
 
annual_cat_2_df = cat_2_df\
    .reset_index()\
    .groupby(['category_2','variable'])\
    .resample('y', on='order_date')\
    .agg({'Sales': 'sum'})\
    .reset_index()
  
    


file_path = '/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/04_artifacts/bikeshop_prediction.pkl'

with open(file_path, 'rb') as f:
    bikeshop_sales_df = pickle.load(f)
    
    
bike_shops = bikeshop_sales_df['bikeshop_name'].unique().tolist()

bikeshop_name = 'Albuquerque Cycles'

bikeshop = bikeshop_sales_df[bikeshop_sales_df['bikeshop_name'] == bikeshop_name]


    
bikeshop_sales_df.info()

q_bikeshops_df = bikeshop_sales_df\
    .reset_index()\
    .groupby(['bikeshop_name', 'variable'])\
    .resample('Q', on='order_date')\
    .agg({'Sales': 'sum'})\
    .reset_index()
    
q_bikeshops_df['quarter'] = q_bikeshops_df['order_date'].dt.quarter



# First, convert 'order_date' to datetime if it's not already
#df['order_date'] = pd.to_datetime(df['order_date'])

# Create a 'quarter' column
bikeshop_sales_df['quarter'] = bikeshop_sales_df['order_date'].dt.to_period('Q')

# Group by 'bikeshop_name' and 'quarter'
grouped = bikeshop_sales_df.groupby(['bikeshop_name', 'quarter'])

bikeshop_sales_df['month_number'] = grouped['order_date'].rank(method = 'dense').astype(int)


q_list = ['2015Q1', '2016Q1']
bikeshop_sales_df['month_number'].unique()
print (bikeshop_sales_df)



fig = fig = make_subplots(rows=1, cols=1)
categories = bikeshop_sales_df['bikeshop_name'].unique()

color_map = {
    ('LSTM_prediction', '2016Q1'): 'orange',
    ('Arima_prediction', '2016Q1'): 'green',
    ('Actuals', '2015Q4'): 'blue',
    ('Actuals', '2015Q1'): 'lightpink',
    ('Actuals', '2015Q2'): 'blue',
    ('Actuals', '2015Q3'): 'blue',
}


for category in categories:
    for variable in bikeshop_sales_df['variable'].unique():
        for quarter in q_list:
            category_data = bikeshop_sales_df[(bikeshop_sales_df['bikeshop_name'] == category) & 
                                              (bikeshop_sales_df['variable'] == variable) & 
                                              (bikeshop_sales_df['quarter'] == quarter)]
            trace_name = f'{variable}_{quarter}'
            color = color_map.get((variable, quarter), 'grey')
            
            fig.add_trace(
                go.Scatter(
                    x=category_data['month_number'],
                    y=category_data['Sales'],
                    mode='lines',
                    name=trace_name,
                    line=dict(color=color),
                    legendgroup=f'{variable}',
                    showlegend=True
                )
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
fig.show()
    
    
    
fig = px.line(
bikeshop_sales_df[bikeshop_sales_df['bikeshop_name'] == category],
x= 'month_number',
y='Sales',

)

annual_bikeshops_df = bikeshop_sales_df\
    .reset_index()\
    .groupby(['bikeshop_name', 'variable'])\
    .resample('y', on='order_date')\
    .agg({'Sales': 'sum'})\
    .reset_index()


categories = quarter_cat_2_df['category_2'].unique()

color_map = {'Actuals': 'grey', 'LSTM_prediction': 'red', 'Arima_prediction': 'green'}


fig = make_subplots(rows=len(categories), cols=1, 
                    subplot_titles=categories,
                    shared_xaxes=True,
                    vertical_spacing=0.05)

#


# Category 2 Sales Forecasting
#Lstm prediction


fig = make_subplots(
    rows=len(categories), cols=1,
    subplot_titles=categories,
    shared_xaxes=True,
    vertical_spacing=0.1
)
for i, category in enumerate(categories, start=1):
    # Iterate over variables within each category
    for variable in quarter_cat_2_df['variable'].unique():
        category_data = quarter_cat_2_df[(quarter_cat_2_df['category_2'] == category) & 
                                         (quarter_cat_2_df['variable'] == variable)]
        fig.add_trace(
            go.Bar(
                x=category_data['order_date'],
                y=category_data['Sales'],
                name=variable,
                marker=dict(color=color_map.get(variable, 'black')),
                legendgroup=f'{variable}_{category}',
                showlegend=True if i == 1 else False
            ),
            row=i,
            col=1
        )

# Update layout of the figure
fig.update_layout(
    height=800,
    width=1000,
    title_text='Sales Forecast by Category',
    xaxis=dict(title='Order Date'),
    yaxis=dict(title='Sales'),
    showlegend=True,  # Show legend for each subplot
    template='plotly_white'  # White background theme
)

# Update legend properties
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

# Show the figure
fig.show()

# Iterate over categories (bikeshops)
for i, category in enumerate(categories, start=1):
    # Filter data for the current category (bikeshop)
    category_data = bikeshop_sales_df[bikeshop_sales_df['bikeshop_name'] == category]
    
    # Iterate over variables (Actuals, LSTM_prediction, Arima_prediction)
    for j, variable in enumerate(bikeshop_sales_df['variable'].unique(), start=1):
        # Filter data for the current variable
        variable_data = category_data[category_data['variable'] == variable]
        
        # Add grouped bar trace for the current variable
        fig.add_trace(
            go.Bar(
                x=variable_data['order_date'],
                y=variable_data['Sales'],
                name=variable,
                marker=dict(color=color_map.get(variable, 'black')),  # Use color map
                legendgroup=f'{variable}_{category}',  # Group for legends
                showlegend=True if i == 1 and j == 1 else False  # Show legend for the first subplot
            ),
            row=i,
            col=1
        )

# Update layout
fig.update_layout(
    height=1600,  # Adjust height for better display of subplots
    width=1000,
    title_text='Sales Forecast by Category (Grouped Bar Charts)',
    xaxis=dict(title='Order Date'),
    yaxis=dict(title='Sales'),
    showlegend=True,  # Show legend for each subplot
    template='plotly_white'  # White background theme
)

# Update legend properties
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

fig.show()



# Create figure for grouped bar charts
fig = go.Figure()

# Iterate over categories (bikeshops)
for category in categories:
    # Filter data for the current category (bikeshop)
    category_data = bikeshop_sales_df[bikeshop_sales_df['bikeshop_name'] == category]
    
    # Create grouped bar trace for each variable
    for variable in category_data['variable'].unique():
        variable_data = category_data[category_data['variable'] == variable]
        fig.add_trace(
            go.Bar(
                x=variable_data['order_date'],
                y=variable_data['Sales'],
                name=variable,
                marker=dict(color=color_map.get(variable, 'black')),  # Use color map
                legendgroup=f'{variable}_{category}',  # Group for legends
                showlegend=True if category == categories[0] else False  # Show legend for the first bikeshop
            )
        )

# Update layout
fig.update_layout(
    height=800,
    width=1000,
    title_text='Sales Forecast by Bikeshop (Grouped Bar Charts)',
    xaxis=dict(title='Order Date'),
    yaxis=dict(title='Sales'),
    barmode='group',  # Grouped bar mode
    showlegend=True,   # Show legend for each subplot
    template='plotly_white'  # White background theme
)

# Update legend properties
fig.update_layout(
    legend=dict(
        x=1.05,   # Adjust legend x position
        y=0.5,    # Adjust legend y position
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

fig.show()




