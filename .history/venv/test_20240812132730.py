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
   



@app.callback(
    Output("line", "figure"),
    [Input("Overview-option", "n_clicks"),
     Input("Mountain-option", "n_clicks"),
     Input("Road-option", "n_clicks"),
     Input("Bikeshops-option", "n_clicks")]
)

def plot_forecast_line(overview_clicks, mountain_clicks, road_clicks, bikeshops_clicks):
    
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        
        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
        fig = go.Figure()
        
        if clicked_id == "Overview-option":
            df = forecast_data_pull(Total = True)
            title = "Total Forecast"
        if clicked_id == "Mountain-option":
            df = forecast_data_pull(Mountain = True)
            title = "Mountain and Subcategories Forecast"
            
        elif clicked_id == "Road-option":
            df = forecast_data_pull(Road = False)
            title = "Road and Subcategories Forecast"
        
        elif clicked_id == "Bikeshops-option":
            df = forecast_data_pull(Bikeshops = True)
            title = "Bikeshops Forecast"
   
    


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
            
    =

    fig.update_layout(
        title_text=title ,
        plot_bgcolor=PLOT_BACKGROUND,
        
    )

    return fig

plot_forecast_line()    



forecast_data_pull(Mountain = True)




