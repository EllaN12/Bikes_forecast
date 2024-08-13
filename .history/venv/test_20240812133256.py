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
   


external_stylesheets = [dbc.themes.CYBORG]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Constants for styling
PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'
COMMON_STYLE = {
    "font-size": "1.25rem",
    "color": " white",
    "text-align": "left",
    "margin-top": "1rem",
    "margin-bottom": "1rem",
    "font-weight": "none",
    "font-family": "Times New Roman",
    "background-color": "transparent"
}

# PATHS
BASE_PATH = pathlib.Path(__file__).parent.resolve()
ART_PATH = BASE_PATH.joinpath("artifacts").resolve()

title = html.H2("Bike Sales and Forecasting Dashboard", className="text-center bg-transparent text-dark-yellow p-2")


# Layout items
dropdown = dbc.DropdownMenu(
    id="overall_dropdown",
    label="Total",
    children=[
        dbc.DropdownMenuItem("Total", id="Total-option", n_clicks=0),
        dbc.DropdownMenuItem("Mountain", id="Mountain-option", n_clicks=0),
        dbc.DropdownMenuItem("Road", id="Road-option", n_clicks=0),
        dbc.DropdownMenuItem("Bikeshops", id="Bikeshops-option", n_clicks=0)
    ],
)

Q12016_forecast_line = dcc.Graph(
    id="line",
    config={"displayModeBar": False},
    style={"background-color": PLOT_BACKGROUND},
)

app.layout = dbc.Container([
    dbc.Row(dbc.Col(title)),
    dbc.Row([html.Div(
                    id="intro",
                    children="Review of bike sales data for FY2015 and forecast for Q12016 across various bike categories, sub-categories  and stores."
                )]),
    dbc.Row([
        dbc.Col([
            dropdown,
            html.H5("Sales Forecast"),           
            Q12016_forecast_line,
        ], width=12, lg=7)
    ]),
    
], fluid=True)

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
    
        fig.update_layout(
            title_text=title ,
            plot_bgcolor=PLOT_BACKGROUND,
            paper_bgcolor=PLOT_BACKGROUND,
            font_color=PLOT_FONT_COLORm
            
        )
        return fig
    
    





