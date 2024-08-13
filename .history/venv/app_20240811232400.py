import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import callback_context as ctx
from dash import no_update

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import pandas as pd
import pathlib
import pickle

# Data collection
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.Viz import get_top_performers, get_top_stores, get_total, forecast_data_pull

# DataFrame setup
df = collect_data()

if df is not None:
    Mountain = df[df['category_1'] == 'Mountain']['category_2'].unique().tolist()
    #Mountain.append('Mountain')

    Road = df[df['category_1'] == 'Road']['category_2'].unique().tolist()
   #Road.append('Road')
    bikeshop_name = df['bikeshop_name'].unique().tolist()

categories = df['category_1'].unique().tolist()
categories.append('Total')
categories.append('Bikeshops')

#%%
# APP SETUP
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

dropdown2 = dcc.Dropdown(
    id="sub_categories_dropdown",
    options=[{'label': 'Total', 'value': 'Total'}],  # Set "Total" as the default option
                value='Total',  # Set the default value to "Total"
                placeholder="Select a Sub-Category"
            )


TreeMap = dcc.Graph(
    id="TreeMap",
    config={"displayModeBar": False},
    style={"background-color": PLOT_BACKGROUND},
)

Bikeshops_bar = dcc.Graph(
    id="Bikeshops_bar",
    config={"displayModeBar": True},
    style={"background-color": PLOT_BACKGROUND},
)

Q12016_forecast_line = dcc.Graph(
    id="line",
    config={"displayModeBar": False},
    style={"background-color": PLOT_BACKGROUND},
)

first_card =( [
    dbc.CardHeader("Sales"),
    dbc.CardBody( id = "total_price_card", className="card-text"),
        ],)
second_card = ([
    dbc.CardHeader("Quantity"),
    dbc.CardBody( id = "Quantity_card", className="card-text"),
        ],)

cards = dbc.Row(
    [
        dbc.Col(first_card, width=4),
        dbc.Col(second_card, width=4),
    ]
)

app.layout = dbc.Container([
    dbc.Row(dbc.Col(title)),
    dbc.Row([html.Div(
                    id="intro",
                    children="Review of bike sales data for FY2015 and forecast for Q12016 across various bike categories, sub-categories  and stores."
                )]),
    dbc.Row([
        dbc.Col([dropdown], width=4),
        dbc.Col([dropdown2], width=4),]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Sales Distribution"),
                cards,
                html.Br(),
                html.Hr(),
                dcc.Markdown("By Category"),
                TreeMap,
                html.Br(),
                html.Hr(),
                dcc.Markdown("By Bikeshop"),
                Bikeshops_bar
            ])
        ], width=12, lg=5),
        dbc.Col([
            html.H5("Sales Forecast"),
            Q12016_forecast_line,
        ], width=12, lg=7)
    ]),
    
], fluid=True)


@app.callback(
    Output('sub_categories_dropdown', 'options'),
    [Input('overall_dropdown', 'label')]
)
def update_sub_categories(selected_category):
    if selected_category == 'Mountain':
        return [{'label': x, 'value': x} for x in Mountain]
    elif selected_category == 'Road':
        return [{'label': x, 'value': x} for x in Road]
    elif selected_category == 'Bikeshops':
        return [{'label': x, 'value': x} for x in bikeshop_name]
    else:
        return [{'label': 'Total', 'value': 'Total'}]  # Return "Total" by default


@app.callback(
    [Output('total_price_card', 'children'),
     Output('quantity_card', 'children')],
    [Input("Overview-option", "n_clicks"),
     Input("Mountain-option", "n_clicks"),
     Input("Road-option", "n_clicks"),
     Input("Bikeshops-option", "n_clicks"),
     Input("sub-categories_dropdown", "value")]
)



def update_cards(total_clicks, mountain_clicks, road_clicks, bikeshops_clicks, selected_subcategory):
    
    filtered_path = '04_artifacts/sales_quant_df.pkl'
    resolved_filtered_path = os.path.abspath(filtered_path) 
    
    
    file_path =  resolved_filtered_path
    with open(file_path, 'rb') as f:
        sales_quant_df = pickle.load(f)
        df = sales_quant_df.copy()
        
    
    # Determine which button was clicked
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'Total-option':
        
    elif triggered_id == 'Mountain-option':
        filtered_df = df[df['category_1'] == 'Mountain']
    elif triggered_id == 'Road-option':
        filtered_df = df[df['category_1'] == 'Road']
    else:
        filtered_df = df

    # Sum total_price and quantity
    total_price = filtered_df['total_price'].sum()
    quantity = filtered_df['quantity'].sum()

    return f"${total_price:,}", f"{quantity:,} units"



@app.callback(
    Output("TreeMap", "figure"),
    [Input("Overview-option", "n_clicks"),
     Input("Mountain-option", "n_clicks"),
     Input("Road-option", "n_clicks"),
     Input("Bikeshops-option", "n_clicks"),
     Input("sub-categories_dropdown", "value")]
)
def update_tree_map(overview_clicks, mountain_clicks, road_clicks, bikeshops_clicks):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if clicked_id == "Overview-option":
            market_df = df.groupby(['category_1', 'category_2']) \
                .agg(total_price=('total_price', sum)) \
                .reset_index()
            fig = px.treemap(market_df, path=['category_1', 'category_2'], values='total_price', color="category_1", title='Bike Sales Distribution by Category')
            fig.update_traces(textinfo='label+percent parent')
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            fig.update_layout(plot_bgcolor=PLOT_BACKGROUND, paper_bgcolor=PLOT_BACKGROUND, font_color=PLOT_FONT_COLOR)
            return fig
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update


@app.callback(
    Output("Bikeshops_bar", "figure"),
    [Input("Overview-option", "n_clicks"),
     Input("Mountain-option", "n_clicks"),
     Input("Road-option", "n_clicks"),
     Input("Bikeshops-option", "n_clicks"),
     Input("sub-categories_dropdown", "value")]
)
def update_bikeshops_bar_chart(overview_clicks, mountain_clicks, road_clicks, bikeshops_clicks):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if clicked_id == "Overview-option":
            bikestore_df = df \
                .groupby(['bikeshop_name']) \
                .agg(total_price=('total_price', sum)) \
                .sort_values('total_price', ascending=True) \
                .reset_index()

            fig = px.bar(bikestore_df, x='total_price', y='bikeshop_name', orientation='h', title="Sales per Bikeshop",
                         color='total_price',  # Optional: color by total_price for visual distinction
                         color_continuous_scale='viridis')  # Optional: color scale for color gradient
            return fig
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update


@app.callback(
    Output('forecast_dropdown', 'options'),
    [Input("Mountain-option", "n_clicks"),
     Input("Road-option", "n_clicks"),
     Input("Bikeshops-option", "n_clicks")]
)
def set_dropdown_options(mountain_clicks, road_clicks, bikeshops_clicks):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if clicked_id == "Mountain-option":
            options = [{'label': x, 'value': x} for x in Mountain]
        elif clicked_id == "Road-option":
            options = [{'label': x, 'value': x} for x in Road]
        elif clicked_id == "Bikeshops-option":
            options = [{'label': x, 'value': x} for x in bikeshop_name]
        else:
            options = [{'label': 'Total', 'value': 'Total'}]

        return options
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update


@app.callback(
    [Output('line', 'figure'), Output('QBar', 'figure')],
    [Input("Mountain-option", "n_clicks"),
     Input("Road-option", "n_clicks"),
     Input("Bikeshops-option", "n_clicks"),
     Input('forecast_dropdown', 'value')]
)
def make_forecast_graphs(mountain_clicks, road_clicks, bikeshops_clicks, dropdown_value):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
        store_names = bikeshop_name

        if clicked_id == "Mountain-option":
            if dropdown_value in Mountain:
                store_names = df[df['category_2'] == dropdown_value]['bikeshop_name'].unique()
            else:
                store_names = df[df['category_1'] == 'Mountain']['bikeshop_name'].unique()
        elif clicked_id == "Road-option":
            if dropdown_value in Road:
                store_names = df[df['category_2'] == dropdown_value]['bikeshop_name'].unique()
            else:
                store_names = df[df['category_1'] == 'Road']['bikeshop_name'].unique()

        forecast_df = forecast_data_pull(store_names, 2016)
        fig_line = px.line(forecast_df, x='Month', y='Forecast', title='Q12016 Sales Forecast')
        fig_bar = px.bar(forecast_df, x='Month', y='Forecast', title='Q12016 Sales Forecast Bar')

        fig_line.update_layout(plot_bgcolor=PLOT_BACKGROUND, paper_bgcolor=PLOT_BACKGROUND, font_color=PLOT_FONT_COLOR)
        fig_bar.update_layout(plot_bgcolor=PLOT_BACKGROUND, paper_bgcolor=PLOT_BACKGROUND, font_color=PLOT_FONT_COLOR)

        return fig_line, fig_bar
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update


if __name__ == "__main__":
    app.run_server(debug=True, port = 8051)

# %%
