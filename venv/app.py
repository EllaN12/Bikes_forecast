import dash
from dash import dcc
from dash import html
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
from plotly import graph_objects as go

import pandas as pd
import pathlib
import pickle


# data collection
from my_pandas_extensions.database import collect_data

from my_pandas_extensions.Viz import get_top_performers, get_top_stores, get_total, forecast_data_pull




#%%


df = collect_data()

if df is not None:
    Mountain = df[df['category_1'] == 'Mountain']['category_2'].unique().tolist()
    Mountain.append('Mountain')

    Road = df[df['category_1'] == 'Road']['category_2'].unique().tolist()
    Road.append('Road')
    bikeshop_name = df['bikeshop_name'].unique().tolist()


#APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
# Initialize the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)

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


# PATHS
BASE_PATH = pathlib.Path(__file__).parent.resolve()
ART_PATH = BASE_PATH.joinpath("artifacts").resolve()

title  = html.H2("Bike Sales and Forecasting Dashboard", className = "text-center bg-transparent text-dark-yellow p-2"),




# Layout items 

#sales_card = dbc.CardHeader("SalgX

dropdown = dbc.DropdownMenu(
    id="overall_dropdown",
    label="Select View",
    children=[
        dbc.DropdownMenuItem("Overview", id="Overview-option"),
        dbc.DropdownMenuItem("Mountain", id="Mountain-option"),
        dbc.DropdownMenuItem("Road", id="Road-option"),
        dbc.DropdownMenuItem("Bikeshops", id="Bikeshops-option"),
    ],
)


TreeMap = dcc.Graph(
    id = "TreeMap",
    config = {"displayModeBar": False},
    style = {"background-color": PLOT_BACKGROUND},
)

Bikeshops_bar = dcc.Graph(
    id = "Bikeshops_bar",
    config = {"displayModeBar": True},
    style = {"background-color": PLOT_BACKGROUND},
)
performers = dcc.Graph(
    id = "performers_bar",
    config = {"displayModeBar": True},
    style = {"background-color": PLOT_BACKGROUND},
)


Q12016_forecast_line = dcc.Graph(
    id = "line",
    config = {"displayModeBar": False},
    style = {"background-color": PLOT_BACKGROUND},
)

Q12016_forecast_bar = dcc.Graph(
    id = "QBar",
    config = {"displayModeBar": False},
    style = {"background-color": PLOT_BACKGROUND},
)


forecast_dropdown = dcc.Dropdown(
    id="forecast_dropdown",
    options=[],
    value='Total',
    style={'color': 'black'}
)

app.layout= dbc.Container([
        dbc.Row(dbc.Col(title)),
        dbc.Row([
            dbc.Col([
                    html.Div([
                            html.Div(
                                id="intro",
                                children="Review of bike sales data for FY2015 and forecast for Q12016 across various bike categories and stores."
                            ),   
                html.Br(),
                html.Hr(),
                dropdown,
                html.Br(),
                html.H5("Sales Distribution"),
                dcc.Markdown("By Category"),
                TreeMap,
                html.Br(),
                dcc.Markdown("By Bikeshop"),
                Bikeshops_bar
                ]) 
                ], width = 12, lg = 5),
            dbc.Col(
                [html.H5("FY15 Performance"),
                performers,
                html.Br(),
                html.Hr(),
                html.H5("Sales Forecast"),
                forecast_dropdown,
                Q12016_forecast_line,
                Q12016_forecast_bar],
                
                width = 12, lg = 7)
            ])
            ], fluid = True,)
       

@app.callback(
        Output("TreeMap", "figure"),
        Input ("overall_dropdown", "value")  
)

def update_tree_map(selected_dropdown_value):
    try:
        if selected_dropdown_value == "Overview":
            market_df = df.groupby(['category_1', 'category_2']) \
                .agg(total_price=('total_price', sum)) \
                .reset_index()
            fig = px.treemap(market_df, path = ['category_1', 'category_2'], values = 'total_price', color = "category_1", title = 'Bike Sales Distribution by Category')
            fig.update_traces(textinfo = 'label+percent parent')
            fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            fig.update_layout(plot_bgcolor = PLOT_BACKGROUND, paper_bgcolor = PLOT_BACKGROUND, font_color = PLOT_FONT_COLOR)
            return fig
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update
        


@app.callback(
    Output("Bikeshops_bar", "figure"),
    Input("overall_dropdown", "value"))

def update_bikeshops_bar_chart(input_id):
    try:
        if input_id == "Overview":
            bikestore_df = df\
                .groupby(['bikeshop_name'])\
                .agg(total_price=('total_price', sum))\
                .sort_values('total_price', ascending=True)\
                .reset_index()

            fig = px.bar(bikestore_df, x='total_price', y='bikeshop_name', orientation='h', title="Sales per Bikeshop",
                        color='total_price',  # Optional: color by total_price for visual distinction
                        color_continuous_scale='viridis')  # Optional: color scale for color gradient
            return fig
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update


@app.callback(
        Output("performers_bar", "figure"),
        Input("overall_dropdown", "value")
)
def topPerformer_viz(selected_dropdown_value):
    
    try:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Top Performers", "Top Bikeshops"))
        
        if selected_dropdown_value in ["Mountain", "Road"]:
            top_performers = get_top_performers(df, 2015, selected_dropdown_value, 5)
            top_stores = get_top_stores(df, 2015, selected_dropdown_value, 5)
            y_column = 'category_2'
            title = f"Top {selected_dropdown_value} sub-categories"
        else:  # Overview or any other case
            top_performers, top_stores = get_total(df, 2015, 5)
            y_column = 'category_1'
            title = "All categories"

        # Top Performers
        fig1 = px.bar(top_performers, 
                    x='total_price', 
                    y=y_column, 
                    orientation='h',
                    title=title,
                    color='total_price',
                    color_continuous_scale='viridis')

        for trace in fig1['data']:
            fig.add_trace(trace, row=1, col=1)
        
        # Top Stores
        fig2 = px.bar(top_stores, 
                    x='total_price', 
                    y='bikeshop_name', 
                    orientation='h',
                    title="Top Stores",
                    color='total_price',
                    color_continuous_scale='viridis')
        
        for trace in fig2['data']:
            fig.add_trace(trace, row=1, col=2)
            
        fig.update_layout(xaxis_title="Total Sales")
        return fig 
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update

@app.callback(
        Output("forecast_dropdown", "options"),
        Input ("overall_dropdown", "value")
)

def update_forecast_dropdown(selected_dropdown_value):
    try:
        if selected_dropdown_value == "Overview":
            return [{"label": "Total", "value": "Total"}]
        elif selected_dropdown_value == "Mountain":
            return [{"label": shop, "value": shop} for shop in Mountain]
        elif selected_dropdown_value == "Road":
            return [{"label": shop, "value": shop} for shop in Road]
        elif selected_dropdown_value == "Bikeshops":
            return [{"label": shop, "value": shop} for shop in bikeshop_name]
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update

   
@app.callback(
        Output("line", "figure"),
        Input ("forecast_dropdown", "value")
)

def plot_forecast_line(selected_dropdown_value):
    try:
        if selected_dropdown_value == "Total":
            df1 = forecast_data_pull(total=True).dropna()
        elif selected_dropdown_value in Mountain:
            df1 = forecast_data_pull(Mountain=True).dropna()
        elif selected_dropdown_value in Road:
            df1 = forecast_data_pull(Road=True),dropna()
        elif selected_dropdown_value in bikeshop_name:
            df1 = forecast_data_pull(bikeshop=True).dropna()
        else:
            df1 = pd.DataFrame()  # Handle the case where no valid option is selected
        
        
        fig = go.Figure()

        color_map = {
            'LSTM_prediction': 'orange',
            'Arima_prediction': 'green'
        }
        
        for variable in df1['variable'].unique():
            data_subset = df1[df1['variable'] == variable]
            color = color_map.get(variable, 'blue')
            fig.add_trace(
                go.Scatter(
                    x=data_subset['order_date'],
                    y=data_subset['Sales'],
                    mode='lines',
                    name=variable,
                    line=dict(color=color),
                    showlegend=True
                )
            )
            
        fig.update_layout(
            title='Sales Forecast Comparison',
            xaxis_title='Order Date',
            yaxis_title='Sales',
            legend_title='Variables',
        )
        return fig
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update


@app.callback(
        Output("QBar", "figure"),
        Input ("forecast_dropdown", "value"))

def plot_Bar_Chart(selected_dropdown_value):
    try:
        # Bikeshops Data Bar Chart
        if selected_dropdown_value == "Total":
            df1 = forecast_data_pull(total=True).dropna()
        elif selected_dropdown_value in Mountain:
            df1 = forecast_data_pull(Mountain=True).dropna()
        elif selected_dropdown_value in Road:
            df1 = forecast_data_pull(Road=True).dropna()
        elif selected_dropdown_value in bikeshop_name:
            df1 = forecast_data_pull(bikeshop=True).dropna()
        else:
            df1 = pd.DataFrame()  # Handle the case where no valid option is selected

        # Remove unnecessary columns
        columns_not_needed = ['ci_lower', 'ci_upper']
        df1 = df1.drop(columns_not_needed, axis=1)

        # Remove nan values
        df1 = df1.dropna()

        if selected_dropdown_value in bikeshop_name:
            q_df1 = df1.reset_index().groupby(['bikeshop_name', 'variable']).resample('QE', on='order_date').agg({'Sales': 'sum'}).reset_index()
        elif selected_dropdown_value in [Mountain, Road]:
            q_df1 = df1.reset_index().groupby(['category', 'variable']).resample('QE', on='order_date').agg({'Sales': 'sum'}).reset_index()
        else:
            q_df1 = df1.reset_index().groupby(['variable']).resample('QE', on='order_date').agg({'Sales': 'sum'}).reset_index()

        # Define the color map
        color_map = {
            'LSTM_prediction': 'orange',
            'Arima_prediction': 'green'
        }

        # Create the initial figure
        fig = go.Figure()

        # Create traces for each bikeshop
        if selected_dropdown_value in bikeshop_name:
            for bikeshop in q_df1['bikeshop_name'].unique():
                for variable in q_df1['variable'].unique():
                    bikeshop_data = q_df1[(q_df1['bikeshop_name'] == bikeshop) & (q_df1['variable'] == variable)]
                    trace_name = f'{bikeshop}_{variable}'
                    color = color_map.get(variable, 'blue')

                    fig.add_trace(
                        go.Bar(
                            x=bikeshop_data['order_date'],
                            y=bikeshop_data['Sales'],
                            name=trace_name,
                            marker=dict(color=color),
                            visible=False  # Start with all traces hidden
                        )
                    )
        elif selected_dropdown_value in [Mountain, Road]:
            for category in q_df1['category'].unique():
                for variable in q_df1['variable'].unique():
                    category_data = q_df1[(q_df1['category'] == category) & (q_df1['variable'] == variable)]
                    trace_name = f'{category}_{variable}'
                    color = color_map.get(variable, 'blue')

                    fig.add_trace(
                        go.Bar(
                            x=category_data['order_date'],
                            y=category_data['Sales'],
                            name=trace_name,
                            marker=dict(color=color),
                            visible=False  # Start with all traces hidden
                        )
                    )
        elif selected_dropdown_value == "Total":
            for variable in q_df1['variable'].unique():
                total_data = q_df1[q_df1['variable'] == variable]
                trace_name = f'{variable}'
                color = color_map.get(variable, 'blue')

                fig.add_trace(
                    go.Bar(
                        x=total_data['order_date'],
                        y=total_data['Sales'],
                        name=trace_name,
                        marker=dict(color=color),
                        visible=False  # Start with all traces hidden
                    )
                )

        fig.update_layout(
            updatemenus=[dict(active=0)],
            title='Sales Data',
            xaxis_title='Date',
            yaxis_title='Sales',
            height=600,
            width=800,
            barmode='group'
        )

        return fig
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update
    
    
if __name__ == '__main__':
    app.run_server(debug=True, port = 8051)

# %%
