import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import pandas as pd
import pathlib
import pickle

# Function to collect data
def collect_data():
    # Dummy data function, replace with actual data collection
    data = {
        'category_1': ['Mountain', 'Mountain', 'Road', 'Road'],
        'category_2': ['A', 'B', 'C', 'D'],
        'bikeshop_name': ['Shop1', 'Shop2', 'Shop3', 'Shop4'],
        'total_price': [100, 200, 300, 400],
        'order_date': pd.date_range(start='1/1/2015', periods=4, freq='Q'),
        'variable': ['Actuals', 'Actuals', 'LSTM_prediction', 'Arima_prediction'],
        'Sales': [500, 600, 700, 800]
    }
    return pd.DataFrame(data)

df = collect_data()

Mountain = df[df['category_1'] == 'Mountain']['category_2'].unique().tolist()
Mountain.append('Mountain')

Road = df[df['category_1'] == 'Road']['category_2'].unique().tolist()
Road.append('Road')
bikeshop_name = df['bikeshop_name'].unique().tolist()

# APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
# Initialize the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)

# Constants for styling
PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'
COMMON_STYLE = {
    "font-size": "1.25rem",
    "color": "white",
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

title = html.H1("Bike Sales and Forecasting Dashboard", style=COMMON_STYLE)

# Layout items 
dropdown = dcc.Dropdown(
    id="overall_dropdown",
    options=[
        {"label": "Overview", "value": "Overview"},
        {"label": "Mountain", "value": "Mountain"},
        {"label": "Road", "value": "Road"},
        {"label": "Bikeshops", "value": "Bikeshops"}
    ],
    value="Overview"
)

TreeMap = dcc.Graph(
    id="TreeMap",
    config={"displayModeBar": False},
    style={"background-color": PLOT_BACKGROUND}
)

Bikeshops_bar = dcc.Graph(
    id="Bikeshops_bar",
    config={"displayModeBar": True},
    style={"background-color": PLOT_BACKGROUND}
)

performers = dcc.Graph(
    id="performers_bar",
    config={"displayModeBar": True},
    style={"background-color": PLOT_BACKGROUND}
)

Q12016_forecast_line = dcc.Graph(
    id="line",
    config={"displayModeBar": False},
    style={"background-color": PLOT_BACKGROUND}
)

Q12016_forecast_bar = dcc.Graph(
    id="QBar",
    config={"displayModeBar": False},
    style={"background-color": PLOT_BACKGROUND}
)

forecast_dropdown = dcc.Dropdown(
    id="forecast_dropdown",
    options=[],
    value='Total',
    style={'color': 'black'}
)

app.layout = html.Div([
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
        ]),
        dbc.Col([
            html.H5("FY15 Performance"),
            performers,
            html.Br(),
            html.Hr(),
            html.H5("Sales Forecast"),
            forecast_dropdown,
            Q12016_forecast_line,
            Q12016_forecast_bar
        ])
    ])
])

@app.callback(
    Output("TreeMap", "figure"),
    Input("overall_dropdown", "value")
)
def update_tree_map(selected_dropdown_value):
    if selected_dropdown_value == "Overview":
        market_df = df.groupby(['category_1', 'category_2']).agg(total_price=('total_price', 'sum')).reset_index()
        fig = px.treemap(market_df, path=['category_1', 'category_2'], values='total_price', color="category_1", title='Bike Sales Distribution by Category')
        fig.update_traces(textinfo='label+percent parent')
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.update_layout(plot_bgcolor=PLOT_BACKGROUND, paper_bgcolor=PLOT_BACKGROUND, font_color=PLOT_FONT_COLOR)
        return fig

@app.callback(
    Output("Bikeshops_bar", "figure"),
    Input("overall_dropdown", "value")
)
def update_bikeshops_bar_chart(selected_dropdown_value):
    if selected_dropdown_value == "Overview":
        bikestore_df = df.groupby(['bikeshop_name']).agg(total_price=('total_price', 'sum')).sort_values('total_price', ascending=True).reset_index()
        fig = px.bar(bikestore_df, x='total_price', y='bikeshop_name', orientation='h', title="Sales per Bikeshop", color='total_price', color_continuous_scale='viridis')
        fig.update_layout(xaxis_title="Total Sales")
        return fig

@app.callback(
    Output("performers_bar", "figure"),
    Input("overall_dropdown", "value")
)
def topPerformer_viz(selected_dropdown_value):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top Performers", "Top Bikeshops"))
    
    if selected_dropdown_value in ["Mountain", "Road"]:
        top_performers = get_top_performers(df, 2015, selected_dropdown_value, 5)
        top_stores = get_top_stores(df, 2015, selected_dropdown_value, 5)
        y_column = 'category_2'
        title = f"Top {selected_dropdown_value} sub-categories"
    else:
        top_performers, top_stores = get_total(df, 2015, 5)
        y_column = 'category_1'
        title = "All categories"

    fig1 = px.bar(top_performers, x='total_price', y=y_column, orientation='h', title=title, color='total_price', color_continuous_scale='viridis')
    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)
    
    fig2 = px.bar(top_stores, x='total_price', y='bikeshop_name', orientation='h', title="Top Stores", color='total_price', color_continuous_scale='viridis')
    for trace in fig2['data']:
        fig.add_trace(trace, row=1, col=2)
        
    fig.update_layout(xaxis_title="Total Sales")
    
    return fig 

@app.callback(
    Output("forecast_dropdown", "options"),
    Input("overall_dropdown", "value")
)
def update_forecast_dropdown(selected_dropdown_value):
    if selected_dropdown_value == "Overview":
        return [{"label": "Total", "value": "Total"}]
    elif selected_dropdown_value == "Mountain":
        return [{"label": shop, "value": shop} for shop in Mountain]
    elif selected_dropdown_value == "Road":
        return [{"label": shop, "value": shop} for shop in Road]
    elif selected_dropdown_value == "Bikeshops":
        return [{"label": shop, "value": shop} for shop in bikeshop_name]

@app.callback(
    Output("line", "figure"),
    Input("forecast_dropdown", "value")
)
def plot_forecast_line(selected_dropdown_value):
    if selected_dropdown_value == "Total":
        df1 = df
    elif selected_dropdown_value in Mountain:
        df1 = df[df['category_2'].isin(Mountain)]
    elif selected_dropdown_value in Road:
        df1 = df[df['category_2'].isin(Road)]
    elif selected_dropdown_value in bikeshop_name:
        df1 = df[df['bikeshop_name'] == selected_dropdown_value]
    else:
        df1 = df
        
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
        legend_title='Variables'
    )
    return fig

@app.callback(
    Output("QBar", "figure"),
    Input("forecast_dropdown", "value")
)
def plot_Bar_Chart(selected_dropdown_value):
    file_path = ART_PATH.joinpath('bikeshop_prediction.pkl')

    with open(file_path, 'rb') as f:
        bikeshop_sales_df = pickle.load(f)

    columns_not_needed = ['ci_lower', 'ci_upper']
    bikeshop_sales_df = bikeshop_sales_df.drop(columns_not_needed, axis=1)
    bikeshop_sales_df = bikeshop_sales_df.dropna()

    def rename_variable_columns(row):
        if row['order_date'] == '2015Q1':
            return 'Q1 YoY'
        elif row['order_date'] == '2015Q4':
            return "Actuals"
        elif row['variable'] == 'Actuals':
            return 'Other Quarters'
        else:
            return row['variable']

    q_bikeshops_df = bikeshop_sales_df.reset_index().groupby(['bikeshop_name', 'variable']).resample('QE', on='order_date').agg({'Sales': 'sum'}).reset_index()
    q_bikeshops_df['variable'] = q_bikeshops_df.apply(rename_variable_columns, axis=1)

    color_map = {
        'Q1 YoY': 'pink',
        'Actuals': 'purple',
        'Other Quarters': 'grey',
        'LSTM_prediction': 'orange',
        'Arima_prediction': 'green'
    }

    fig = go.Figure()

    for bikeshop in q_bikeshops_df['bikeshop_name'].unique():
        for variable in q_bikeshops_df['variable'].unique():
            bikeshop_data = q_bikeshops_df[(q_bikeshops_df['bikeshop_name'] == bikeshop) & (q_bikeshops_df['variable'] == variable)]
            trace_name = f'{bikeshop}_{variable}'
            color = color_map.get(variable, 'blue')

            fig.add_trace(
                go.Bar(
                    x=bikeshop_data['order_date'],
                    y=bikeshop_data['Sales'],
                    name=trace_name,
                    marker=dict(color=color),
                    visible=False
                )
            )

    for trace in fig.data:
        if 'Shop1' in trace.name:
            trace.visible = True
   
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

if __name__ == '__main__':
    app.run_server(debug=True)
