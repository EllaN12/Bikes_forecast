import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

import plotly.express as px
from plotly.express import bar
from plotly.subplots import make_subplots
from plotly import graph_objects as go

import pandas as pd

import pandas_flavor as pf
import pathlib
import pickle

# data collection

from my_pandas_extensions.database import collect_data

df = collect_data()

@pf.register_dataframe_method
def get_top_performers(df, year, category, top_n):
    df_filtered = df[(df['order_date'].dt.year == year) & (df['category_1'] == category)]
    
    top_performers = df_filtered.groupby('category_2')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
        .head(top_n)\
        .reset_index()
    
    return top_performers

get_top_performers(df, 2015, 'Mountain', 5)
@pf.register_dataframe_method
def get_top_stores(df, year, category, top_n):
    # No need to call collect_data() here, as df is already passed as an argument
    df_filtered = df[(df['order_date'].dt.year == year) & (df['category_1'] == category)]
    top_stores = df_filtered.groupby('bikeshop_name')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
        .head(top_n)\
        .reset_index()
    return top_stores


@pf.register_dataframe_method
def get_total(df, year, top_n):
    # No need to call collect_data() here, as df is already passed as an argument
    df_year = df[df['order_date'].dt.year == year]
    
    top_performers = df_year.groupby('category_1')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
        .head(top_n)\
        .reset_index()
    
    top_stores = df_year.groupby('bikeshop_name')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
        .head(top_n)\
        .reset_index()
    
    return top_performers, top_stores



@pf.register_dataframe_method
def topPerformer_viz(top_performers, top_stores):
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top Performers", "Top Stores"))
    
    # Top Sub-Categories
    fig1 = px.bar(top_performers, x='total_price', y='category_2', orientation='h',
                  title = "Top sub-categories",
                  color='total_price',  # Optional: color by total_price for visual distinction
                  color_continuous_scale='viridis')
    
    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)
        
    # Top Stores
    fig2 = px.bar(top_stores, x='total_price', y='bikeshop_name', orientation='h',
                  title  = "Top Stores",
                  color='total_price',  # Optional: color by total_price for visual distinction
                  color_continuous_scale='viridis')
    
    for trace in fig2['data']:
        fig.add_trace(trace, row=1, col=2)
        
    fig.update_layout(xaxis_title="Total Sales")
    



    return fig 
df = collect_data()

get_top_performers(df, 2015, 'Mountain', 5)


@pf.register_dataframe_method
def forecast_data_pull(total = False, Mountain = False, Road = False, bikeshop = False):
    
    df = collect_data()

    Mountain_list = df[df['category_1'] == 'Mountain']['category_2'].unique().tolist()

    Road_list = df[df['category_1'] == 'Road']['category_2'].unique().tolist()
    bikeshop_name = df['bikeshop_name'].unique().tolist()
    
    # total
    file_path = '/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/04_artifacts/total_prediction.pkl'
    with open(file_path, 'rb') as f:
        total_df = pickle.load(f)
        df = total_df.copy()
        
    # Mountain and Road
    file_path = '/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/04_artifacts/cat_1_prediction.pkl'
    with open(file_path, 'rb') as f:
        cat_1_df = pickle.load(f)
        
        
    file_path = '/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/04_artifacts/cat_2_prediction.pkl'
    with open(file_path, 'rb') as f:
        cat_2_df = pickle.load(f)
        
# Mountain 
    M = cat_1_df[cat_1_df['category_1'] == 'Mountain'].rename(columns = {'category_1': 'Category'})
    M2 = cat_2_df[cat_2_df['category_2'].isin(Mountain_list)].rename(columns = {'category_2': 'Category'})
    mountain_df = pd.concat([M, M2], axis = 0)
    
# road
    R = cat_1_df[cat_1_df['category_1'] == 'Road'].rename(columns = {'category_1': 'Category'})
    R2 = cat_2_df[cat_2_df['category_2'].isin(Road_list)].rename(columns = {'category_2': 'Category'})
    road_df = pd.concat([R, R2], axis = 0)
    
# bikeshops
    file_path = '/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/04_artifacts/bikeshop_prediction.pkl'

    with open(file_path, 'rb') as f:
        bikeshops_df = pickle.load(f)
    
    if total:
        return df
    elif Mountain:
        return mountain_df
    elif Road:
        return road_df
    elif bikeshop:
        return bikeshops_df
    else:
        return 'Please select a category'
    
    

forecast_data_pull(bikeshop = True)