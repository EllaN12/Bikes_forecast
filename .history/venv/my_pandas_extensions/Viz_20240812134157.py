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
import os
import pandas_flavor as pf
import pathlib
import pickle

# data collection

from my_pandas_extensions.database import collect_data, summarize_by_time
df = collect_data()



@pf.register_dataframe_method
def get_top_performers(df, year, category):
    df_filtered = df[(df['order_date'].dt.year == year) & (df['category_1'] == category)]
    
    top_performers = df_filtered.groupby('category_2')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
        .reset_index()
    
    return top_performers

get_top_performers(df, 2015, 'Road')
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
def get_total(df, year):
    df_year = df[df['order_date'].dt.year == year]
    top_performers = df_year.groupby('category_1')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
        .reset_index()
    
    top_stores = df_year.groupby('bikeshop_name')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
        .reset_index()
    
    return top_performers, top_stores

get_total(df, 2015)


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




df = collect_data()


@pf.register_dataframe_method
def forecast_data_pull(Total = False, Mountain = False, Road = False, Bikeshops = False):
    
    df = collect_data()

    Mountain_list = df[df['category_1'] == 'Mountain']['category_2'].unique().tolist()

    Road_list = df[df['category_1'] == 'Road']['category_2'].unique().tolist()
    bikeshop_name = df['bikeshop_name'].unique().tolist()

    print("Current working directory:", os.getcwd())
    total_path = '04_artifacts/total_prediction.pkl'
    cat_1_path = '04_artifacts/cat_1_prediction.pkl'
    cat_2_path = '04_artifacts/cat_2_prediction.pkl'
    bikeshop_path = '04_artifacts/bikeshop_prediction.pkl'
    resolved_total_path = os.path.abspath(total_path)
    resolved_cat_1_path = os.path.abspath(cat_1_path)
    resolved_cat_2_path = os.path.abspath(cat_2_path)
    resolved_bikeshop_path = os.path.abspath(bikeshop_path)
    print("Resolved path to total file:", resolved_total_path)
    print("Resolved path to cat 1 file:", resolved_cat_1_path)
    print("Resolved path to cat 2 file:", resolved_cat_2_path)
    print("Resolved path to bikeshop file:", resolved_bikeshop_path)
    
    # total
    file_path = resolved_total_path
    with open(file_path, 'rb') as f:
        total_df = pickle.load(f)
        df = total_df.copy()
        
    # Mountain and Road
    file_path = resolved_cat_1_path
    with open(file_path, 'rb') as f:
        cat_1_df = pickle.load(f)
        
        
    file_path = resolved_cat_2_path
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
    file_path = resolved_bikeshop_path

    with open(file_path, 'rb') as f:
        bikeshops_df = pickle.load(f)
    
    if Total:
        return df
    elif Mountain:
        return mountain_df
    elif Road:
        return road_df
    elif Bikeshops:
        return bikeshops_df
    else:
        return 'Please select a category'
    
    

forecast_data_pull(Total= = True)




cat_1 = df[df['order_date'].dt.year == 2015]\
    .summarize_by_time(
        date_column = 'order_date', 
        value_column =['total_price', 'quantity'],
        groups = 'category_1',
        aggregate_function = 'sum',
        kind = 'timestamp',
        rule = 'Y',
        wide_format = False)\
        .reset_index()\
        .set_index('category_1')
    
    

cat_2 = df[df['order_date'].dt.year == 2015]\
    .summarize_by_time(
        date_column = 'order_date', 
        value_column =['total_price', 'quantity'],
        groups = 'category_2',
        aggregate_function = 'sum',
        kind = 'timestamp',
        rule = 'Y',
        wide_format = False)\
        .reset_index()\
        .set_index('category_2')
    
 
 
bikeshop = df[df['order_date'].dt.year == 2015]\
    .summarize_by_time(
        date_column = 'order_date', 
        value_column =['total_price', 'quantity'],
        groups = 'bikeshop_name',
        aggregate_function = 'sum',
        kind = 'timestamp',
        rule = 'Y',
        wide_format = False)\
        .reset_index()\
        .set_index('bikeshop_name')
            
            
sales_quant_df = pd.concat([cat_1, cat_2, bikeshop], axis = 0)       

sales_quant_df.to_pickle("/Users/ellandalla/Desktop/Bikes_forecast/venv/04_artifacts/sales_quant_df.pkl")