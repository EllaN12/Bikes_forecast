
import os

#data
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time
from my_pandas_extensions.Viz import get_top_performers, get_top_stores

#Analysis 
import numpy as np
import pandas as pd


# viz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict


#dash
import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output


app = dash.Dash(__name__)

# import data

df = collect_data()

df.head()

df1 = df[df['category_1']== 'Mountain']['category_2'].unique()

df2 = df[df['category_1']== 'Road']['category_2'].unique()

# Create an empty list to store the DataFrames
df1_list = []

for item in df1:
    # Aggregate total_price for the current item
    agg_df = df[df['category_2'] == item]\
              .groupby('category_2')\
              .agg(total_price=('total_price', sum))\
              .reset_index()\
              .assign(parent = 'Category_1: Mountain')
            
    agg_df['category_2'] = 'Category_2: ' + agg_df['category_2']
    # Append the result to the list
    df1_list.append(agg_df)

# Concatenate all the DataFrames in the list into a single DataFrame
cat1_df = pd.concat(df1_list, ignore_index=True)
cat1_df =cat1_df.set_index('category_2')

print(cat1_df)

df2_list = []

for item in df2:
    # Aggregate total_price for the current item
    agg_df = df[df['category_2'] == item]\
              .groupby('category_2')\
              .agg(total_price=('total_price', sum))\
              .reset_index()\
            .assign(parent = 'Category_1: Road')
        
    agg_df['category_2'] = 'Category_2: ' + agg_df['category_2']
    # Append the result to the list
    df2_list.append(agg_df)

# Concatenate all the DataFrames in the list into a single DataFrame
cat2_df = pd.concat(df2_list, ignore_index=True)
cat2_df= cat2_df.set_index('category_2')
print(cat2_df)



df_cat1 = df\
    .groupby('category_1')\
    .agg(total_price=(['total_price', '' sum))\
    .reset_index()\
    .set_index('category_1')\
    .assign(parent = 'Total')
df_cat1.index = 'Category_1: ' + df_cat1.index
    
total = df\
    .agg(total=('total_price', sum))\
    .reset_index()\
    .rename(columns={'index':'Total'})\
    .set_index('Total')\
    .assign(parent = '')
    
market_share_df = pd.concat([total, df_cat1, cat1_df, cat2_df],  axis=0)
ids = market_share_df.index
labels = market_share_df.index

#total = market_share_df[market_share_df['parent'] == '']
#Category_1 = market_share_df.index.str.contains('Category_1')
#Category_2 = market_share_df[market_share_df.index.str.contains('Category_2')]

fig = go.Figure(go.Sunburst(
      labels = market_share_df.index,
        parents = market_share_df['parent'].values,
        values = market_share_df['total_price'].values,
        
    ))

fig.update_layout(
    title='Sunburst Chart of Total Prices by Category',
    #sunburstcolorway=["#636efa","#ef553b","#00cc96","#ab63fa","#19d3f3",
                      #"#ffa15a","#ff6692","#b6e880","#ff97ff","#afbff9",
                      #"#eafc9f","#fffbcc"]
    width=800,
    height=800,
)
fig.show()

import plotly.io as pio
pio.renderers.default = "browser"
fig.show()

market_share_df\
    .reset_index()\
    .rename(columns = {'index': 'child'})



df= collect_data()

market_df = df.groupby(['category_1', 'category_2']) \
             .agg(total_price=('total_price', sum)) \
             .reset_index()

market_df.info()

# Function to format total_price as US dollar with two decimal places
def format_price(price):
    return "${:.2f}".format(price)

market_df['total_price_formatted'] = market_df['total_price'].apply(format_price)

# Assuming market_df is your original dataframe
fig = px.treemap(market_df.copy(), path=["category_1", "category_2"], values='total_price',
                 color="category_1", # Optional: color by category_1 for visual distinction
                 title="Bike Sales Distribution(Treemap)",
                 labels={'total_price': 'total_price_formatted'})

fig.update_layout(width=800, height=800)
fig.show()




# Store distribution viz in a variable

df = collect_data()

bikestore_df = df\
.groupby(['bikeshop_name'])\
.aggregate(total_price = ('total_price', sum))\
.sort_values('total_price', ascending=True)\
.reset_index()

fig = px.bar(bikestore_df, x='total_price', y='bikeshop_name', orientation='h', title="Sales per Bikeshop",
             color='total_price',  # Optional: color by total_price for visual distinction
             color_continuous_scale='viridis')  # Optional: color scale for color gradient

fig.update_layout(xaxis_title="Total Sales")  # Customize axis titles as needed

fig.show()


# top performers
df = collect_data()

df.info()
#filter 2015 data
df_2015 = df[df['order_date'].dt.year == 2015]

# filter mountain data

def get_top_performers(df, year, category,  top_n):
    
    df = collect_data()
    df_year = df[df['order_date'].dt.year == year]
    
    top_performers =  df_year[ df_year['category_1'] == category]\
        .groupby('category_2')\
        .aggregate(total_price = ('total_price', sum))\
        .sort_values('total_price', ascending=True)\
        .head(top_n)\
        .reset_index()
    return top_performers

Y15_top_performers = get_top_performers(df, 2015, "Road", 5)



def get_total(df, year, top_n):
    df = collect_data()
    df_year = df[df['order_date'].dt.year == year]
    
    top_performers = df_year\
        .groupby('category_1')\
        .aggregate(total_price = ('total_price', sum))\
        .sort_values('total_price', ascending=True)\
        .head(top_n)\
        .reset_index()
    
    
    top_stores = df_year\
        .groupby('bikeshop_name')\
        .aggregate(total_price = ('total_price', sum))\
        .sort_values('total_price', ascending=True)\
        .head(top_n)
    return top_performers, top_stores

p,s = get_total(df, 2015, 5)



def get_top_stores(df, year, category, top_n):
    df = collect_data()
    df_year = df[df['order_date'].dt.year == year]
    
    top_stores = df_year[df_year['category_1'] == category]\
        .groupby('bikeshop_name')\
        .aggregate(total_price = ('total_price', sum))\
        .sort_values('total_price', ascending=True)\
        .head(top_n)\
        .reset_index()
    return top_stores

df = collect_data()

Y15_top_stores = get_top_stores(df, 2015, "Road", 5)

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
                                    
topPerformer_viz(
    top_performers = Y15_top_performers,
    top_stores = Y15_top_stores
)

df = collect_data()
get_top_performers(df, 2015, 'Road', 5)

get_top_stores(df, 2015, 5)