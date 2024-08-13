@app.callback(
    Output("performers_bar", "figure"),
    [Input("Overview-option", "n_clicks"),
     Input("Mountain-option", "n_clicks"),
     Input("Road-option", "n_clicks"),
     Input("Bikeshops-option", "n_clicks")]
)
def topPerformer_viz(overview_clicks, mountain_clicks, road_clicks, bikeshops_clicks):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Top Performers", "Top Bikeshops"))

        if clicked_id == "Mountain-option":
            top_performers = get_top_performers(df, 2015, "Mountain", 5)
            top_stores = get_top_stores(df, 2015, "Mountain", 5)
            y_column = 'category_2'
            title = "Top Mountain sub-categories"
        elif clicked_id == "Road-option":
            top_performers = get_top_performers(df, 2015, "Road", 5)
            top_stores = get_top_stores(df, 2015, "Road", 5)
            y_column = 'category_2'
            title = "Top Road sub-categories"
        else:  # Overview or any other case
            top_performers, top_stores = get_total(df, 2015, 5)
            y_column = 'category_1'
            title = "Top categories"

        fig.add_trace(go.Bar(x=top_performers['total_price'], y=top_performers[y_column], orientation='h'), row=1, col=1)
        fig.add_trace(go.Bar(x=top_stores['total_price'], y=top_stores['bikeshop_name'], orientation='h'), row=1, col=2)

        fig.update_layout(title_text=title, plot_bgcolor=PLOT_BACKGROUND, paper_bgcolor=PLOT_BACKGROUND, font_color=PLOT_FONT_COLOR)
        return fig
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update



"""performers = dcc.Graph(
    id="performers_bar",
    config={"displayModeBar": True},
    style={"background-color": PLOT_BACKGROUND},
)"""



# Bijal and Team Attempt
"""
def make_forecast_graphs(mountain_clicks, road_clicks, selected_subcategory):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if clicked_id == "Overview-option":

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

        fig_line.update_layout(plot_bgcolor=PLOT_BACKGROUND, paper_bgcolor=PLOT_BACKGROUND, font_color=PLOT_FONT_COLOR)

        return fig_line
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return no_update """"

"""
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
def get_top_stores(df, year, category,):
    # No need to call collect_data() here, as df is already passed as an argument
    df_filtered = df[(df['order_date'].dt.year == year) & (df['category_1'] == category)]
    top_stores = df_filtered.groupby('bikeshop_name')\
        .agg(total_price=('total_price', 'sum'))\
        .sort_values('total_price', ascending=False)\
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

get_top_performers(df, 2015, 'Mountain', 5) """



