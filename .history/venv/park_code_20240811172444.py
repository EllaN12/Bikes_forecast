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
