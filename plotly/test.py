import cudf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Initialize Dash app
app = dash.Dash(__name__)

# Read the dataset using cuDF (RAPIDS)
df = cudf.read_csv('passenger_satisfaction_cleaned.csv')

# Convert the necessary DataFrame to Pandas only when required for Plotly
def convert_to_pandas(cudf_df):
    return cudf_df.to_pandas()

# Columns for checklist and filtering
class_order = ['Eco', 'Eco Plus', 'Business']  # actual values in the dataset
class_options = [
    {'label': 'Economy', 'value': 'Eco'},
    {'label': 'Economy Plus', 'value': 'Eco Plus'},
    {'label': 'Business', 'value': 'Business'}
]
customer_types = sorted(df['customer_type'].to_pandas().unique())

customer_type_options = [
     {'label': 'Loyal Customer', 'value': 'Loyal Customer'},
     {'label': 'Disloyal Customer', 'value': 'disloyal Customer'}
     
]
# Columns for online services
online_services = ['online_support', 'ease_of_online_booking', 'online_boarding']

# Columns for inflight services bar chart
inflight_services = [
    'cleanliness', 'baggage_handling', 'leg_room_service', 'on-board_service',
    'inflight_entertainment', 'checkin_service', 'inflight_wifi_service', 'food_and_drink', 'seat_comfort'
]

# Columns for violin plot (default)
violin_services = ['cleanliness', 'leg_room_service', 'baggage_handling']

# KPIs helper function
def calculate_kpis(filtered_df):
    if filtered_df.empty:
        return 0, 0, 0
    satisfaction_rate = filtered_df['satisfaction'].mean()
    avg_delay_dissatisfied = filtered_df[filtered_df['satisfaction'] < 0.5]['departure_delay_in_minutes'].mean()
    avg_delay_satisfied = filtered_df[filtered_df['satisfaction'] >= 0.5]['departure_delay_in_minutes'].mean()
    return satisfaction_rate, avg_delay_dissatisfied, avg_delay_satisfied

app.layout = html.Div([
    html.H1(
        "Passenger Flight Experience Dashboard",
        style={'textAlign': 'center', 'marginBottom': '20px'}
    ),

    # KPIs (full-width spacing)
    html.Div(id='kpi-container', style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'gap': '20px',
        'marginBottom': '40px',
    }),

    # Filters
    html.Div([
        html.Div([
            html.Label("Select Class:"),
            dcc.Checklist(
                id='class-filter',
                options=class_options,
                value=class_order,  # default selected values
                inline=True,
                inputStyle={"margin-right": "5px", "margin-left": "15px"},
            )
        ], style={
            'flex': '1',
            'border': '1px solid #ccc',
            'padding': '15px',
            'borderRadius': '8px',
            'backgroundColor': '#ffffff'  # box remains white
        }),

        html.Div([
            html.Label("Select Customer Type:"),
            dcc.Checklist(
                id='customer-type-filter',
                options=customer_type_options,
                value=list(customer_types),  # default selection
                inline=True,
                inputStyle={"margin-right": "5px", "margin-left": "15px"},
            )
        ], style={
            'flex': '1',
            'border': '1px solid #ccc',
            'padding': '15px',
            'borderRadius': '8px',
            'backgroundColor': '#ffffff'  # box remains white
        })
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '40px'}),

    # Graphs sections...
    html.Div([
        html.Div([dcc.Graph(id='fig3')], style={
            'flex': '1',
            'border': '1px solid #ccc',
            'padding': '10px',
            'borderRadius': '8px',
            'backgroundColor': '#ffffff'
        }),
        html.Div([dcc.Graph(id='fig4')], style={
            'flex': '1',
            'border': '1px solid #ccc',
            'padding': '10px',
            'borderRadius': '8px',
            'backgroundColor': '#ffffff'
        })
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '40px'}),

    html.Div([dcc.Graph(id='fig1')], style={
        'border': '1px solid #ccc',
        'padding': '10px',
        'borderRadius': '8px',
        'marginBottom': '40px',
        'backgroundColor': '#ffffff'
    }),

    html.Div([
        html.Div([dcc.Graph(id='fig2')], style={
            'flex': '1',
            'border': '1px solid #ccc',
            'padding': '10px',
            'borderRadius': '8px',
            'backgroundColor': '#ffffff'
        }),
        html.Div([
            html.Label("Select Inflight Services for Distribution:"),
            dcc.Dropdown(
                id='service-filter',
                options=[{'label': col.replace('_', ' ').title(), 'value': col} for col in inflight_services],
                value=violin_services,
                multi=True,
                placeholder="Select services..."
            ),
            dcc.Graph(id='fig5')
        ], style={
            'flex': '1',
            'border': '1px solid #ccc',
            'padding': '10px',
            'borderRadius': '8px',
            'backgroundColor': '#ffffff'
        })
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '40px'}),

    html.Div([
        html.Label("Minimum passengers to display:"),
        dcc.Slider(
            id='min-passengers-slider',
            min=0,
            max=6673,  # <-- manually set maximum
            step=1,
            value=0,
            marks={0: '0', 6673: 'Max'},  # show 0 and Max labels
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={
        'border': '1px solid #ccc',
        'padding': '10px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'backgroundColor': '#ffffff'
    }),

    html.Div([dcc.Graph(id='fig6')], style={
        'border': '1px solid #ccc',
        'padding': '10px',
        'borderRadius': '8px',
        'backgroundColor': '#ffffff'
    })
], style={'backgroundColor': '#D7E3FC', 'padding': '20px', 'minHeight': '100vh'})

@app.callback(
    [
        Output('kpi-container', 'children'),
        Output('fig1', 'figure'),
        Output('fig2', 'figure'),
        Output('fig3', 'figure'),
        Output('fig4', 'figure'),
        Output('fig5', 'figure'),
        Output('fig6', 'figure')
    ],
    [
        Input('class-filter', 'value'),
        Input('customer-type-filter', 'value'),
        Input('service-filter', 'value'),
        Input('min-passengers-slider', 'value')
    ]
)
def update_dashboard(selected_classes, selected_customer_types, selected_services, min_passengers):
    # Filter cuDF DataFrame
    filtered_df = df[
        df['class'].isin(selected_classes) &
        df['customer_type'].isin(selected_customer_types)
    ]

    # --- KPIs ---
    satisfaction_rate, avg_delay_dissatisfied, avg_delay_satisfied = calculate_kpis(filtered_df)
    kpi_style = {
        'textAlign': 'center', 
        'padding': '20px', 
        'border': '1px solid #ccc',
        'borderRadius': '8px', 
        'flex': '1',  # makes KPIs evenly spaced
        'backgroundColor': '#f9f9f9'
    }
    kpis = [
        html.Div([
            html.H4("Average Satisfaction Rate"),
            html.P(f"{satisfaction_rate:.2f}", style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style=kpi_style),
        html.Div([
            html.H4("Avg Departure Delay (Dissatisfied)"),
            html.P(f"{avg_delay_dissatisfied:.1f} mins", style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style=kpi_style),
        html.Div([
            html.H4("Avg Departure Delay (Satisfied)"),
            html.P(f"{avg_delay_satisfied:.1f} mins", style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style=kpi_style)
    ]

    if filtered_df.empty:
        fig1 = go.Figure()
    else:
        # Step 1: Bin the flight distance into 100-mile ranges
        filtered_df['flight_distance_bins'] = (filtered_df['flight_distance'] // 100) * 100

        # Step 2: Group by flight distance bins and calculate the mean satisfaction
        grouped = filtered_df.groupby('flight_distance_bins')['satisfaction'].mean().reset_index()

        # Step 3: Convert grouped DataFrame to Pandas for Plotly compatibility
        grouped = grouped.to_pandas()  # Convert to Pandas

        # Step 4: Sort the data by flight distance bins
        grouped = grouped.sort_values('flight_distance_bins')

        # Step 5: Plot the line chart
        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(
            x=grouped['flight_distance_bins'],  # Flight distance bins on x-axis
            y=grouped['satisfaction'],  # Average satisfaction on y-axis
            mode='lines+markers',  # Use lines and markers
            line=dict(color='#719BFF', width=2),
            marker=dict(color='#1F77B4', size=6),
            name='Passenger Satisfaction',
            hovertemplate=(
                '<b>Flight Distance: </b>%{x} miles<br>'
                '<b>Average Satisfaction: </b>%{y:.2f}<extra></extra>'
            )
        ))

        # Step 6: Update layout to ensure x-axis starts at 0
        fig1.update_layout(
            title='Passenger Satisfaction Rate vs Flight Distance',
            xaxis=dict(
                title='Flight Distance (bins)',
                zeroline=False,  # Axis line color at zero
                linecolor='#8D99AE',  # Axis line color
                range=[0, grouped['flight_distance_bins'].max()]  # Ensure x-axis starts at 0
            ),
            yaxis=dict(
                title='Average Satisfaction',
                zeroline = False,  # Axis line color at zero
                linecolor='#8D99AE',  # Axis line color
            ),
            template='plotly_white',
            plot_bgcolor='white',  # White background for the plot
            margin=dict(t=40, b=40, l=40, r=40)  # Padding around the plot
        )
    # --- Figure 2: Horizontal Bar Chart (Inflight Services Satisfaction) ---
    fig2 = go.Figure()
    if not filtered_df.empty:
        normalized_df = filtered_df[inflight_services].astype(float) / 5.0
        avg_satisfaction_inflight = normalized_df.mean()
        # Convert to Pandas only when necessary for Plotly
        avg_satisfaction_inflight_pandas = convert_to_pandas(avg_satisfaction_inflight)
        fig2 = px.bar(
            avg_satisfaction_inflight_pandas.sort_values(ascending=True),
            orientation='h',
            labels={'value': 'Average Satisfaction (0-1)', 'index': 'Inflight Service'},
            title='Average Satisfaction by Inflight Services'
        )
        fig2.update_traces(
            marker_color='#719BFF',
            showlegend=False,
            hovertemplate='Inflight Service: %{y}<br>Average Satisfaction: %{x:.2f}<extra></extra>'
        )

        fig2.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(zerolinecolor='#8D99AE', showgrid=True,  # Enable gridlines
                gridcolor='lightgray', linecolor='#8D99AE'),
            yaxis=dict(zerolinecolor='#8D99AE', linecolor='#8D99AE'),
            title='Average Satisfaction by Inflight Services',
        )

    # --- Figure 3: Butterfly Chart (Departure vs Arrival Delay Satisfaction) ---


    fig3 = go.Figure()

    if not filtered_df.empty:
        # Bin the departure and arrival delays
        filtered_df['dep_delay_bins'] = (filtered_df['departure_delay_in_minutes'] // 250) * 250
        filtered_df['arr_delay_bins'] = (filtered_df['arrival_delay_in_minutes'] // 250) * 250

        # Group by departure and arrival delay bins and calculate the mean satisfaction
        dep_group = filtered_df.groupby('dep_delay_bins').agg(
            satisfaction=('satisfaction', 'mean'),
            passenger_count=('satisfaction', 'count')
        ).reset_index()

        arr_group = filtered_df.groupby('arr_delay_bins').agg(
            satisfaction=('satisfaction', 'mean'),
            passenger_count=('satisfaction', 'count')
        ).reset_index()

        # Convert cuDF DataFrame to Pandas for Plotly compatibility
        dep_group = dep_group.to_pandas()  # Convert to Pandas
        arr_group = arr_group.to_pandas()  # Convert to Pandas

        # Plot the stacked chart (departure delays on the left, arrival delays on the right)
        fig3.add_trace(go.Bar(
            y=dep_group['dep_delay_bins'],
            x=dep_group['satisfaction'],
            name='Departure Delay',
            orientation='h',
            marker_color='salmon',
            text=dep_group['satisfaction'].round(2),
            textposition='inside',
            hovertemplate=(
                '<b>Delay Bin: </b>%{y} mins<br>'
                '<b>Avg Satisfaction: </b>%{x:.2f}<br>'
                '<b>Passengers: </b>%{customdata}'
                '<extra></extra>'
            ),
            customdata=dep_group['passenger_count']  # Pass passenger counts to hover
        ))

        fig3.add_trace(go.Bar(
            y=arr_group['arr_delay_bins'],
            x=arr_group['satisfaction'],
            name='Arrival Delay',
            orientation='h',
            marker_color='lightblue',
            text=arr_group['satisfaction'].round(2),
            textposition='inside',
            hovertemplate=(
                '<b>Delay Bin: </b>%{y} mins<br>'
                '<b>Avg Satisfaction: </b>%{x:.2f}<br>'
                '<b>Passengers: </b>%{customdata}'
                '<extra></extra>'
            ),
            customdata=arr_group['passenger_count']  # Pass passenger counts to hover
        ))
        # Determine the max bin to pad nicely
        # Determine the max bin
        fig3.update_layout(
            yaxis=dict(
                title='Delay Bins (minutes)',
                autorange=False,  # Turn off automatic range
                range=[-100, 1350],  # Set max tick to 1250
                dtick=250,        # Show ticks every 250
                zeroline=False,
                linecolor='#8D99AE'
            ),
            barmode='group',
            xaxis=dict(title='Average Satisfaction', range=[0,1], zeroline=False, linecolor='#8D99AE'),
            template='plotly_white',
            title='Departure vs Arrival Delay Average Satisfaction',
        )
    # --- Figure 4: Radar Chart (Online Services Satisfaction) ---
    fig4 = go.Figure()
    if not filtered_df.empty:
        normalized_online = filtered_df[online_services].astype(float) / 5.0
        avg_online = normalized_online.mean()
        # Convert to Pandas only when necessary for Plotly
        avg_online_pandas = convert_to_pandas(avg_online)
        fig4.add_trace(go.Scatterpolar(
            r=avg_online_pandas.values,
            theta=[col.replace('_',' ').title() for col in online_services],
            fill='toself',
            fillcolor='#719BFF',
            name='Average Satisfaction',
            hovertemplate='<b>Avg Satisfaction: </b>%{r:.2f}<br><b>Service: </b>%{theta}<extra></extra>'
        ))
        fig4.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='#8D99AE')),
            title='Online Services Satisfaction',
            template='plotly_white'
        )

    # --- Figure 5: Violin Chart (Inflight Service Distribution) ---
    fig5 = go.Figure()
    if not filtered_df.empty and selected_services:
        violin_df = filtered_df[selected_services].astype(float) / 5.0
        violin_df = violin_df.melt(var_name='Service', value_name='Satisfaction')
        # Convert to Pandas only when necessary for Plotly
        violin_df_pandas = convert_to_pandas(violin_df)
        fig5 = px.violin(
            violin_df_pandas, x='Service', y='Satisfaction', box=True, points='all',
            title='Distribution of Satisfaction for Selected Inflight Services'
        )
        fig5.update_traces(line=dict(color='#719BFF'))
        fig5.update_layout(
            template='plotly_white',
            yaxis=dict(range=[0, 1], zerolinecolor='#8D99AE', linecolor='#8D99AE'),
            violinmode='group'
        )

    # --- Figure 6: Scatter with Improved Layout ---
    fig6 = go.Figure()
    if not filtered_df.empty:
        # Prepare the data for scatter plot
        scatter_df = filtered_df[['id', 'flight_distance', 'cleanliness']].copy()
        scatter_df['distance_bin'] = (scatter_df['flight_distance'] // 100) * 100
        scatter_df['cleanliness_norm'] = scatter_df['cleanliness'] / 5.0

        # Grouping by 'distance_bin' and calculating average cleanliness and passenger count
        grouped = scatter_df.groupby('distance_bin').agg(
            avg_cleanliness=('cleanliness_norm', 'mean'),
            passenger_count=('id', 'count')
        ).reset_index()

        # Convert to Pandas DataFrame for easier manipulation with idxmax()
        grouped = grouped.to_pandas()  # Convert cuDF to Pandas DataFrame

        # Filter out groups with fewer passengers than the minimum threshold
        grouped = grouped[grouped['passenger_count'] >= min_passengers]

        if not grouped.empty:
            # Calculate maximum and minimum passenger counts
            max_count = grouped['passenger_count'].max()
            min_count = grouped['passenger_count'].min()

            # Size of the marker bubbles is proportional to the passenger count (log scale)
            # Convert to NumPy array here
            marker_sizes = np.log1p(grouped['passenger_count'].to_numpy()) / np.log1p(max_count) * 100

            # Convert Pandas DataFrame columns to Pandas Series
            distance_bin_pandas = grouped['distance_bin']
            avg_cleanliness_pandas = grouped['avg_cleanliness']
            passenger_count_pandas = grouped['passenger_count']

            # Adding scatter plot for flight distances and cleanliness satisfaction
            fig6.add_trace(go.Scatter(
                x=distance_bin_pandas,  # Use Pandas Series
                y=avg_cleanliness_pandas,  # Use Pandas Series
                mode='markers',
                marker=dict(
                    size=marker_sizes,  # Set bubble size based on the log of passenger count
                    color=avg_cleanliness_pandas,  # Color by cleanliness satisfaction
                    colorscale='Viridis',  # Colorscale for cleanliness satisfaction
                    showscale=True,  # Display the color scale on the side
                    colorbar=dict(title='Cleanliness', x=1.03),  # Adjust the color bar position
                    line=dict(width=1, color='black')  # Set a border color for each marker
                ),
                text=[f"Flight Distance: {d} miles<br>Avg Cleanliness: {c:.2f}<br>Passengers: {p}"
                    for d, c, p in zip(distance_bin_pandas, avg_cleanliness_pandas, passenger_count_pandas)],  # Use Pandas Series
                hoverinfo='text',
                showlegend=False
            ))

            # Adding invisible (dummy) traces for the legend to represent the bubble sizes
            fig6.add_trace(go.Scatter(
                x=[None], y=[None],  # Invisible points
                mode='markers',
                marker=dict(
                    size=10,  # Size for the dummy bubble
                    color='rgba(0,0,0,0)',  # Transparent color for the plot
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name=f'{min_count} passengers',  # Show a legend entry for the smallest size
                showlegend=True
            ))

            fig6.add_trace(go.Scatter(
                x=[None], y=[None],  # Invisible points
                mode='markers',
                marker=dict(
                    size=30,  # Size for the dummy bubble
                    color='rgba(0,0,0,0)',  # Transparent color for the plot
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name=f'{int(max_count / 4)} passengers',  # Show a legend entry for a larger size
                showlegend=True
            ))

            fig6.add_trace(go.Scatter(
                x=[None], y=[None],  # Invisible points
                mode='markers',
                marker=dict(
                    size=50,  # Size for the dummy bubble
                    color='rgba(0,0,0,0)',  # Transparent color for the plot
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name=f'{int(max_count / 2)} passengers',  # Show a legend entry for an even larger size
                showlegend=True
            ))

            fig6.add_trace(go.Scatter(
                x=[None], y=[None],  # Invisible points
                mode='markers',
                marker=dict(
                    size=500,  # Size for the dummy bubble
                    color='rgba(0,0,0,0)',  # Transparent color for the plot
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name=f'{max_count} passengers',  # Show a legend entry for the largest size
                showlegend=True
            ))

            # Annotations for max/min
            max_row = grouped.loc[grouped['passenger_count'].idxmax()]
            min_row = grouped.loc[grouped['passenger_count'].idxmin()]

            fig6.add_annotation(
                x=max_row['distance_bin'], y=max_row['avg_cleanliness'],
                text=f"{max_row['passenger_count']} passengers",
                showarrow=True, arrowhead=2, ax=0, ay=-60
            )

            # Update layout
            fig6.update_layout(
                title='Average Cleanliness per 100 Miles (Bubble Size = Passenger Count)',
                xaxis=dict(
                    title='Flight Distance (100-mile bins)',
                    zeroline=False,
                    linecolor='#8D99AE'
                ),
                yaxis=dict(
                    title='Avg Cleanliness Satisfaction (0-1)',
                    linecolor='#8D99AE'
                ),
                template='plotly_white',
                legend=dict(x=1.1, y=1, xanchor='left', yanchor='top'),
                margin=dict(t=40, b=40, l=20, r=20)  # Increase right margin for extra space
            )
        return kpis, fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    app.run(debug=True)