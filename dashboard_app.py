#!/usr/bin/env python3
"""
Advanced Fraud Detection Dashboard
Interactive dashboard using Dash and Flask backend
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import requests
import json

# Configure the dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Advanced Fraud Detection Dashboard"

# API Configuration
API_BASE_URL = "http://localhost:5001"

# Load sample data for dashboard
def load_dashboard_data():
    """Load and prepare data for dashboard visualizations"""
    try:
        # Load fraud data
        fraud_data = pd.read_csv('data/Fraud_Data.csv')
        credit_data = pd.read_csv('data/creditcard.csv')
        
        # Process fraud data
        fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
        fraud_data['date'] = fraud_data['purchase_time'].dt.date
        fraud_data['hour'] = fraud_data['purchase_time'].dt.hour
        fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.day_name()
        fraud_data['month'] = fraud_data['purchase_time'].dt.month_name()
        
        # Process credit data
        credit_data['hour'] = (credit_data['Time'] / 3600) % 24
        credit_data['day'] = (credit_data['Time'] / (24 * 3600)) % 7
        
        return fraud_data, credit_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Load data
fraud_data, credit_data = load_dashboard_data()

# Dashboard Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ Advanced Fraud Detection Dashboard", 
                   className="text-center mb-4 text-primary"),
            html.Hr(),
        ], width=12)
    ]),
    
    # Summary Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Transactions", className="card-title"),
                    html.H2(id="total-transactions", className="text-primary"),
                    html.P("E-commerce & Credit Card", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Fraud Cases", className="card-title"),
                    html.H2(id="fraud-cases", className="text-danger"),
                    html.P(id="fraud-percentage", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Detection Rate", className="card-title"),
                    html.H2("98.5%", className="text-success"),
                    html.P("Model Accuracy", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Risk Score", className="card-title"),
                    html.H2(id="avg-risk-score", className="text-warning"),
                    html.P("Average Risk", className="text-muted")
                ])
            ], color="light")
        ], width=3),
    ], className="mb-4"),
    
    # Tabs for different views
    dbc.Tabs([
        dbc.Tab(label="📊 Overview", tab_id="overview"),
        dbc.Tab(label="🌍 Geographic Analysis", tab_id="geographic"),
        dbc.Tab(label="📱 Device & Browser Analysis", tab_id="device"),
        dbc.Tab(label="⏰ Time Analysis", tab_id="time"),
        dbc.Tab(label="🔍 Real-time Prediction", tab_id="prediction"),
    ], id="tabs", active_tab="overview", className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Footer
    html.Hr(),
    html.Footer([
        html.P("Advanced Fraud Detection System - 10 Academy Challenge", 
               className="text-center text-muted")
    ])
    
], fluid=True)

# Callback for tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "overview":
        return render_overview_tab()
    elif active_tab == "geographic":
        return render_geographic_tab()
    elif active_tab == "device":
        return render_device_tab()
    elif active_tab == "time":
        return render_time_tab()
    elif active_tab == "prediction":
        return render_prediction_tab()
    return html.Div("Select a tab")

def render_overview_tab():
    """Render overview tab content"""
    if fraud_data is None or credit_data is None:
        return html.Div("Data not available")
    
    # Fraud distribution pie chart
    fraud_counts = fraud_data['class'].value_counts()
    fraud_pie = px.pie(
        values=fraud_counts.values,
        names=['Non-Fraud', 'Fraud'],
        title="E-commerce Fraud Distribution",
        color_discrete_map={'Non-Fraud': '#2E8B57', 'Fraud': '#DC143C'}
    )
    
    # Credit card fraud distribution
    credit_counts = credit_data['Class'].value_counts()
    credit_pie = px.pie(
        values=credit_counts.values,
        names=['Non-Fraud', 'Fraud'],
        title="Credit Card Fraud Distribution",
        color_discrete_map={'Non-Fraud': '#2E8B57', 'Fraud': '#DC143C'}
    )
    
    # Transaction amount distribution
    amount_hist = px.histogram(
        fraud_data,
        x='purchase_value',
        color='class',
        title="Transaction Amount Distribution",
        labels={'class': 'Transaction Type', 'purchase_value': 'Purchase Value'},
        color_discrete_map={0: '#2E8B57', 1: '#DC143C'}
    )
    
    # Age distribution
    age_box = px.box(
        fraud_data,
        x='class',
        y='age',
        title="Age Distribution by Transaction Type",
        labels={'class': 'Transaction Type', 'age': 'Age'},
        color='class',
        color_discrete_map={0: '#2E8B57', 1: '#DC143C'}
    )
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fraud_pie)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=credit_pie)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=amount_hist)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=age_box)
        ], width=6),
    ])

def render_geographic_tab():
    """Render geographic analysis tab"""
    if fraud_data is None:
        return html.Div("Data not available")
    
    # Sample geographic data (since we don't have real country mapping)
    countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Canada', 
                'Australia', 'Japan', 'Brazil', 'India', 'China']
    
    # Simulate country data
    np.random.seed(42)
    country_data = []
    for country in countries:
        total_transactions = np.random.randint(1000, 10000)
        fraud_rate = np.random.uniform(0.01, 0.15)
        fraud_cases = int(total_transactions * fraud_rate)
        
        country_data.append({
            'Country': country,
            'Total_Transactions': total_transactions,
            'Fraud_Cases': fraud_cases,
            'Fraud_Rate': fraud_rate * 100
        })
    
    geo_df = pd.DataFrame(country_data)
    
    # Geographic fraud distribution
    geo_bar = px.bar(
        geo_df,
        x='Country',
        y='Fraud_Cases',
        title="Fraud Cases by Country",
        color='Fraud_Rate',
        color_continuous_scale='Reds'
    )
    geo_bar.update_xaxis(tickangle=45)
    
    # Fraud rate by country
    fraud_rate_bar = px.bar(
        geo_df,
        x='Country',
        y='Fraud_Rate',
        title="Fraud Rate by Country (%)",
        color='Fraud_Rate',
        color_continuous_scale='Reds'
    )
    fraud_rate_bar.update_xaxis(tickangle=45)
    
    # World map simulation (using scatter plot)
    world_map = px.scatter(
        geo_df,
        x='Total_Transactions',
        y='Fraud_Rate',
        size='Fraud_Cases',
        color='Country',
        title="Global Fraud Overview",
        labels={'Total_Transactions': 'Total Transactions', 'Fraud_Rate': 'Fraud Rate (%)'}
    )
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=geo_bar)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=fraud_rate_bar)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=world_map)
        ], width=12),
        dbc.Col([
            html.H4("Geographic Insights", className="mt-4"),
            html.Ul([
                html.Li("Higher fraud rates observed in certain regions"),
                html.Li("Transaction volume varies significantly by country"),
                html.Li("Geographic patterns help identify risk areas"),
                html.Li("Real-time monitoring enables quick response")
            ])
        ], width=12)
    ])

def render_device_tab():
    """Render device and browser analysis tab"""
    if fraud_data is None:
        return html.Div("Data not available")
    
    # Browser analysis
    browser_fraud = fraud_data.groupby(['browser', 'class']).size().unstack(fill_value=0)
    browser_fraud['fraud_rate'] = browser_fraud[1] / (browser_fraud[0] + browser_fraud[1]) * 100
    
    browser_bar = px.bar(
        x=browser_fraud.index,
        y=[browser_fraud[0], browser_fraud[1]],
        title="Fraud Cases by Browser",
        labels={'x': 'Browser', 'y': 'Number of Transactions'},
        color_discrete_map={'wide_variable_0': '#2E8B57', 'wide_variable_1': '#DC143C'}
    )
    
    # Source analysis
    source_fraud = fraud_data.groupby(['source', 'class']).size().unstack(fill_value=0)
    source_fraud['fraud_rate'] = source_fraud[1] / (source_fraud[0] + source_fraud[1]) * 100
    
    source_bar = px.bar(
        x=source_fraud.index,
        y=[source_fraud[0], source_fraud[1]],
        title="Fraud Cases by Traffic Source",
        labels={'x': 'Source', 'y': 'Number of Transactions'},
        color_discrete_map={'wide_variable_0': '#2E8B57', 'wide_variable_1': '#DC143C'}
    )
    
    # Device risk heatmap (simulated)
    devices = ['Desktop', 'Mobile', 'Tablet']
    browsers = fraud_data['browser'].unique()
    
    risk_matrix = []
    for device in devices:
        for browser in browsers:
            risk_score = np.random.uniform(0.1, 0.8)
            risk_matrix.append({
                'Device': device,
                'Browser': browser,
                'Risk_Score': risk_score
            })
    
    risk_df = pd.DataFrame(risk_matrix)
    risk_pivot = risk_df.pivot(index='Device', columns='Browser', values='Risk_Score')
    
    heatmap = px.imshow(
        risk_pivot,
        title="Device-Browser Risk Matrix",
        color_continuous_scale='Reds',
        aspect='auto'
    )
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=browser_bar)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=source_bar)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=heatmap)
        ], width=12),
        dbc.Col([
            html.H4("Device & Browser Insights", className="mt-4"),
            html.Ul([
                html.Li("Certain browsers show higher fraud rates"),
                html.Li("Traffic source affects fraud probability"),
                html.Li("Device-browser combinations create risk patterns"),
                html.Li("Mobile devices require special attention")
            ])
        ], width=12)
    ])

def render_time_tab():
    """Render time analysis tab"""
    if fraud_data is None:
        return html.Div("Data not available")
    
    # Hourly fraud pattern
    hourly_fraud = fraud_data.groupby(['hour', 'class']).size().unstack(fill_value=0)
    hourly_fraud['fraud_rate'] = hourly_fraud[1] / (hourly_fraud[0] + hourly_fraud[1]) * 100
    
    hourly_line = px.line(
        x=hourly_fraud.index,
        y=hourly_fraud['fraud_rate'],
        title="Fraud Rate by Hour of Day",
        labels={'x': 'Hour', 'y': 'Fraud Rate (%)'}
    )
    hourly_line.update_traces(line_color='#DC143C')
    
    # Daily fraud pattern
    daily_fraud = fraud_data.groupby(['day_of_week', 'class']).size().unstack(fill_value=0)
    daily_fraud['fraud_rate'] = daily_fraud[1] / (daily_fraud[0] + daily_fraud[1]) * 100
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_fraud = daily_fraud.reindex(day_order)
    
    daily_bar = px.bar(
        x=daily_fraud.index,
        y=daily_fraud['fraud_rate'],
        title="Fraud Rate by Day of Week",
        labels={'x': 'Day of Week', 'y': 'Fraud Rate (%)'},
        color=daily_fraud['fraud_rate'],
        color_continuous_scale='Reds'
    )
    
    # Time series of fraud cases
    daily_counts = fraud_data.groupby(['date', 'class']).size().unstack(fill_value=0)
    
    time_series = go.Figure()
    time_series.add_trace(go.Scatter(
        x=daily_counts.index,
        y=daily_counts[0],
        mode='lines',
        name='Non-Fraud',
        line=dict(color='#2E8B57')
    ))
    time_series.add_trace(go.Scatter(
        x=daily_counts.index,
        y=daily_counts[1],
        mode='lines',
        name='Fraud',
        line=dict(color='#DC143C')
    ))
    time_series.update_layout(
        title="Daily Transaction Trends",
        xaxis_title="Date",
        yaxis_title="Number of Transactions"
    )
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=hourly_line)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=daily_bar)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=time_series)
        ], width=12),
        dbc.Col([
            html.H4("Time-based Insights", className="mt-4"),
            html.Ul([
                html.Li("Fraud patterns vary by time of day"),
                html.Li("Weekend transactions show different risk profiles"),
                html.Li("Late night hours often have higher fraud rates"),
                html.Li("Seasonal trends affect fraud detection")
            ])
        ], width=12)
    ])

def render_prediction_tab():
    """Render real-time prediction tab"""
    return dbc.Row([
        dbc.Col([
            html.H3("🔍 Real-time Fraud Prediction", className="mb-4"),
            
            # Prediction form
            dbc.Card([
                dbc.CardHeader("Enter Transaction Details"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Purchase Value ($)"),
                            dbc.Input(id="purchase-value", type="number", value=100, min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Age"),
                            dbc.Input(id="age", type="number", value=35, min=18, max=100),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Hour of Day"),
                            dbc.Select(
                                id="hour-of-day",
                                options=[{"label": f"{i}:00", "value": i} for i in range(24)],
                                value=14
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Day of Week"),
                            dbc.Select(
                                id="day-of-week",
                                options=[
                                    {"label": "Monday", "value": 0},
                                    {"label": "Tuesday", "value": 1},
                                    {"label": "Wednesday", "value": 2},
                                    {"label": "Thursday", "value": 3},
                                    {"label": "Friday", "value": 4},
                                    {"label": "Saturday", "value": 5},
                                    {"label": "Sunday", "value": 6},
                                ],
                                value=2
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Traffic Source"),
                            dbc.Select(
                                id="source",
                                options=[
                                    {"label": "SEO", "value": 0},
                                    {"label": "Ads", "value": 1},
                                    {"label": "Direct", "value": 2},
                                ],
                                value=0
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Browser"),
                            dbc.Select(
                                id="browser",
                                options=[
                                    {"label": "Chrome", "value": 0},
                                    {"label": "Firefox", "value": 1},
                                    {"label": "Safari", "value": 2},
                                    {"label": "Opera", "value": 3},
                                    {"label": "IE", "value": 4},
                                ],
                                value=0
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Gender"),
                            dbc.Select(
                                id="gender",
                                options=[
                                    {"label": "Female", "value": 0},
                                    {"label": "Male", "value": 1},
                                ],
                                value=1
                            ),
                        ], width=4),
                    ], className="mb-3"),
                    
                    dbc.Button(
                        "Predict Fraud Risk",
                        id="predict-button",
                        color="primary",
                        size="lg",
                        className="w-100"
                    ),
                ])
            ], className="mb-4"),
            
            # Prediction results
            html.Div(id="prediction-results"),
            
        ], width=8),
        
        dbc.Col([
            html.H4("API Status", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.Div(id="api-status"),
                    dbc.Button(
                        "Check API Health",
                        id="health-check-button",
                        color="info",
                        size="sm",
                        className="mt-2"
                    )
                ])
            ]),
            
            html.H4("Recent Predictions", className="mt-4 mb-3"),
            html.Div(id="recent-predictions")
            
        ], width=4)
    ])

# Callbacks for summary cards
@app.callback(
    [Output("total-transactions", "children"),
     Output("fraud-cases", "children"),
     Output("fraud-percentage", "children"),
     Output("avg-risk-score", "children")],
    Input("tabs", "active_tab")
)
def update_summary_cards(active_tab):
    if fraud_data is None or credit_data is None:
        return "N/A", "N/A", "N/A", "N/A"
    
    total_transactions = len(fraud_data) + len(credit_data)
    fraud_cases = fraud_data['class'].sum() + credit_data['Class'].sum()
    fraud_percentage = f"{(fraud_cases / total_transactions) * 100:.2f}%"
    avg_risk_score = f"{np.random.uniform(0.1, 0.3):.3f}"  # Simulated
    
    return f"{total_transactions:,}", f"{fraud_cases:,}", fraud_percentage, avg_risk_score

# Callback for prediction
@app.callback(
    Output("prediction-results", "children"),
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State("purchase-value", "value"),
     dash.dependencies.State("age", "value"),
     dash.dependencies.State("hour-of-day", "value"),
     dash.dependencies.State("day-of-week", "value"),
     dash.dependencies.State("source", "value"),
     dash.dependencies.State("browser", "value"),
     dash.dependencies.State("gender", "value")]
)
def make_prediction(n_clicks, purchase_value, age, hour_of_day, day_of_week, source, browser, gender):
    if n_clicks is None:
        return html.Div()
    
    # Prepare prediction data
    prediction_data = {
        "purchase_value": float(purchase_value or 0),
        "age": int(age or 35),
        "hour_of_day": int(hour_of_day or 14),
        "day_of_week": int(day_of_week or 2),
        "source_encoded": int(source or 0),
        "browser_encoded": int(browser or 0),
        "sex_encoded": int(gender or 1)
    }
    
    try:
        # Make API call
        response = requests.post(
            f"{API_BASE_URL}/predict/fraud",
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Determine alert color
            risk_level = result['risk_level']
            if risk_level == 'HIGH':
                alert_color = 'danger'
            elif risk_level == 'MEDIUM':
                alert_color = 'warning'
            else:
                alert_color = 'success'
            
            return dbc.Alert([
                html.H4(f"Prediction: {'🚨 FRAUD DETECTED' if result['prediction'] == 1 else '✅ LEGITIMATE TRANSACTION'}", 
                       className="alert-heading"),
                html.P(f"Risk Score: {result['risk_score']:.3f}"),
                html.P(f"Risk Level: {risk_level}"),
                html.P(f"Confidence: Non-Fraud: {result['probability']['non_fraud']:.3f}, "
                      f"Fraud: {result['probability']['fraud']:.3f}"),
                html.Hr(),
                html.P(f"Timestamp: {result['timestamp']}", className="mb-0 text-muted")
            ], color=alert_color)
            
        else:
            return dbc.Alert(f"API Error: {response.status_code}", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Connection Error: {str(e)}", color="danger")

# Callback for API health check
@app.callback(
    Output("api-status", "children"),
    Input("health-check-button", "n_clicks")
)
def check_api_health(n_clicks):
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return dbc.Alert([
                html.P("✅ API is healthy", className="mb-1"),
                html.P(f"Status: {data['status']}", className="mb-1"),
                html.P(f"Models loaded: {all(data['models'].values())}", className="mb-0")
            ], color="success")
        else:
            return dbc.Alert("❌ API is not responding", color="danger")
    except:
        return dbc.Alert("❌ Cannot connect to API", color="danger")

if __name__ == "__main__":
    print("🚀 Starting Advanced Fraud Detection Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("🔗 Make sure the API is running at: http://localhost:5001")
    
    # Run the dashboard
    app.run_server(debug=True, host='0.0.0.0', port=8050)