import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
from datetime import datetime, timedelta
from prophet import Prophet

# Load the trained Prophet model
model = joblib.load('prophet_model.pkl')

# Sample data for the initial display (this should be replaced with your actual data)
df = pd.read_csv('goldstock.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Function to get forecast data
def get_forecast_data(start_date, end_date):
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]

    future = model.make_future_dataframe(periods=365 * 2)
    forecast = model.predict(future)
    forecast_filtered = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]

    return filtered_df, forecast_filtered

# Custom colors and styling
custom_primary_color = '#2a9df4'  # Custom primary color (blue)
custom_secondary_color = '#Yellow'  # Custom secondary color (light orange)
custom_text_color = 'Blue'  # Custom text color (dark gray)
custom_background_color = '#f0f0f0'  # Custom background color (light gray)
custom_sidebar_color = 'Yellow'  # Custom sidebar background color (light pink)
custom_metrics_color = '#fce4ec'  # Custom metrics background color (light pink)
custom_metrics_text_color = 'Black'  # Custom metrics text color (dark pink)

# Apply custom CSS styles using st.markdown()
st.markdown(
    f"""
    <style>
    .sidebar .sidebar-content {{
        background-color: {custom_sidebar_color} !important;
        color: {custom_text_color};
        z-index: 1; /* Ensure sidebar elements are on top */
    }}
    .sidebar .sidebar-content .sidebar-section {{
        background-color: {custom_sidebar_color} !important;
    }}
    .sidebar .sidebar-content .stButton {{
        color: {custom_text_color} !important;
        background-color: {custom_primary_color} !important;
    }}
    .summary-metrics {{
        background-color: {custom_metrics_color};
        padding: 20px;
        margin-top: 20px;
        border-radius: 10px;
    }}
    .summary-metrics ul li {{
        color: {custom_metrics_text_color};
        font-weight: bold;
        font-size: 18px;
        padding: 10px;
        margin-top: 10px;
        border-radius: 5px;
    }}
    .stApp header {{
        background-color: {custom_primary_color} !important;
        padding: 10px;
        border-radius: 5px;
        position: sticky; /* Make the title sticky */
        top: 0; /* Stick to the top */
        z-index: 1000; /* Ensure it's above other content */
    }}
    .stApp h1, .stApp h2 {{
        color: {custom_primary_color} !important;
    }}
    .stApp p, .stApp li {{
        color: {custom_text_color} !important;
    }}
    .st-df-body {{
        color: {custom_text_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and main content
st.title("📈 Gold Stock Prediction Dashboard")


st.sidebar.header("Select Date Range:")
default_start_date = df['Date'].min()
default_end_date = pd.to_datetime('2019-01-01')
max_end_date = datetime.now() + timedelta(days=365 * 4)
start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date, max_value=max_end_date)
submit_button = st.sidebar.button("Submit", key='submit_button')

# Initialize filtered data
if submit_button:
    filtered_df, forecast_filtered = get_forecast_data(start_date, end_date)
else:
    filtered_df, forecast_filtered = get_forecast_data(default_start_date, default_end_date)

# Plotting the graphs
st.header("Gold Stock")

price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], mode='lines', name='Actual Close', line=dict(color='blue')))
price_fig.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yhat'], mode='lines', name='Predicted Close', line=dict(color='red')))
price_fig.update_layout(title='Gold Closing Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(price_fig, use_container_width=True)

volume_fig = go.Figure()
volume_fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Volume'], mode='lines', name='Volume', line=dict(color='yellow')))
volume_fig.update_layout(title='Gold Trading Volume', xaxis_title='Date', yaxis_title='Volume')
st.plotly_chart(volume_fig, use_container_width=True)

ohl_fig = go.Figure()
ohl_fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Open'], mode='lines', name='Open', line=dict(color='purple')))
ohl_fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['High'], mode='lines', name='High', line=dict(color='orange')))
ohl_fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Low'], mode='lines', name='Low', line=dict(color='brown')))
ohl_fig.update_layout(title='Gold Prices: Open, High, Low', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(ohl_fig, use_container_width=True)

# Summary Metrics in a separate box below the submit button
st.markdown("<br>", unsafe_allow_html=True)  # Add some space below the charts

st.markdown(
    f"""
    <div class="summary-metrics">
    <h2 style="color: {custom_primary_color}; text-align: center;">Summary Metrics</h2>
    <ul style="list-style-type: none; padding: 0;">
        <li>
            <strong style="color: {custom_metrics_text_color}; font-size: 24px;">Predicted Closing Price on {end_date}:</strong> <span style="font-size: 24px;">{forecast_filtered['yhat'].iloc[-1]:.2f}</span>
        </li>
        <li>
            <strong style="color: {custom_metrics_text_color}; font-size: 24px;">Highest Predicted Price:</strong> <span style="font-size: 24px;">{forecast_filtered['yhat'].max():.2f}</span>
        </li>
        <li>
            <strong style="color: {custom_metrics_text_color}; font-size: 24px;">Lowest Predicted Price:</strong> <span style="font-size: 24px;">{forecast_filtered['yhat'].min():.2f}</span>
        </li>
        <li>
            <strong style="color: {custom_metrics_text_color}; font-size: 24px;">Latest Open Price:</strong> <span style="font-size: 24px;">{filtered_df['Open'].iloc[-1] if not filtered_df.empty else None:.2f}</span>
        </li>
        <li>
            <strong style="color: {custom_metrics_text_color}; font-size: 24px;">Latest Volume:</strong> <span style="font-size: 24px;">{filtered_df['Volume'].iloc[-1] if not filtered_df.empty else None}</span>
        </li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)
