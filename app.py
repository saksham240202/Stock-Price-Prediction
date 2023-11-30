import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from datetime import date
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

st.title('Data Visualization', help='This app is created to forecast the stock market price of a selected stock')

# Input Start and End date
st.sidebar.header('User Input')
start_date = st.sidebar.date_input('Start date', value=date(2020, 1, 1), help="YYYY/MM/DD")
end_date = st.sidebar.date_input('End date', value=datetime.date.today())

ticker_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "TCS", "^NSEI", "RELIANCE.NS"]
ticker = st.sidebar.selectbox('Select Stock', ticker_list)

data = yf.download(ticker, start_date, end_date)
data = data.reset_index()

st.write(start_date, "to", end_date)
st.write(data)

st.subheader("Plot the data", help='Select a Column and Zoom in')
fig = px.line(data, x='Date', y=data.columns, width=1000, height=600)
st.plotly_chart(fig, use_container_width=True, className='stPlotlyChart')

column = st.selectbox('Select the column', data.columns[1:])
data = data[['Date', column]]
st.write(data)

# Calculate moving averages
ma100 = data[column].rolling(100).mean()
ma200 = data[column].rolling(200).mean()

fig_ma200 = px.line()
fig_ma200.add_scatter(x=data.index, y=ma100, mode='lines', name='100MA', line=dict(color='red'))
fig_ma200.add_scatter(x=data.index, y=ma200, mode='lines', name='200MA', line=dict(color='green'))
fig_ma200.add_scatter(x=data.index, y=data[column], mode='lines', name=column, line=dict(color='blue'))

fig_ma200.update_layout(width=900, height=400)  # Set width and height
st.plotly_chart(fig_ma200, use_container_width=True, className='stPlotlyChart')

# ADF test Check Stationarity
st.header('Is data Stationary?', help='If p<=0.05 then data is stationary')
st.write(adfuller(data[column])[1] < 0.05)

decomposition = seasonal_decompose(data[column], model='additive', period=12)

st.subheader("Plotting the Decomposition in Plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=900, height=400,
                        labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'),
                use_container_width=True, className='stPlotlyChart')
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=900, height=400,
                        labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'),
                use_container_width=True, className='stPlotlyChart')
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=900, height=400,
                        labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red'),
                use_container_width=True, className='stPlotlyChart')

# CSS for responsiveness
st.markdown(
    """
    <style>
        .stPlotlyChart {
            max-width: 100%;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)
