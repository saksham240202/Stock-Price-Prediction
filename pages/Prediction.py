import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN


st.title('Stock Market Prediction')

# INPUT Start and End date
st.sidebar.header('User Input')
start_date = st.sidebar.date_input('Start date', value=date(2020, 1, 1), help="YYYY/MM/DD")
end_date = st.sidebar.date_input('End date', value=datetime.date.today())

ticker_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "TCS", "^NSEI", "RELIANCE.NS"]
ticker = st.sidebar.selectbox('Select Stock', ticker_list)

data = yf.download(ticker, start_date, end_date)
data = data.reset_index()

st.write(start_date, "to", end_date)

column = st.selectbox('Select the column used for forecasting', data.columns[1:])
data = data[['Date', column]]
st.write(data)


def Sarima():
    # Fit the Model
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)
    seasonal_order = st.number_input('Select the seasonal value of p', 0, 24, 12)

    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model = model.fit()

    # Predict the values
    st.write("<p style= 'color:green; font-size:50px; ,font-weight:bold;'>Forcasting the data</p>", unsafe_allow_html=True)
    forecast_period = st.number_input("Select the number of days to forecast", 1, 365, 30)
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
    predictions = predictions.predicted_mean

    # Add Index to results dataframe as dates
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq="D")
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, 'Date', predictions.index)
    predictions.reset_index(drop=True, inplace=True)
    
    # Lets Plot the data
    fig = go.Figure()

    # Add acutal data to the Plot
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Historical Data', line=dict(color='blue')))

    # Add prediction data to the plot
    fig.add_trace(
        go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'], mode='lines', name='Forecast',
                   line=dict(color='red')))

    # Set the title and axis labels
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)

    # Display the Plot
    st.plotly_chart(fig, use_container_width=True, className='st-sc')


def Random_Forest():
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['day_of_month'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year

    # User input 
    days = st.number_input("Enter the number of days to predict", min_value=30, step=1)
    X = data[['day_of_week', 'day_of_month', 'month', 'year']]
    y = data[column]

    # Model Fitting
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Creating future dates
    future_dates = pd.date_range(start=data['Date'].max(), periods=days)

    # Predicting the future
    future_features = pd.DataFrame({
        'day_of_week': future_dates.dayofweek,
        'day_of_month': future_dates.day,
        'month': future_dates.month,
        'year': future_dates.year
    })

    forecast = model.predict(future_features)
    
    rmse = np.sqrt(np.mean((forecast - data[column][-days:]) ** 2))
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')
    # Plotting historical and forecasted data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Historical Data'))

    future_dates = pd.DataFrame({'Date': future_dates, column: forecast})
    fig.add_trace(
        go.Scatter(x=future_dates['Date'], y=future_dates[column], mode='lines', name='Forecast', line=dict(color='orange')))

    fig.update_layout(xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True, className='st-sc')



def FbProphet():
    data1 = data[['Date', column]]
    data1 = data.rename(columns={'Date': 'ds', column: 'y'})

    days = st.number_input("Enter the number of days to predict", min_value=1, step=1, value=30)

    # Fitting the model
    model = Prophet(daily_seasonality=True)
    model.fit(data1)

    # Creating future dates
    future = model.make_future_dataframe(periods=days)

    # Predicting the future
    forecast = model.predict(future)
    rmse = np.sqrt(np.mean((forecast['yhat'].tail(days).values - data[column].tail(days).values) ** 2))
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')

    # Plotting historical and forecasted data 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data1['ds'], y=data1['y'], mode='lines', name='Historical Data'))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='orange')))

    fig.update_layout(xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig, use_container_width=True, className='st-sc')



def LSTM_Model():
    # Normalize the data
    
    days = st.number_input("Enter the number of days to predict", min_value=1, step=1, value=30)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

    # Create training data
    training_data_len = int(np.ceil(len(scaled_data) * .95))

    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create testing data
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = data[column][training_data_len:].values

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    future_date_range = pd.date_range(start=end_date, periods=days)

    # Create the future data for prediction
    future_data = scaled_data[-60:]  # Use the last 60 days as input for prediction

    future_predictions = []
    for i in range(days):
        future_input = np.reshape(future_data, (1, future_data.shape[0], 1))  # Reshape for prediction
        future_output = model.predict(future_input)
        future_data = np.append(future_data, future_output, axis=0)
        future_predictions.append(future_output[0, 0])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Prepare future dates for plotting
    future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=days)

    # Plot the predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Forecast'))

    fig.update_layout(xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True, className='st-sc')


def RNN_Model():
    # Normalize the data
    
    days = st.number_input("Enter the number of days to predict", min_value=1, step=1, value=30)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

    # Create training data
    training_data_len = int(np.ceil(len(scaled_data) * .95))
    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the RNN model
    model = Sequential()
    model.add(SimpleRNN(units=50, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create testing data
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = data[column][training_data_len:].values

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions


 # Get today's date and define the future date range for prediction
    future_date_range = pd.date_range(start=end_date, periods=days)

    # Create the future data for prediction
    future_data = scaled_data[-60:]  # Use the last 60 days as input for prediction

    future_predictions = []
    for i in range(days):
        future_input = np.reshape(future_data, (1, future_data.shape[0], 1))  # Reshape for prediction
        future_output = model.predict(future_input)
        future_data = np.append(future_data, future_output, axis=0)
        future_predictions.append(future_output[0, 0])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Prepare future dates for plotting
    future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=days)

    # Plot the predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Forecast'))

    fig.update_layout(xaxis_title='Date', yaxis_title='Price')                      
    st.plotly_chart(fig, use_container_width=True, className='st-sc')


choice = st.sidebar.selectbox('Select Model', ['Random Forest', 'Fb Prophet', 'Sarima', 'LSTM', 'RNN'])

if choice == 'Random Forest':
    st.write("<div class='full-width'>", unsafe_allow_html=True)
    Random_Forest()
    st.write("</div>", unsafe_allow_html=True)
elif choice == 'Fb Prophet':
    st.write("<div class='full-width'>", unsafe_allow_html=True)
    FbProphet()
    st.write("</div>", unsafe_allow_html=True)
elif choice == 'Sarima':
    st.write("<div class='full-width'>", unsafe_allow_html=True)
    Sarima()
    st.write("</div>", unsafe_allow_html=True)
elif choice == 'LSTM':
    st.write("<div class='full-width'>", unsafe_allow_html=True)
    LSTM_Model()
    st.write("</div>", unsafe_allow_html=True)
elif choice == 'RNN':
    st.write("<div class='full-width'>", unsafe_allow_html=True)
    RNN_Model()
    st.write("</div>", unsafe_allow_html=True)

# CSS for responsiveness
st.markdown(
    """
    <style>
        .full-width {
            width: 100%;
            max-width: 100%;
        }
        .st-iframe {
            width: 100%;
            max-width: 100%;
        }
        .st-df {
            width: 100%;
            max-width: 100%;
        }
        .st-sc {
            width: 100%;
            max-width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)