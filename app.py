import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

# Load the pre-trained LSTM model (make sure you have it saved in your directory)
model = load_model('your_model_path.keras')  # Update this path

# List of stock tickers
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 
    'BRK-B', 'JPM', 'V', 'JNJ', 'PG', 'UNH', 'HD', 'DIS', 
    'VZ', 'MA', 'CMCSA', 'ADBE', 'PEP', 'T', 'INTC', 
    'CSCO', 'NKE', 'MRK', 'XOM', 'PFE', 'ABT', 'IBM', 'CRM',
    'TCS.NS', 'TRIDENT.NS', 'TATAMOTORS.NS'
]

# Streamlit App
st.header('Stock Market Predictor')

# Dropdown for selecting stock ticker
selected_ticker = st.selectbox('Select Stock Symbol', tickers)

# Timeframe options
timeframe = st.selectbox('Select Prediction Time Frame', ['1 Day', '1 Week', '1 Month', '1 Year'])

# Define the time period for historical data
start = dt.datetime.today() - dt.timedelta(5 * 365)
end = dt.datetime.today()

# Download stock data
data = yf.download(selected_ticker, start=start, end=end)

st.subheader('Stock Data')
st.write(data)

# Prepare data for prediction
data = data[['Close']]
data = data.dropna()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare inputs for the model
x_input = scaled_data[-100:].reshape(1, -1, 1)

# Function to make predictions
def make_predictions(model, x_input, timeframe):
    if timeframe == '1 Day':
        predictions = model.predict(x_input)
        predictions = scaler.inverse_transform(predictions)
        return predictions[0][0]
    else:
        future_steps = {'1 Week': 5, '1 Month': 30, '1 Year': 252}
        predictions = []
        
        for _ in range(future_steps[timeframe]):
            predicted_price = model.predict(x_input)
            predictions.append(predicted_price[0][0])
            x_input = np.append(x_input[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
        
        predictions = scaler.inverse_transform(predictions)
        return predictions.flatten()

# Make predictions based on selected timeframe
if st.button('Predict'):
    predicted_price = make_predictions(model, x_input, timeframe)
    
    st.subheader(f'Predicted Price for {selected_ticker} ({timeframe}):')
    st.write(f"${predicted_price:.2f}")

    # Plotting actual vs predicted prices (optional)
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-100:], data['Close'][-100:], label='Actual Price')
    plt.axhline(y=predicted_price, color='r', linestyle='--', label='Predicted Price')
    plt.title(f'{selected_ticker} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)
