import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import streamlit as st

# List of stock tickers
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 
    'BRK-B', 'JPM', 'V', 'JNJ', 'PG', 'UNH', 'HD', 'DIS', 
    'VZ', 'MA', 'CMCSA', 'ADBE', 'PEP', 'T', 'INTC', 
    'CSCO', 'NKE', 'MRK', 'XOM', 'PFE', 'ABT', 'IBM', 'CRM',
    'TCS.NS', 'TRIDENT.NS', 'TATAMOTORS.NS'
]

# Streamlit App
st.title('Stock Market Predictor')

# Dropdown for selecting stock ticker
selected_ticker = st.selectbox('Select Stock Symbol', tickers)

# Timeframe selection
timeframe = st.selectbox('Select Prediction Time Frame', ['1 Day', '1 Week', '1 Month', '1 Year'])

# Define the time period for historical data
start = dt.datetime.today() - dt.timedelta(5 * 365)
end = dt.datetime.today()

# Button to trigger prediction
if st.button('Predict'):
    # Download stock data
    data = yf.download(selected_ticker, start=start, end=end)
    
    st.subheader('Stock Data')
    st.write(data)

    # Ensure data is not empty and has a 'Close' column
    if data.empty or 'Close' not in data.columns:
        st.error("No data available for the selected stock.")
    else:
        # Prepare data for LSTM
        data = data[['Close']].dropna()  # Keep only the 'Close' column and drop NaNs
        if data.empty:
            st.error("Data is empty after dropping NaNs.")
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Create training and testing datasets
            train_data_len = int(np.ceil(0.8 * len(scaled_data)))
            train_data = scaled_data[0:train_data_len]
            x_train, y_train = [], []

            # Prepare training data
            for i in range(100, len(train_data)):
                x_train.append(train_data[i-100:i])
                y_train.append(train_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape data for LSTM
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(60, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(80, return_sequences=True))
            model.add(Dropout(0.4))
            model.add(LSTM(120))
            model.add(Dropout(0.5))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Fit model
            model.fit(x_train, y_train, batch_size=32, epochs=5)

            # Prepare test data
            test_data = scaled_data[train_data_len - 100:]
            x_test = []
            for i in range(100, len(test_data)):
                x_test.append(test_data[i-100:i])

            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Make predictions
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            # Calculate the number of steps based on the selected time frame
            if timeframe == '1 Day':
                future_steps = 1
            elif timeframe == '1 Week':
                future_steps = 5
            elif timeframe == '1 Month':
                future_steps = 30
            elif timeframe == '1 Year':
                future_steps = 252  # Approximate trading days in a year
            
            # Prepare future predictions
            future_predictions = []
            last_input = test_data[-100:]

            for _ in range(future_steps):
                last_input = np.append(last_input[1:], [[predictions[-1]]], axis=0)
                last_input = last_input.reshape((1, 100, 1))
                next_prediction = model.predict(last_input)
                future_predictions.append(next_prediction[0, 0])
                predictions = np.append(predictions, next_prediction)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Plotting actual vs predicted prices
            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label='Actual Price', color='g')
            plt.plot(data.index[train_data_len:], predictions, label='Predicted Price', color='r')
            plt.title(f'{selected_ticker} Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

            # Plot future predictions
            future_dates = pd.date_range(start=data.index[-1] + dt.timedelta(days=1), periods=future_steps, freq='B')
            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_predictions, label='Future Predictions', color='orange')
            plt.title(f'{selected_ticker} Future Price Predictions ({timeframe})')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

            # Plotting Moving Averages
            st.subheader('Moving Averages')
            ma_50 = data['Close'].rolling(50).mean()
            ma_100 = data['Close'].rolling(100).mean()
            ma_200 = data['Close'].rolling(200).mean()

            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label='Close Price', color='g')
            plt.plot(ma_50, label='MA 50', color='r')
            plt.plot(ma_100, label='MA 100', color='b')
            plt.plot(ma_200, label='MA 200', color='purple')
            plt.title(f'{selected_ticker} Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

            # Show predicted price
            predicted_price = predictions[-1][0]
            st.subheader(f'Predicted Price for {selected_ticker}: ${predicted_price:.2f}')
