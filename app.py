import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import yfinance as yf
import datetime as dt

# Function to download and preprocess data
def get_stock_data(ticker):
    start = dt.datetime.today() - dt.timedelta(5 * 365)
    end = dt.datetime.today()
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    data.set_index('Date', inplace=True)
    return data

# Function to plot results
def plot_results(train, valid, title):
    plt.figure(figsize=(20, 10))
    plt.plot(train['Close'], label='Train')
    plt.plot(valid[['Close', 'Predictions']], label=['Valid', 'Predictions'])
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

# Function to create LSTM model
def create_lstm_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(output_shape))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Streamlit app
st.title('Stock Price Prediction')

# Sidebar for user input
st.sidebar.header('User Input')
stocks = ['TCS.NS', 'TATAMOTORS.NS', 'TRIDENT.NS', 'AAPL', 'MSFT', 'GOOGL', 
          'AMZN', 'TSLA', 'NFLX', 'NVDA', 'BRK-B', 'JPM', 'V', 'JNJ', 'PG', 
          'UNH', 'HD', 'DIS', 'VZ', 'MA', 'CMCSA', 'ADBE', 'PEP', 'T', 
          'INTC', 'CSCO', 'NKE', 'MRK', 'XOM', 'PFE', 'ABT', 'IBM', 'CRM']
ticker = st.sidebar.selectbox('Select Stock Ticker', stocks)
prediction_window = st.sidebar.selectbox('Select Prediction Window', ['1 Day', '1 Week', '1 Month'])

# Button to make predictions
if st.sidebar.button('Predict'):
    # Get stock data
    data = get_stock_data(ticker)

    # Display the first few rows of the data
    st.write("Last few days stock price data:")
    st.write(data.Tail())

    # Use 'Close' column
    target_col = 'Close'

    if target_col not in data.columns:
        st.error(f"'{target_col}' column not found in the data for {ticker}. Please check the data source.")
    else:
        # Preprocess data
        df = data[[target_col]].copy()
        df.reset_index(inplace=True)  # Keep the Date as a column
        df.set_index('Date', inplace=True)

        # Split data into train and valid
        train_size = int(len(df) * 0.8)  # Use 80% of the data for training
        train, valid = df.iloc[:train_size], df.iloc[train_size:]

        # LSTM model
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        x_train, y_train = [], []
        window_size = 60  # Lookback period

        for i in range(window_size, len(train)):
            x_train.append(scaled_data[i-window_size:i, 0])
            if prediction_window == '1 Day':
                y_train.append(scaled_data[i, 0])
            elif prediction_window == '1 Week':
                y_train.append(scaled_data[i:i + 5, 0])  # Next 5 days
            elif prediction_window == '1 Month':
                y_train.append(scaled_data[i:i + 21, 0])  # Next 21 days (approx. 1 month)

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Train the model
        model = create_lstm_model((x_train.shape[1], 1), 1 if prediction_window == '1 Day' else 5 if prediction_window == '1 Week' else 21)
        model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)

        # Prepare test data
        inputs = scaled_data[len(scaled_data) - len(valid) - window_size:]
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        x_test = []
        for i in range(window_size, inputs.shape[0]):
            x_test.append(inputs[i-window_size:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Predictions
        closing_price = model.predict(x_test)
        closing_price = scaler.inverse_transform(closing_price)

        # Add predictions to the valid set
        valid = df[len(train):].copy()
        if prediction_window == '1 Day':
            valid['Predictions'] = closing_price
        elif prediction_window == '1 Week':
            for i in range(len(closing_price)):
                valid.loc[valid.index[i]:valid.index[i + 4], 'Predictions'] = closing_price[i]
        elif prediction_window == '1 Month':
            for i in range(len(closing_price)):
                valid.loc[valid.index[i]:valid.index[i + 20], 'Predictions'] = closing_price[i]

        title = f'Predicted {target_col} vs Actual {target_col} on {ticker} ({prediction_window})'

        # Plot results
        plot_results(pd.DataFrame(train), valid, title)

        # Access last actual and predicted prices
        last_actual = valid[target_col].iloc[-1]
        last_predicted = valid['Predictions'].iloc[-1]

        # Ensure they are scalar values
        if isinstance(last_actual, pd.Series):
            last_actual = last_actual.values[0]
        if isinstance(last_predicted, pd.Series):
            last_predicted = last_predicted.values[0]

        # Suggestion logic
        if last_predicted > last_actual:
            suggestion = "Buy"
        elif last_predicted < last_actual:
            suggestion = "Sell"
        else:
            suggestion = "Hold"

        st.write(f'Last Actual Price: {last_actual:.2f}')
        st.write(f'Last Predicted Price: {last_predicted:.2f}')
        st.write(f'Suggestion: {suggestion}')
        st.write(f'RMSE: {np.sqrt(mean_squared_error(valid[target_col], valid["Predictions"]))}')
