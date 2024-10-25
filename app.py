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

# Streamlit app
st.title('Stock Price Prediction')

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.selectbox('Select Stock Ticker', ('TCS.NS', 'TATAMOTORS.NS', 'TRIDENT.NS'))

# Button to make predictions
if st.sidebar.button('Predict'):
    # Get stock data
    data = get_stock_data(ticker)

    # Debugging: Display the first few rows of the data
    st.write("First few rows of the downloaded data:")
    st.write(data.head())

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
        for i in range(60, len(train)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        inputs = df[len(df) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        valid['Predictions'] = closing_price
        title = f'Predicted {target_col} vs Actual {target_col} on {ticker} using LSTM'

        # Plot results
        plot_results(pd.DataFrame(train), valid, title)
        
        # Buy/Sell suggestion
        last_actual = valid[target_col].iloc[-1]
        last_predicted = valid['Predictions'].iloc[-1]

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
