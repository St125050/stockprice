import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
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
    plt.plot(train['Adj Close'], label='Train')
    plt.plot(valid[['Adj Close', 'Predictions']], label=['Valid', 'Predictions'])
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

# Streamlit app
st.title('Stock Price Prediction')

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.selectbox('Select Stock Ticker', ('TCS.NS', 'TATAMOTORS.NS', 'TRIDENT.NS'))
model_type = st.sidebar.selectbox('Select Model', ('LSTM', 'KNN', 'Linear Regression'))

# Get stock data
data = get_stock_data(ticker)

# Debugging: Display the first few rows of the data
st.write("First few rows of the downloaded data:")
st.write(data.head())

# Check if 'Adj Close' column exists
if 'Adj Close' not in data.columns:
    st.error(f"'Adj Close' column not found in the data for {ticker}. Please check the data source.")
else:
    # Preprocess data
    df = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Adj Close'])
    for i in range(0, len(data)):
        df['Date'][i] = data.index[i]
        df['Adj Close'][i] = data['Adj Close'][i]
    df.index = df['Date']
    df.drop('Date', axis=1, inplace=True)

    # Split data into train and valid
    dataset = df.values
    train = dataset[0:990, :]
    valid = dataset[990:, :]

    if model_type == 'LSTM':
        # LSTM model
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
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
        
        valid = pd.DataFrame(valid, columns=['Adj Close'])
        valid['Predictions'] = closing_price
        title = f'Predicted adj close price vs actual close price on {ticker} using LSTM'

    elif model_type == 'KNN':
        # KNN model
        df['Date'] = df.index
        df['Year'] = df.Date.dt.year
        df['Month'] = df.Date.dt.month
        df['Day'] = df.Date.dt.day
        df['DayOfWeek'] = df.Date.dt.dayofweek
        df['DayOfYear'] = df.Date.dt.dayofyear
        df.drop('Date', axis=1, inplace=True)
        
        train = df[:990]
        valid = df[990:]
        x_train = train.drop('Adj Close', axis=1)
        y_train = train['Adj Close']
        x_valid = valid.drop('Adj Close', axis=1)
        y_valid = valid['Adj Close']
        
        params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=5)
        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        
        valid = pd.DataFrame(valid, columns=['Adj Close'])
        valid['Predictions'] = preds
        title = f'Predicted adj close price vs actual close price on {ticker} using KNN'

    else:
        # Linear Regression model
        df['Date'] = df.index
        df['Year'] = df.Date.dt.year
        df['Month'] = df.Date.dt.month
        df['Day'] = df.Date.dt.day
        df['DayOfWeek'] = df.Date.dt.dayofweek
        df['DayOfYear'] = df.Date.dt.dayofyear
        df.drop('Date', axis=1, inplace=True)
        
        train = df[:990]
        valid = df[990:]
        x_train = train.drop('Adj Close', axis=1)
        y_train = train['Adj Close']
        x_valid = valid.drop('Adj Close', axis=1)
        y_valid = valid['Adj Close']
        
        model = LinearRegression()
        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        
        valid = pd.DataFrame(valid, columns=['Adj Close'])
        valid['Predictions'] = preds
        title = f'Predicted adj close price vs actual close price on {ticker} using Linear Regression'

    # Plot results
    plot_results(pd.DataFrame(train, columns=['Adj Close']), valid, title)
    st.write(f'RMSE: {np.sqrt(mean_squared_error(valid["Adj Close"], valid["Predictions"]))}')
