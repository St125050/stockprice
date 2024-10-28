import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

# Load the Keras model
model = load_model('stock_prediction_model.keras')  # Ensure this is the correct path to your model

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
stock = st.selectbox('Select Stock Symbol', tickers)
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start=start, end=end)

st.subheader('Stock Data')
st.write(data)

# Prepare data for prediction
data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x_test, y_test = [], []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i - 100:i])
    y_test.append(data_test_scale[i, 0])

x_test = np.array(x_test)

# Predict
y_predict = model.predict(x_test)
scale = 1 / scaler.scale_[0]
y_predict = y_predict * scale
y_test = np.array(y_test) * scale

# Plotting results
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y_predict, 'r', label='Predicted Price')
plt.plot(y_test, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Moving averages
st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA 50')
plt.plot(data['Close'], 'g', label='Close Price')
plt.title('Price vs MA50')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA 50')
plt.plot(ma_100_days, 'b', label='MA 100')
plt.plot(data['Close'], 'g', label='Close Price')
plt.title('Price vs MA50 vs MA100')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA 100')
plt.plot(ma_200_days, 'b', label='MA 200')
plt.plot(data['Close'], 'g', label='Close Price')
plt.title('Price vs MA100 vs MA200')
plt.legend()
st.pyplot(fig3)
