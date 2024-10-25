import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import yfinance as yf

import yfinance as yf
import datetime as dt
tickers = ['TCS.NS', 'TRIDENT.NS','TATAMOTORS.NS']
start = dt.datetime.today() - dt.timedelta(5 * 365)
end = dt.datetime.today()

for i in tickers:
    data = yf.download(i,start,end)
    data.to_csv(f'{i}.csv', index=True)
    
tcs = pd.read_csv('TCS.NS.csv')
tata = pd.read_csv('TATAMOTORS.NS.csv')
trident = pd.read_csv('TRIDENT.NS.csv')
tcs.head()


tcs.index=tcs['Date']

tcs_df = tcs.sort_index(ascending=True, axis=0)
tcs_new = pd.DataFrame(index=range(0, len(tcs)), columns=['Date', 'Adj Close'])

for i in range(0,len(tcs_df)):
    tcs_new['Date'][i] = tcs_df['Date'][i]
    tcs_new['Adj Close'][i] = tcs_df['Adj Close'][i]

#splitting training and validation dataset
train = tcs_new[:990]
valid = tcs_new[990:]

#shapes of training set
print(train.shape)

#shapes of valid set
print(valid.shape)

preds = []
for i in range(0,valid.shape[0]):
    a = train['Adj Close'][len(train)-246+i:].sum() + sum(preds)
    b = a/246
    preds.append(b)
    
# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Adj Close'])-preds),2)))
print('\nRMSE value on validation set:')
print(rms)

#plot
valid=valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.title('Predicted adj close price vs actual close price on TCS ');

tata.index=tcs['Date']

tata_df = tata.sort_index(ascending=True, axis=0)
tata_new = pd.DataFrame(index=range(0, len(tata)), columns=['Date', 'Adj Close'])

for i in range(0,len(tata_df)):
    tata_new['Date'][i] = tata_df['Date'][i]
    tata_new['Adj Close'][i] = tata_df['Adj Close'][i]

#splitting training and validation dataset
train = tata_new[:990]
valid = tata_new[990:]

#shapes of training set
print(train.shape)

#shapes of valid set
print(valid.shape)

preds = []
for i in range(0,valid.shape[0]):
    a = train['Adj Close'][len(train)-246+i:].sum() + sum(preds)
    b = a/246
    preds.append(b)
    
# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Adj Close'])-preds),2)))
print('\nRMSE value on validation set:')
print(rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.title('Predicted adj close price vs actual close price on Tata Motors ');
trident.index=trident['Date']

trident_df = trident.sort_index(ascending=True, axis=0)
trident_new = pd.DataFrame(index=range(0, len(trident)), columns=['Date', 'Adj Close'])

for i in range(0,len(trident_df)):
    trident_new['Date'][i] = trident_df['Date'][i]
    trident_new['Adj Close'][i] = trident_df['Adj Close'][i]

#splitting training and validation dataset
train = trident_new[:990]
valid = trident_new[990:]

#shapes of training set
print(train.shape)

#shapes of valid set
print(valid.shape)

preds = []
for i in range(0,valid.shape[0]):
    a = train['Adj Close'][len(train)-246+i:].sum() + sum(preds)
    b = a/246
    preds.append(b)
    
# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Adj Close'])-preds),2)))
print('\nRMSE value on validation set:')
print(rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.title('Predicted adj close price vs actual close price on Trident ');


#create features

tcs_new['Date'] = tcs_new['Date'].astype('datetime64[ns]')
tcs_new["Year"] = tcs_new.Date.dt.year
tcs_new["Month"] = tcs_new.Date.dt.month
tcs_new["Day"] = tcs_new.Date.dt.day
tcs_new["DayOfWeek"] = tcs_new.Date.dt.dayofweek
tcs_new["DayOfYear"] = tcs_new.Date.dt.dayofyear

tcs_new.drop('Date', axis=1,inplace=True)

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

#split into train and validation
train = tcs_new[:990]
valid = tcs_new[990:]

x_train = train.drop('Adj Close', axis=1)
y_train = train['Adj Close']
x_valid = valid.drop('Adj Close', axis=1)
y_valid = valid['Adj Close']

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)
print('Model Fitted!')

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMSE is',rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Adj Close', 'Predictions']])
plt.plot(train['Adj Close']);
plt.title('Predicted adj close price vs actual close price on TCS ');



#create features

tata_new['Date'] = tata_new['Date'].astype('datetime64[ns]')
tata_new["Year"] = tata_new.Date.dt.year
tata_new["Month"] = tata_new.Date.dt.month
tata_new["Day"] = tata_new.Date.dt.day
tata_new["DayOfWeek"] = tata_new.Date.dt.dayofweek
tata_new["DayOfYear"] = tata_new.Date.dt.dayofyear

tata_new.drop('Date', axis=1,inplace=True)


#split into train and validation
train = tata_new[:990]
valid = tata_new[990:]

x_train = train.drop('Adj Close', axis=1)
y_train = train['Adj Close']
x_valid = valid.drop('Adj Close', axis=1)
y_valid = valid['Adj Close']

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)
print('Model Fitted!')

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMSE is',rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Adj Close', 'Predictions']])
plt.plot(train['Adj Close']);
plt.title('Predicted adj close price vs actual close price on Tata Motors ');

#create features

trident_new['Date'] = trident_new['Date'].astype('datetime64[ns]')
trident_new["Year"] = trident_new.Date.dt.year
trident_new["Month"] = trident_new.Date.dt.month
trident_new["Day"] = trident_new.Date.dt.day
trident_new["DayOfWeek"] = trident_new.Date.dt.dayofweek
trident_new["DayOfYear"] = trident_new.Date.dt.dayofyear

trident_new.drop('Date', axis=1,inplace=True)



#split into train and validation
train = trident_new[:990]
valid = trident_new[990:]

x_train = train.drop('Adj Close', axis=1)
y_train = train['Adj Close']
x_valid = valid.drop('Adj Close', axis=1)
y_valid = valid['Adj Close']

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)
print('Model Fitted!')

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMSE is',rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Adj Close', 'Predictions']])
plt.plot(train['Adj Close']);
plt.title('Predicted adj close price vs actual close price on Trident ');


#split into train and validation
train = tcs_new[:990]
valid = tcs_new[990:]

x_train = train.drop('Adj Close', axis=1)
y_train = train['Adj Close']
x_valid = valid.drop('Adj Close', axis=1)
y_valid = valid['Adj Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
print('Model Fitted!')

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMSE is',rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = tcs_new[990:].index
train.index = tcs_new[:990].index

plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']]);
plt.title('Predicted adj close price vs actual close price on TCS ');

#split into train and validation
train = tata_new[:990]
valid = tata_new[990:]

x_train = train.drop('Adj Close', axis=1)
y_train = train['Adj Close']
x_valid = valid.drop('Adj Close', axis=1)
y_valid = valid['Adj Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
print('Model Fitted!')

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMSE is',rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = tata_new[990:].index
train.index = tata_new[:990].index

plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']]);
plt.title('Predicted adj close price vs actual close price on Tata Motors ');

#split into train and validation
train = trident_new[:990]
valid = trident_new[990:]

x_train = train.drop('Adj Close', axis=1)
y_train = train['Adj Close']
x_valid = valid.drop('Adj Close', axis=1)
y_valid = valid['Adj Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
print('Model Fitted!')

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMSE is',rms)

#plot
valid = valid.copy()
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = trident_new[990:].index
train.index = trident_new[:990].index

plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']]);
plt.title('Predicted adj close price vs actual close price on Trident ');


#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error
#creating dataframe
tcs_lstm = pd.DataFrame(index=range(0,len(tcs_df)),columns=['Date', 'Adj Close'])
for i in range(0,len(tcs_df)):
    tcs_lstm['Date'][i] = tcs_df['Date'][i]
    tcs_lstm['Adj Close'][i] = tcs_df['Adj Close'][i]

#setting index
tcs_lstm.index = tcs_lstm.Date
tcs_lstm.drop('Date', axis=1, inplace=True)


#creating train and test sets
dataset = tcs_lstm.values

train = dataset[0:990,:]
valid = dataset[990:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
print('Fitting Model')
#predicting 246 values, using past 60 from the train data
inputs = tcs_lstm[len(tcs_lstm) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(mean_squared_error(closing_price,valid))
print('RMSE is',rms)

#for plotting
train = tcs_lstm[:990]
valid = tcs_lstm[990:]
valid['Predictions'] = closing_price
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close','Predictions']]);
plt.title('Predicted adj close price vs actual close price on TCS ');


#creating dataframe
tata_lstm = pd.DataFrame(index=range(0,len(tata_df)),columns=['Date', 'Adj Close'])
for i in range(0,len(tata_df)):
    tata_lstm['Date'][i] = tata_df['Date'][i]
    tata_lstm['Adj Close'][i] = tata_df['Adj Close'][i]

#setting index
tata_lstm.index = tata_lstm.Date
tata_lstm.drop('Date', axis=1, inplace=True)


#creating train and test sets
dataset = tata_lstm.values

train = dataset[0:990,:]
valid = dataset[990:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
print('Fitting Model')
#predicting 246 values, using past 60 from the train data
inputs = tata_lstm[len(tata_lstm) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(mean_squared_error(closing_price,valid))
print('RMSE is',rms)

#for plotting
train = tata_lstm[:990]
valid = tata_lstm[990:]
valid['Predictions'] = closing_price
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close','Predictions']]);
plt.title('Predicted adj close price vs actual close price on Tata Motors ');

#creating dataframe
tri_lstm = pd.DataFrame(index=range(0,len(trident_df)),columns=['Date', 'Adj Close'])
for i in range(0,len(trident_df)):
    tri_lstm['Date'][i] = trident_df['Date'][i]
    tri_lstm['Adj Close'][i] = trident_df['Adj Close'][i]

#setting index
tri_lstm.index = tri_lstm.Date
tri_lstm.drop('Date', axis=1, inplace=True)


#creating train and test sets
dataset = tri_lstm.values

train = dataset[0:990,:]
valid = dataset[990:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
print('Fitting Model')
#predicting 246 values, using past 60 from the train data
inputs = tri_lstm[len(tri_lstm) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(mean_squared_error(closing_price,valid))
print('RMSE is',rms)

#for plotting
train = tri_lstm[:990]
valid = tri_lstm[990:]
valid['Predictions'] = closing_price
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close','Predictions']]);
plt.title('Predicted adj close price vs actual close price on Trident ');


