import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit


def lstm_split(data, target, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps + 1):
        X.append(data[i:i + n_steps, :-1])
        y.append(target[i + n_steps-1])
    return np.array(X), np.array(y)


data_path = '../data/'
picks = os.listdir(data_path)


df = pd.read_pickle(data_path + picks[1])
        
next_day = df['open'].shift(-1)
next_day_rise = (next_day > df['open']).replace({True: 1, False: 0})
target_y = next_day_rise
x_feat = df[['open', 'close', 'high', 'low', 'vol']].values

sc = StandardScaler()
x_feat_sc = sc.fit_transform(x_feat)
x_feat_sc = pd.DataFrame(columns=['open', 'close', 'high', 'low', 'vol'],
                                data = x_feat_sc, index=df['ctmString'])

X1, y1 = lstm_split(x_feat_sc.values, target_y, 30)

train_split = 0.8
split_idx = int(np.ceil(len(X1) * train_split))
date_index = df['ctmString']

X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

print(X1.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]),
activation='relu'))
lstm.add(Dense(1, activation='sigmoid'))
lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
lstm.summary()

history = lstm.fit(X_train, y_train, epochs=50, batch_size=4,
                            verbose=2, shuffle=False)


px.line(history.history['loss'])

lstm.evaluate(x=X_test, y=y_test)
