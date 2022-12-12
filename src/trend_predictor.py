import glob
import pickle as pkl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import plotly.express as px
from tensorflow import keras
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SpatialDropout1D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    accuracy_score,
)
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from utils import (
    adapt_data,
    plot_stock,
    plot_stock_pred,
    lstm_split,
    lstm_accumulative_split,
    print_stats,
    build_test_df,
)


picks = glob.glob("../data/*_1440.pickle")
predictable_symbols = []

for pick in picks:
    df = adapt_data(pd.read_pickle(pick))
    symbol = pick.split("_")[0].split("\\")[-1]

    df["target"] = (df["close"].shift(-10) > df['open']).replace({True: 1, False: 0})
    df.set_index(df['ctmString'])

    rises_n =  df[df['target'] == 1].shape[0]
    lowers_n =  df[df['target'] == 0].shape[0]
    print(f"Rises: {rises_n} | Lowers: {lowers_n}")

    x_feat = df[["open", "close", "high", "low"]].values

    sc = StandardScaler()
    x_feat_sc = sc.fit_transform(x_feat)
    x_feat_sc = pd.DataFrame(
        columns=["open", "close", "high", "low"],
        data=x_feat_sc,
        index=df["ctmString"],
    )

    X1, y1 = lstm_split(
        x_feat_sc[["open", "close", "high", "low"]].values,
        df["target"],
        n_steps=10,
    )

    train_split = 0.8
    split_idx = int(np.ceil(len(X1) * train_split))
    date_index = df["ctmString"]

    X_train, X_test = X1[:split_idx], X1[split_idx:]
    y_train, y_test = y1[:split_idx], y1[split_idx:]
    X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]
    print(X1.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    n_lstm = 128
    drop_lstm = 0.2

    model = Sequential()
    # model.add(SpatialDropout1D(drop_lstm))
    model.add(LSTM(n_lstm, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(Dropout(drop_lstm))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    

    num_epochs = 30
    early_stop = EarlyStopping(monitor="val_loss", patience=10)

    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=2,
    )

    train_dense_results = model.evaluate(X_train, y_train, verbose=2, batch_size=16)
    valid_dense_results = model.evaluate(X_test, y_test, verbose=2, batch_size=16)
    print(f'Train accuracy: {train_dense_results[1]*100:0.2f}')
    print(f'Valid accuracy: {valid_dense_results[1]*100:0.2f}')

    # loss = model.evaluate(x=X_test, y=y_test)
    y_pred = model.predict(x=X_test).reshape(-1)

    # acc = accuracy_score(y_test, y_pred)

    # SHOW RESULTS
    test_df = build_test_df(X_test, y_test, y_pred, X_test_date)
    plot_stock_pred(test_df, df, history.history, pred_threshold=0.5)
    input()
