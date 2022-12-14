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
    get_today,
    generate_clustered_dataset,
    build_lstm_v1,
    train_model,
    get_test_df
)

today = get_today()

with open("../clusters/grouper_" + today + ".pickle", "rb") as f:
    group_dict = pkl.load(f)

picks = glob.glob("../data/*_1440.pickle")
dataset_dict = generate_clustered_dataset(picks, group_dict["Clusters"][0])

print(dataset_dict.keys())

model = build_lstm_v1(
    128, [dataset_dict["X_train"].shape[1], dataset_dict["X_train"].shape[2]]
)

model, history = train_model(
    model,
    dataset_dict["X_train"],
    dataset_dict["X_val"],
    dataset_dict["X_test"],
    dataset_dict["y_train"],
    dataset_dict["y_val"],
    dataset_dict["y_test"],
    epochs=100,
    patience=20,
)

# prev_split = 0
# for split in dataset_dict["Splits"]:
#     X_test = dataset_dict["X_test"][prev_split:split, :, :]
#     y_test = dataset_dict["y_test"][prev_split:split]
#     X_test_date = dataset_dict["X_test_date"][prev_split:split]
#     y_pred = model.predict(x=X_test).reshape(-1)
#     test_df = build_test_df(X_test, y_test, y_pred, X_test_date)
#     plot_stock_pred(test_df, None, history.history, pred_threshold=0.5)
#     input()

for pick in picks:
    if any(symbol in pick for symbol in group_dict["Clusters"][0]):
        test_df, real_df, y_pred = get_test_df(pick, model)
        plot_stock_pred(test_df, real_df, history, pred_threshold=0.5)
        input()
