from datetime import datetime, timedelta
import glob
import pandas as pd
import pytz
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SpatialDropout1D
from keras.callbacks import EarlyStopping
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from ta.trend import MACD
from ta.momentum import StochasticOscillator

TIMEZONE = pytz.timezone("GMT")
INITIAL_TIME = datetime(1970, 1, 1, 00, 00, 00, 000000, tzinfo=TIMEZONE)


def get_today():
    return datetime.now().strftime("%m-%d-%Y")


def xtb_time_to_date(time):
    initial = INITIAL_TIME
    date = initial + timedelta(milliseconds=time)
    return date


def date_to_xtb_time(target):
    target = target.astimezone(TIMEZONE)
    diff = (target - INITIAL_TIME).days * 24 * 3600 * 1000
    return diff


def plot_stock(df, df2=None, params=None, symbols=None):
    config = {
        "scrollZoom": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "html",
            "filename": "custom_image",
            "height": 500,
            "width": 700,
            "scale": 1,
        },
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
    }
    if df2 is None:
        print("None")
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=[0.5, 0.1, 0.2, 0.2],
        )
        fig.add_trace(
            go.Candlestick(
                x=df["ctmString"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbols[0],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA5"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="MA 5",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="MA 20",
            )
        )
        MACD1, STOCH1 = get_macd_stoch(df)
        fig.add_trace(go.Bar(x=df.index, y=df["vol"]), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=MACD1.macd_diff()), row=3, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=MACD1.macd(), line=dict(color="black", width=2)),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=MACD1.macd_signal(), line=dict(color="blue", width=1)
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=STOCH1.stoch(), line=dict(color="black", width=2)),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=STOCH1.stoch_signal(), line=dict(color="blue", width=1)
            ),
            row=4,
            col=1,
        )
    else:
        fig = make_subplots(
            rows=4,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=[0.5, 0.1, 0.2, 0.2],
            column_widths=[0.5, 0.5],
        )
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbols[0],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Candlestick(
                x=df2.index,
                open=df2["open"],
                high=df2["high"],
                low=df2["low"],
                close=df2["close"],
                name=symbols[1],
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA5"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="MA 5",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="MA 20",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df2.index,
                y=df2["MA5"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="MA 5",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df2.index,
                y=df2["MA20"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="MA 20",
            ),
            row=1,
            col=2,
        )
        MACD1, STOCH1 = get_macd_stoch(df)
        fig.add_trace(go.Bar(x=df.index, y=df["vol"]), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=MACD1.macd_diff()), row=3, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=MACD1.macd(), line=dict(color="black", width=2)),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=MACD1.macd_signal(), line=dict(color="blue", width=1)
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=STOCH1.stoch(), line=dict(color="black", width=2)),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=STOCH1.stoch_signal(), line=dict(color="blue", width=1)
            ),
            row=4,
            col=1,
        )

        MACD2, STOCH2 = get_macd_stoch(df2)
        fig.add_trace(go.Bar(x=df2.index, y=df2["vol"]), row=2, col=2)
        fig.add_trace(go.Bar(x=df2.index, y=MACD2.macd_diff()), row=3, col=2)
        fig.add_trace(
            go.Scatter(x=df2.index, y=MACD2.macd(), line=dict(color="black", width=2)),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=MACD2.macd_signal(), line=dict(color="blue", width=1)
            ),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=STOCH2.stoch(), line=dict(color="black", width=2)
            ),
            row=4,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=STOCH2.stoch_signal(), line=dict(color="blue", width=1)
            ),
            row=4,
            col=2,
        )

    if params:
        fig.add_hline(
            y=params["buy_price"],
            name="BP",
            line_dash="dash",
            line_color="yellow",
            line_width=3,
            row=1,
            col=1,
        )
        fig.add_hline(
            y=params["stop_loss"],
            name="SL",
            line_dash="dash",
            line_color="red",
            line_width=3,
            row=1,
            col=1,
        )
        fig.add_hline(
            y=params["take_profit"],
            name="TP",
            line_dash="dash",
            line_color="green",
            line_width=3,
            row=1,
            col=1,
        )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    # fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.show()


def plot_stock_pred(df, real_df, history, pred_threshold=0.5):
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {"secondary_y": True}],
        ],
        row_heights=[0.3, 0.9],
        vertical_spacing=0.3,
    )
    if real_df is not None:
        fig.add_trace(
            go.Candlestick(
                x=real_df["ctmString"],
                open=real_df["open"],
                high=real_df["high"],
                low=real_df["low"],
                close=real_df["close"],
            ),
            col=1,
            row=1,
            secondary_y=False,
        )
        fig.add_vrect(
            x0=df.index[0],
            x1=df.index[-1],
            fillcolor="#2ca02c",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        ),
        col=2,
        row=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["y_test"],
            mode="markers",
            name="Real",
            marker=dict(
                symbol="circle",
                size=8,
            ),
        ),
        col=2,
        row=2,
        secondary_y=True,
        # secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["y_pred"],
            mode="markers",
            name="Preds-probs",
            marker=dict(
                symbol="cross",
                size=5,
            ),
        ),
        col=2,
        row=2,
        secondary_y=True,
        # secondary_y=True,
    )
    # model = KMeans(n_clusters=2)
    # df["y_pred"][:10] = 0
    # model.fit(df["y_pred"].values.reshape(-1, 1))
    # preds = model.predict(df["y_pred"].values.reshape(-1, 1))
    # df["y_pred"] = np.abs(preds - 1)
    # df["y_pred"][:10] = -1
    df[df["y_pred"] > pred_threshold] = 1
    df[df["y_pred"] < pred_threshold] = 0
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["y_pred"],
            mode="markers",
            name="Pred-labels",
            marker=dict(
                symbol="cross",
                size=5,
            ),
        ),
        col=2,
        row=2,
        secondary_y=True,
        # secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history["loss"]))),
            y=history["loss"],
            name="Train Loss",
        ),
        col=1,
        row=2,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history["val_loss"]))),
            y=history["val_loss"],
            name="Val Loss",
        ),
        col=1,
        row=2,
    )
    fig.update_layout(
        yaxis2=dict(
            range=[
                df[["open", "close", "high", "low"]].max().max(),
                df[["open", "close", "high", "low"]].min().min(),
            ]
        )
    )
    fig.show()


def adapt_data(df, pred_step=1):
    df["close"] = df["open"] + df["close"]
    df["high"] = df["open"] + df["high"]
    df["low"] = df["open"] + df["low"]
    df = df.set_index(pd.to_datetime(df["ctmString"]))

    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA5"] = df["close"].rolling(window=5).mean()

    df = add_supports(df)

    df = generate_target(df, pred_step)

    df = df.dropna()

    return df


def lstm_split(data, target, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps + 1):
        X.append(data[i : i + n_steps, :])
        y.append(target[i + n_steps - 1])
    return np.array(X), np.array(y)


def lstm_accumulative_split(data, target, min_step=30):
    X, y = [], []
    template = np.zeros_like(data)
    for i in range(min_step, len(data) - 1):
        template[-i:, :] = data[0:i, :]
        X.append(template)
        y.append(target[i])
    return np.array(X), np.array(y)


def print_stats(symbol, loss, metric_name, metric):
    print("*******************************************")
    print(f"{symbol} | loss: {loss} | {metric_name}: {metric}")
    print("*******************************************")


def build_test_df(X_test, y_test, preds, X_test_date):
    full_test = X_test[0, :, :]
    full_target = np.empty((X_test.shape[1], 1))
    full_target[:] = np.nan
    full_pred = np.empty((X_test.shape[1], 1))
    full_pred[:] = np.nan
    for i in range(1, X_test.shape[0]):
        full_test = np.append(full_test, X_test[i, -1, :].reshape(1, -1), axis=0)
        full_target = np.append(full_target, np.array(y_test[i]).reshape(1, 1), axis=0)
        full_pred = np.append(full_pred, np.array(preds[i]).reshape(1, 1), axis=0)

    full_test = np.concatenate((full_test, full_target, full_pred), axis=1)
    full_test = full_test[-X_test_date.shape[0] :, :]
    test_df = pd.DataFrame(
        columns=[
            "open",
            "close",
            "high",
            "low",
            "vol",
            "MA20",
            "MA5",
            "supports",
            "y_test",
            "y_pred",
        ],
        data=full_test,
        index=X_test_date,
    )
    test_df = test_df.dropna()
    return test_df


def show_clusters(group_n, symbol_dict, counts):
    for i in range(group_n):
        grid_side = int(np.ceil(np.sqrt(counts[i])))
        fig = make_subplots(rows=grid_side + 1, cols=grid_side + 1)
        cont_col = 0
        cont_row = 0
        for key in symbol_dict.keys():
            if symbol_dict[key] == i:
                cont_col += 1
                pick = glob.glob(f"../data/{key}*_1440.pickle")[0]
                df = adapt_data(pd.read_pickle(pick))
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["close"], mode="lines", name=key),
                    col=cont_col % grid_side + 1,
                    row=cont_row // grid_side + 1,
                )
                cont_row += 1
        fig.show()
        input()


def show_heatmap(df, xs, ys):
    fig = go.Figure(go.Heatmap(z=df, x=xs, y=ys))
    fig.show()
    input()


def add_supports(df):
    supports = np.zeros_like(df["close"], dtype=int)
    for day in range(1, df.shape[0]):
        current_value = df["close"][day]
        past_df = df.iloc[:day, :]
        temp_supports = past_df[current_value * 1.05 > past_df["close"]]
        temp_supports = temp_supports[current_value * 0.95 < temp_supports["close"]][
            "close"
        ]
        supports[day] = temp_supports.shape[0]

    df["supports"] = supports
    return df


def generate_target(df, pred_step):
    targets = np.zeros_like(df["close"], dtype=int)
    for day in range(df.shape[0] - pred_step):
        target = 0
        for preds in range(pred_step):
            if df["close"][day] < df["close"][day + preds]:
                target += 1
        if target == pred_step - 1:
            targets[day] = 1
    df["target"] = targets
    return df


def generate_clustered_dataset(picks, group, steps, pred_step=1):
    (
        splits,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_date1,
        X_val_date1,
        X_test_date1,
    ) = (None, None, None, None, None, None, None, None, None, None)
    for pick in picks:
        if any(symbol in pick for symbol in group):
            df = adapt_data(pd.read_pickle(pick), pred_step=pred_step)
            df = df.set_index(df["ctmString"])
            x_feat = df.iloc[:, 2:-1].values
            sc = StandardScaler()
            x_feat_sc = sc.fit_transform(x_feat)
            x_feat_sc = pd.DataFrame(
                columns=df.iloc[:, 2:-1].columns,
                data=x_feat_sc,
                index=df["ctmString"],
            )
            X1, y1 = lstm_split(
                x_feat_sc.values,
                df["target"],
                n_steps=steps,
            )
            train_split = 0.8
            val_split = 0.1
            train_split_idx = int(np.ceil(len(X1) * train_split))
            val_split_idx = int(np.ceil(len(X1) * (train_split + val_split)))
            date_index = df["ctmString"]

            X_train1, X_val1, X_test1 = (
                X1[:train_split_idx],
                X1[train_split_idx:val_split_idx],
                X1[val_split_idx:],
            )
            y_train1, y_val1, y_test1 = (
                y1[:train_split_idx],
                y1[train_split_idx:val_split_idx],
                y1[val_split_idx:],
            )
            X_train_date1, X_val_date1, X_test_date1 = (
                date_index[:train_split_idx],
                date_index[train_split_idx:val_split_idx],
                date_index[val_split_idx:],
            )

            if X_train is None:
                splits = [X_train1.shape[0]]
                X_train = X_train1
                X_val = X_val1
                X_test = X_test1
                y_train = y_train1
                y_val = y_val1
                y_test = y_test1
                X_train_date = X_train_date1
                X_val_date = X_val_date1
                X_test_date = X_test_date1
            else:
                splits.append(X_train.shape[0])
                X_train = np.append(X_train, X_train1, axis=0)
                X_val = np.append(X_val, X_val1, axis=0)
                X_test = np.append(X_test, X_test1, axis=0)
                y_train = np.append(y_train, y_train1, axis=0)
                y_val = np.append(y_val, y_val1, axis=0)
                y_test = np.append(y_test, y_test1, axis=0)
                X_train_date = np.append(X_train_date, X_train_date1, axis=0)
                X_val_date = np.append(X_val_date, X_val_date1, axis=0)
                X_test_date = np.append(X_test_date, X_test_date1, axis=0)
    return {
        "Splits": splits,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_date": X_train_date,
        "X_val_date": X_val_date,
        "X_test_date": X_test_date,
    }, sc


def build_lstm_v1(n_lstm, shape, show=False):
    n_lstm = 10

    model = Sequential()
    model.add(LSTM(n_lstm, input_shape=(shape[0], shape[1]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    if show:
        model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def train_model(
    model, X_train, X_val, X_test, y_train, y_val, y_test, epochs=100, patience=20
):
    num_epochs = epochs
    early_stop = EarlyStopping(monitor="val_loss", patience=patience)

    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=2,
    )

    train_results = model.evaluate(X_train, y_train, verbose=2, batch_size=16)
    val_results = model.evaluate(X_val, y_val, verbose=2, batch_size=16)
    test_results = model.evaluate(X_test, y_test, verbose=2, batch_size=16)
    print(f"Train accuracy: {train_results[1]*100:0.2f}")
    print(f"Val accuracy: {val_results[1]*100:0.2f}")
    print(f"Test accuracy: {test_results[1]*100:0.2f}")

    return model, history.history, test_results[1]


def get_test_df(pick, model, sc, steps):
    df = adapt_data(pd.read_pickle(pick))
    x_feat = df.iloc[:, 2:-1].values
    x_feat_sc = sc.fit_transform(x_feat)
    x_feat_sc = pd.DataFrame(
        columns=df.iloc[:, 2:-1].columns,
        data=x_feat_sc,
        index=df["ctmString"],
    )
    X1, y1 = lstm_split(
        x_feat_sc.values,
        df["target"],
        n_steps=steps,
    )
    train_split = 0.8
    val_split = 0.1
    val_split_idx = int(np.ceil(len(X1) * (train_split + val_split)))
    date_index = df["ctmString"]

    X_test = X1[val_split_idx:]
    y_test = y1[val_split_idx:]
    X_test_date = date_index[val_split_idx:]

    y_pred = model.predict(x=X_test).reshape(-1)
    test_df = build_test_df(X_test, y_test, y_pred, X_test_date)

    return test_df, df, y_pred


def calculate_stop_loss(df, step=1, short=False):
    if short:
        return df["high"][-step:].max()
    else:
        return df["low"][-step:].min()


def calculate_take_profit(buy_price, stop_loss, short=False):
    if short:
        return buy_price + 1.5 * (buy_price - stop_loss)
    else:
        return buy_price + 1.5 * (buy_price - stop_loss)


def find_maximum_correlation_shift(df1, df2, max_shift=50):
    rs = []
    symbol_diff = df1.shape[0] - df2.shape[0]
    if symbol_diff < 0:
        df2 = df2.iloc[np.abs(symbol_diff) :, :]
    else:
        df1 = df1.iloc[np.abs(symbol_diff) :, :]
    r, p = pearsonr(df1["close"].values, df2["close"].values)
    rs.append(r)
    for shift in range(1, max_shift):
        sv_df1 = df1["close"].shift(-shift).values[:-shift]
        sv_df2 = df2["close"].values[shift:]
        try:
            r, p = pearsonr(sv_df1, sv_df2)
            rs.append(r)
        except ValueError as e:
            rs.append(0)

    return np.argmax(np.abs(rs)), np.max(np.abs(rs)), np.abs(rs[0])


def get_last_step(df, sc, step):
    x_feat = df.iloc[:, 2:-1].values
    x_feat_sc = sc.fit_transform(x_feat)
    return x_feat_sc[-step:, :]


def get_macd_stoch(df):
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    STOCH = StochasticOscillator(
        high=df["high"], close=df["close"], low=df["low"], window=14, smooth_window=3
    )
    return macd, STOCH


# example function for processing ticks from Streaming socket
def procTickExample(msg):
    print("TICK: ", msg)


# example function for processing trades from Streaming socket
def procTradeExample(msg):
    print("TRADE: ", msg)


# example function for processing trades from Streaming socket
def procBalanceExample(msg):
    print("BALANCE: ", msg)


# example function for processing trades from Streaming socket
def procTradeStatusExample(msg):
    print("TRADE STATUS: ", msg)


# example function for processing trades from Streaming socket
def procProfitExample(msg):
    print("PROFIT: ", msg)


# example function for processing news from Streaming socket
def procNewsExample(msg):
    print("NEWS: ", msg)


# Command templates
def baseCommand(commandName, arguments=None):
    if arguments == None:
        arguments = dict()
    return dict([("command", commandName), ("arguments", arguments)])


def loginCommand(userId, password, appName=""):
    return baseCommand("login", dict(userId=userId, password=password, appName=appName))
