from datetime import datetime, timedelta
import pandas as pd
import pytz
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TIMEZONE = pytz.timezone("GMT")
INITIAL_TIME = datetime(1970, 1, 1, 00, 00, 00, 000000, tzinfo=TIMEZONE)


def xtb_time_to_date(time):
    initial = INITIAL_TIME
    date = initial + timedelta(milliseconds=time)
    return date


def date_to_xtb_time(target):
    target = target.astimezone(TIMEZONE)
    diff = (target - INITIAL_TIME).days * 24 * 3600 * 1000
    return diff


def plot_stock(df, next_day_rise, test_split=0.8):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Candlestick(
            x=df["ctmString"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        ),
        secondary_y=False,
    )
    fig.add_vrect(
        x0=df["ctmString"][int(df.shape[0] * 0.8)],
        x1=df["ctmString"].iat[-1],
        fillcolor="#2ca02c",
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    fig.add_trace(
        go.Scatter(x=df["ctmString"], y=next_day_rise, mode="markers", name="Rises"),
        secondary_y=True,
    )
    fig.show()
    input()


def plot_stock_pred(df, real_df, history, pred_threshold=0.5):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'colspan': 2}, None],
                [{}, {"secondary_y": True}],
            ],
            row_heights=[0.3, 0.9],
            vertical_spacing=0.3   
    )
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
    df[df['y_pred'] > pred_threshold] = 1
    df[df['y_pred'] < pred_threshold] = 0
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
            x=list(range(len(history['loss']))), y=history['loss'],
            name='Train Loss',
        ),
        col=1,
        row=2
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history['val_loss']))), y=history['val_loss'],
            name='Val Loss',
        ),
        col=1,
        row=2
    )
    fig.update_layout(yaxis2=dict(range=[df[['open', 'close', 'high', 'low']].max().max(), df[['open', 'close', 'high', 'low']].min().min()]))
    # fig.update_layout(yaxis3=dict(range=[df[['open', 'close', 'high', 'low']].max().max(), df[['open', 'close', 'high', 'low']].min().min()]))
    fig.show()


def adapt_data(df):
    """adapt_data This function removes the relative values of the candles
    and makes them independent

    Args:
        df (_type_): dataframe with columns ['ctm', 'ctmString',
        'open', 'close', 'high', 'low', 'vol']
    """
    df["close"] = df["open"] + df["close"]
    df["high"] = df["open"] + df["high"]
    df["low"] = df["open"] + df["low"]
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
        template[-i:, :] = data[0 : i, :]
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
    test_df = pd.DataFrame(
        columns=["open", "close", "high", "low", "y_test", "y_pred"],
        data=full_test,
        index=X_test_date,
    )
    return test_df