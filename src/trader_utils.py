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
from scipy.signal import argrelextrema
import json

TIMEZONE = pytz.timezone("GMT")
INITIAL_TIME = datetime(1970, 1, 1, 00, 00, 00, 000000, tzinfo=TIMEZONE)

# DATE MANAGING
def get_today():
    return datetime.now().strftime("%m-%d-%Y")


def get_today_ms():
    timestamp = datetime.now()
    t_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    t_int = int(timestamp.strftime("%Y%m%d%H%M%S"))
    return t_str, t_int


def xtb_time_to_date(time):
    initial = INITIAL_TIME
    date = initial + timedelta(milliseconds=time)
    return date


def date_to_xtb_time(target):
    target = target.astimezone(TIMEZONE)
    diff = (target - INITIAL_TIME).days * 24 * 3600 * 1000
    return diff


# STOCK DATA RELATED
def adapt_data(df):
    df["close"] = df["open"] + df["close"]
    df["high"] = df["open"] + df["high"]
    df["low"] = df["open"] + df["low"]
    df = df.set_index(pd.to_datetime(df["ctmString"]))

    df = df.dropna()
    df = add_supports(df)

    return df


def cast_candles_to_types(df, digits):
    if df is not None:
        df["ctm"] = pd.to_numeric(df["ctm"])
        df["ctmString"] = pd.to_datetime(df["ctmString"])
        df["open"] = pd.to_numeric(df["open"])
        df["close"] = pd.to_numeric(df["close"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["vol"] = pd.to_numeric(df["vol"])
        df = df.set_index(df["ctmString"])
        df = numbers_to_decimal(df, digits)
    return df


def numbers_to_decimal(df, digits):
    for column in ["open", "close", "high", "low", "vol"]:
        df[column] = df[column] / np.power(10, digits)
    return df


# Stock intelligence
def add_supports(df, margin=5):
    df["min"] = df.iloc[argrelextrema(df.close.values, np.less_equal, order=margin)[0]][
        "close"
    ]
    df["max"] = df.iloc[
        argrelextrema(df.close.values, np.greater_equal, order=margin)[0]
    ]["close"]
    return df


def add_rolling_means(df, short, long):
    df["MA_short"] = df["close"].rolling(window=short).mean()
    df["MA_long"] = df["close"].rolling(window=long).mean()
    return df


# API RELATED
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


# Login command for xtb
def loginCommand(userId, password, appName=""):
    return baseCommand("login", dict(userId=userId, password=password, appName=appName))


# Convert return data into pandas dataframe
def return_as_df(returnData):
    if len(returnData) != 0:
        columns = returnData[0].keys()
        df_dict = {}
        for col in columns:
            df_dict[col] = []
        for r_dict in returnData:
            for col in columns:
                df_dict[col].append(r_dict[col])
        df = pd.DataFrame.from_dict(df_dict)
        return df
    else:
        return None


# TRADING SYSTEM RELATED
def back_test(df, period, capital, symbol, short, long):
    df = add_rolling_means(df, short=short, long=long)
    plot_stock(df, symbol=symbol)
    transactions = []
    is_bough = False
    profits = []
    potential_profits = [0]
    print(f"Start test")
    for step in range(df.shape[0] - period, df.shape[0]):
        # print(f"Step {step}")
        prev_tick = df.iloc[step - 1, :]
        tick = df.iloc[step, :]
        tick_day = str(tick["ctmString"])

        if is_bough:
            diff = prev_tick["low"] - tick["low"]
            profit = tick["low"] - buy_price
            if profit < 0:
                transactions.append(
                    {"type": "sell", "price": tick["low"], "time": tick_day}
                )
                is_bough = False
                profits.append(profit)
                potential_profits = [0]
                sell_price = tick["low"]
                print(f"Sold at s:{tick_day} | sp: {sell_price} | profit: {profit}")
            elif profit > 0:
                potential_profits.append(diff)
                if prev_tick["low"] > tick["low"] * 1.05:
                    transactions.append(
                        {"type": "sell", "price": tick["low"], "time": tick_day}
                    )
                    is_bough = False
                    profits.append(profit)
                    potential_profits = [0]
                    sell_price = tick["low"]
                    print(f"Sold at s:{tick_day} | sp: {sell_price} | profit: {profit}")
        if (
            (prev_tick["MA_short"] < prev_tick["MA_long"])
            and (tick["MA_short"] > tick["MA_long"])
            or (prev_tick["MA_short"] > prev_tick["MA_long"])
            and (tick["MA_short"] < tick["MA_long"])
        ):
            print(f"CROSSOVER at {tick_day}")
            balance = np.sum(profits) + capital
            c_price = tick["high"]
            print(f"    Balance: {balance}")
            print(f"    Price: {c_price}")

        if (prev_tick["MA_short"] < prev_tick["MA_long"]) and (
            tick["MA_short"] > tick["MA_long"]
        ):
            print(f"Short term upward crossover")
            if np.sum(profits) + capital > tick["high"]:
                transactions.append(
                    {"type": "buy", "price": tick["high"], "time": tick_day}
                )
                is_bough = True
                buy_price = tick["high"]
                print(f"Bought at s:{tick_day} | bp: {buy_price}")
            else:
                print(f"Not enough money to buy")

        if (prev_tick["MA_short"] > prev_tick["MA_long"]) and (
            tick["MA_short"] < tick["MA_long"]
        ):
            print(f"Short term downward crossover")
            if is_bough:
                transactions.append(
                    {"type": "sell", "price": tick["low"], "time": tick_day}
                )
                is_bough = False
                potential_profits = [0]
                sell_price = tick["low"]
                profit = buy_price - tick["low"]
                profits.append(profit)
                print(f"Sold at s:{tick_day} | sp: {sell_price} | profit: {profit}")

    # FINISHED BACKTEST
    if len(profits) > 0:
        np_profits = np.array(profits)
        lossing_trades = np_profits[np_profits < 0]
        winning_trades = np_profits[np_profits > 0]
        print(f"Profits: {np.nansum(profits)}")
        print(f"Winning/Lossing trades: {len(winning_trades)}|{len(lossing_trades)}")
        print(
            f"Average transaction profit: w:{np.nanmean(winning_trades)} | l: {np.nanmean(lossing_trades)}"
        )
        # print(f"TRANSACTIONS: ")
        # print(json.dumps(transactions, indent=4, sort_keys=True))
    else:
        print(f"No trades made")


def test_trend(df, period):
    MA = df.rolling(period).mean()
    trend = []
    for step in range(period):
        if MA[-1] > MA[-step]:
            trend.append(1)
        else:
            trend.append(0)
    trend = np.array(trend)
    ratio = len(trend[trend == 1]) / len(trend[trend == 0])
    if ratio == 1:
        return 1  # "UPWARD"
    elif ratio == 0:
        return 0  # "DOWNWARD"
    else:
        return ratio  # "UPWARD"


# PLOTTING FUNCTIONS
def plot_stock(df, params=None, symbol=None):
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
            name=symbol,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA_short"],
            opacity=0.7,
            line=dict(color="blue", width=2),
            name="MA 5",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA_long"],
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
        go.Scatter(x=df.index, y=MACD1.macd_signal(), line=dict(color="blue", width=1)),
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


def get_macd_stoch(df):
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    STOCH = StochasticOscillator(
        high=df["high"], close=df["close"], low=df["low"], window=14, smooth_window=3
    )
    return macd, STOCH
