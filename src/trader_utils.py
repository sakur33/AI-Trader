from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, DensityMixin
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
from sklearn.model_selection import GridSearchCV
import json
import warnings

warnings.filterwarnings(action="ignore", message="Mean of empty slice")

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
    return date.strftime("%Y-%m-%d %H:%M:%S.%f")


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
        # Dec 4, 2022, 11:00:00 PM
        df["ctmString"] = pd.to_datetime(
            df["ctmString"], format="%b %d, %Y, %I:%M:%S %p"
        )
        df["ctmString"] = df["ctmString"].dt.strftime("%m/%d/%Y %H:%M:%S")
        df["ctmString"] = pd.to_datetime(df["ctmString"])
        df["open"] = pd.to_numeric(df["open"])
        df["close"] = pd.to_numeric(df["close"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["vol"] = pd.to_numeric(df["vol"])
        df = df.set_index(df["ctmString"])
        if digits:
            df = numbers_to_decimal(df, digits)
    return df


def cast_ticks_to_types(df):
    if df is not None:
        df["ask"] = pd.to_numeric(df["ask"])
        df["askVolume"] = pd.to_numeric(df["askVolume"], downcast="integer")
        df["bid"] = pd.to_numeric(df["bid"])
        df["bidVolume"] = pd.to_numeric(df["bidVolume"], downcast="integer")
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["level"] = pd.to_numeric(df["level"], downcast="integer")
        df["spreadRaw"] = pd.to_numeric(df["spreadRaw"])
        df["spreadTable"] = pd.to_numeric(df["spreadTable"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
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
def back_test(
    df,
    period,
    capital,
    symbol,
    short_ma,
    long_ma,
    trend,
    short_enabled,
    fig=None,
    show=False,
):
    df = add_rolling_means(df, short=short_ma, long=long_ma)
    if show:
        fig = plot_stock(df, symbol=symbol, return_fig=True, fig=None)
    if show:
        fig = plot_stock(df, symbol=symbol, return_fig=True, fig=None)
    start_date = df["ctmString"][-period].strftime("%Y-%m-%d")
    transactions = []
    is_bought = False
    is_short = False
    profits = []
    potential_profits = [0]
    if show:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("")
        print("")
        print(f"Start test from {start_date}")
        print(f"    Short selling: {short_enabled}")
        print(f"    Capital at risk: {capital}")
        print(f"    Trend: {trend}")
    if show:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("")
        print("")
        print(f"Start test from {start_date}")
        print(f"    Short selling: {short_enabled}")
        print(f"    Capital at risk: {capital}")
        print(f"    Trend: {trend}")

    for step in range(df.shape[0] - period, df.shape[0]):
        # print(f"Step {step}")
        start_date = df["ctmString"][-period].strftime("%Y-%m-%d %H:%M:%S")
        prev_tick = df.iloc[step - 1, :]
        tick = df.iloc[step, :]
        tick_day = str(tick["ctmString"])
        if is_bought:
            diff = tick["low"] - prev_tick["low"]
            profit = tick["low"] - buy_price
            if show:
                print(f"Tick: {start_date}")
                print(f"Sell Price {tick['low']}")
                print(f"Buy price {buy_price}")
                print(f"Profit {profit}")
                print(f"Diff {diff}")
                print(f"Potential profits {potential_profits}")
            if show:
                print(f"Tick: {start_date}")
                print(f"Sell Price {tick['low']}")
                print(f"Buy price {buy_price}")
                print(f"Profit {profit}")
                print(f"Diff {diff}")
                print(f"Potential profits {potential_profits}")

            if is_short:
                profit = -1 * profit
            if profit < -(buy_price * 0.05):
                (
                    transactions,
                    is_bought,
                    is_short,
                    profits,
                    potential_profits,
                    fig,
                ) = sell_position(
                    transactions, tick, tick_day, profits, profit, fig=fig
                )

            elif profit > 0:
                potential_profits.append(diff)
                if is_short:
                    if (prev_tick["low"] - buy_price) < (
                        tick["low"] - buy_price
                    ) * 1.05:
                        (
                            transactions,
                            is_bought,
                            is_short,
                            profits,
                            potential_profits,
                            fig,
                        ) = sell_position(
                            transactions, tick, tick_day, profits, profit, fig=fig
                        )
                        if show:
                            print("Sold because of upward trend")
                        if show:
                            print("Sold because of upward trend")
                else:
                    if (prev_tick["low"] - buy_price) > (
                        tick["low"] - buy_price
                    ) * 1.05:
                        (
                            transactions,
                            is_bought,
                            is_short,
                            profits,
                            potential_profits,
                            fig,
                        ) = sell_position(
                            transactions, tick, tick_day, profits, profit, fig=fig
                        )
                        if show:
                            print("Sold because of downward trend")
                        if show:
                            print("Sold because of downward trend")
            else:
                potential_profits.append(diff)

        if (
            (prev_tick["MA_short"] < prev_tick["MA_long"])
            and (tick["MA_short"] > tick["MA_long"])
            or (prev_tick["MA_short"] > prev_tick["MA_long"])
            and (tick["MA_short"] < tick["MA_long"])
        ):
            if show:
                print(f"CROSSOVER at {tick_day}")
            if show:
                print(f"CROSSOVER at {tick_day}")
            balance = np.sum(profits) + capital
            c_price = tick["high"]

            ma_short_slope = get_line_slope(
                [
                    (step - 1) / df.shape[0],
                    prev_tick["MA_short"] / df["MA_short"][: step - 1].max(),
                ],
                [
                    step / df.shape[0],
                    tick["MA_short"] / df["MA_short"][:step].max(),
                ],
            )
            ma_long_slope = get_line_slope(
                [
                    (step - 1) / df.shape[0],
                    prev_tick["MA_long"] / df["MA_long"][: step - 1].max(),
                ],
                [step / df.shape[0], tick["MA_long"] / df["MA_long"][:step].max()],
            )
            crossover_angle = get_angle_between_lines(ma_short_slope, ma_long_slope)
            if show:
                print(f"    Balance: {balance}")
                print(f"    Price: {c_price}")
                print(f"    Crossover angle: {crossover_angle}")
            if show:
                print(f"    Balance: {balance}")
                print(f"    Price: {c_price}")
                print(f"    Crossover angle: {crossover_angle}")

        if (prev_tick["MA_short"] < prev_tick["MA_long"]) and (
            tick["MA_short"] > tick["MA_long"]
        ):
            if show:
                print(f"Short term upward crossover")
            if show:
                print(f"Short term upward crossover")
            if np.abs(crossover_angle) > 40:
                if np.sum(profits) + capital > tick["high"]:
                    transactions, is_bought, is_short, buy_price, fig = buy_position(
                        transactions, tick, tick_day, fig=fig
                    )
                else:
                    if show:
                        print(f"Not enough money to buy")
                    if show:
                        print(f"Not enough money to buy")
            else:
                if show:
                    print(f"Crossover angle to small {crossover_angle}")
                if show:
                    print(f"Crossover angle to small {crossover_angle}")

        if (prev_tick["MA_short"] > prev_tick["MA_long"]) and (
            tick["MA_short"] < tick["MA_long"]
        ):
            if show:
                print(f"Short term downward crossover")
            if show:
                print(f"Short term downward crossover")
            if is_bought:
                (
                    transactions,
                    is_bought,
                    is_short,
                    profits,
                    potential_profits,
                    fig,
                ) = sell_position(
                    transactions, tick, tick_day, profits, profit, fig=fig
                )
            if short_enabled:
                if np.abs(crossover_angle) > 40:
                    if np.sum(profits) + capital > tick["high"]:
                        (
                            transactions,
                            is_bought,
                            is_short,
                            buy_price,
                            fig,
                        ) = buy_position(
                            transactions, tick, tick_day, short_enabled, fig=fig
                        )
                    else:
                        if show:
                            print(f"Not enough money to buy")
                        if show:
                            print(f"Not enough money to buy")
                else:
                    if show:
                        print(f"Crossover angle to small {crossover_angle}")
                    if show:
                        print(f"Crossover angle to small {crossover_angle}")

        if step == df.shape[0] - 1 and (is_bought or is_short):
            (
                transactions,
                is_bought,
                is_short,
                profits,
                potential_profits,
                fig,
            ) = sell_position(transactions, tick, tick_day, profits, profit, fig=fig)

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
        print("")
        print("")
        print("End test")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    else:
        print(f"No trades made")
        print("")
        print("")
        print("End test")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    return fig, np.nansum(profits)
    return fig, np.nansum(profits)


def sell_position(transactions, tick, tick_day, profits, profit, fig=None):
    transactions.append({"type": "sell", "price": tick["low"], "time": tick_day})
    is_bought = False
    is_short = False
    profits.append(profit)
    potential_profits = [0]
    sell_price = tick["low"]
    print(f"Sold at s:{tick_day} | sp: {sell_price} | profit: {profit}")
    if fig:
        fig = plot_v_line(fig, tick["ctmString"], color="black")
        fig.show()
    return transactions, is_bought, is_short, profits, potential_profits, fig


def buy_position(transactions, tick, tick_day, is_short=False, fig=None):
    transactions.append({"type": "buy", "price": tick["high"], "time": tick_day})
    is_bought = True
    buy_price = tick["high"]
    print(f"Bought at s:{tick_day} | bp: {buy_price}")
    if fig:
        fig = plot_v_line(fig, tick["ctmString"], color="green")
        fig.show()
    return transactions, is_bought, is_short, buy_price, fig


def test_trend(df, period):
    start_date = df["ctmString"][-period].strftime("%Y-%m-%d")
    MA = df["close"].rolling(period).mean()
    MA = MA.dropna()

    MA = MA.dropna()

    trend = []
    for step in range(1, period):
        if MA.values[-(step + 1)] < MA.values[-step]:
            trend.append(1)
        else:
            trend.append(0)
    trend = np.array(trend)

    upward_ratio = len(trend[trend == 1]) / len(trend)
    downward_ratio = len(trend[trend == 0]) / len(trend)

    print(f"Trend tested from {start_date}: u{upward_ratio} | d{downward_ratio}")
    if downward_ratio > upward_ratio:
        return -downward_ratio
    elif upward_ratio > downward_ratio:
        return upward_ratio
    else:
        return 0


def get_line_slope(p1, p2):  # [x1, y1], [x2, y2]
    x1, y1 = p1
    x2, y2 = p2
    return (y2 - y1) / (x2 - x1)


def get_angle_between_lines(m1, m2):
    return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))


# PLOTTING FUNCTIONS
def plot_stock(df, params=None, symbol=None, return_fig=False, fig=None):
    if not fig:
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
    if "MA_short" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA_short"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="MA short",
            )
        )
    if "MA_long" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA_long"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="MA long",
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
        # xaxis=dict(type="category", categoryorder="category ascending"),
        # xaxis2=dict(type="category", categoryorder="category ascending"),
        # xaxis3=dict(type="category", categoryorder="category ascending"),
        # xaxis4=dict(type="category", categoryorder="category ascending"),
    )
    # fig.update_xaxes(
    #     rangebreaks=[
    #         dict(bounds=[17, 9], pattern="hour"),
    #         dict(bounds=["sat", "mon"]),
    #     ]
    # )
    # fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.show()
    if return_fig:
        return fig


def plot_v_line(fig, day, color):
    fig.add_vline(
        x=day,
        name="BP",
        line_dash="dash",
        line_color=color,
        line_width=3,
        row=1,
        col=1,
    )
    return fig


def get_macd_stoch(df):
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    STOCH = StochasticOscillator(
        high=df["high"], close=df["close"], low=df["low"], window=14, smooth_window=3
    )
    return macd, STOCH


def trading_loss(y, y_pred, **kwargs):
    y_true = np.zeros_like(y)
    return np.sum(y_pred - y_true)


class TradingEstimator(BaseEstimator, DensityMixin):
    def __init__(
        self,
        period,
        capital,
        symbol,
        short_enabled,
        short_ma=None,
        long_ma=None,
        show=False,
        fig=False,
    ) -> None:
        super().__init__()
        self.candles = None
        self.period = period
        self.capital = capital
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.short_enabled = short_enabled
        self.show = show
        if not fig:
            self.fig = fig
        else:
            self.fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.01,
                row_heights=[0.5, 0.1, 0.2, 0.2],
            )

    def fit(self, X, y=None, **kwargs):
        self.candles = X
        return self

    def score(self, X, y=None):
        return self.back_test(X)

    def back_test(self, df=None):
        if df is None:
            df = self.candles
        period = self.period
        capital = self.capital
        symbol = self.symbol
        short_ma = self.short_ma
        long_ma = self.long_ma
        trend = 0
        short_enabled = self.short_enabled
        show = self.show
        fig = self.fig

        df = add_rolling_means(df, short=short_ma, long=long_ma)
        if show:
            fig = plot_stock(df, symbol=symbol, return_fig=True, fig=None)
        start_date = df["ctmString"][-period].strftime("%Y-%m-%d")
        is_bought = False
        is_short = False
        profits = []
        profit = 0
        potential_profits = [0]
        if show:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("")
            print("")
            print(f"Start test from {start_date}")
            print(f"    Short selling: {short_enabled}")
            print(f"    Capital at risk: {capital}")
            print(f"    Trend: {trend}")

        for step in range(df.shape[0] - period, df.shape[0]):
            prev_tick = df.iloc[step - 1, :]
            tick = df.iloc[step, :]
            tick_day = str(tick["ctmString"])
            if is_bought:
                diff = tick["low"] - prev_tick["low"]
                profit = tick["low"] - buy_price
                if show:
                    print(f"Tick: {start_date}")
                    print(f"Sell Price {tick['low']}")
                    print(f"Buy price {buy_price}")
                    print(f"Profit {profit}")
                    print(f"Diff {diff}")
                    print(f"Potential profits {potential_profits}")

                if is_short:
                    profit = -1 * profit
                if profit < -(buy_price * 0.05):
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)

                elif profit > 0:
                    potential_profits.append(diff)
                    if is_short:
                        if (prev_tick["low"] - buy_price) < (
                            tick["low"] - buy_price
                        ) * 1.05:
                            (
                                is_bought,
                                is_short,
                                profits,
                                potential_profits,
                            ) = self.sell_position(profits, profit)
                            if show:
                                print("Sold because of upward trend")
                    else:
                        if (prev_tick["low"] - buy_price) > (
                            tick["low"] - buy_price
                        ) * 1.05:
                            (
                                is_bought,
                                is_short,
                                profits,
                                potential_profits,
                            ) = self.sell_position(profits, profit)
                            if show:
                                print("Sold because of downward trend")
                else:
                    potential_profits.append(diff)

            if (
                (prev_tick["MA_short"] < prev_tick["MA_long"])
                and (tick["MA_short"] > tick["MA_long"])
                or (prev_tick["MA_short"] > prev_tick["MA_long"])
                and (tick["MA_short"] < tick["MA_long"])
            ):
                if show:
                    print(f"CROSSOVER at {tick_day}")
                balance = np.sum(profits) + capital
                c_price = tick["high"]

                ma_short_slope = get_line_slope(
                    [
                        (step - 1) / df.shape[0],
                        prev_tick["MA_short"] / df["MA_short"][: step - 1].max(),
                    ],
                    [
                        step / df.shape[0],
                        tick["MA_short"] / df["MA_short"][:step].max(),
                    ],
                )
                ma_long_slope = get_line_slope(
                    [
                        (step - 1) / df.shape[0],
                        prev_tick["MA_long"] / df["MA_long"][: step - 1].max(),
                    ],
                    [step / df.shape[0], tick["MA_long"] / df["MA_long"][:step].max()],
                )
                crossover_angle = get_angle_between_lines(ma_short_slope, ma_long_slope)
                if show:
                    print(f"    Balance: {balance}")
                    print(f"    Price: {c_price}")
                    print(f"    Crossover angle: {crossover_angle}")

            if (prev_tick["MA_short"] < prev_tick["MA_long"]) and (
                tick["MA_short"] > tick["MA_long"]
            ):
                if show:
                    print(f"Short term upward crossover")
                if np.abs(crossover_angle) > 40:
                    if np.sum(profits) + capital > tick["high"]:
                        (is_bought, is_short, buy_price) = self.buy_position(tick)
                    else:
                        if show:
                            print(f"Not enough money to buy")
                else:
                    if show:
                        print(f"Crossover angle to small {crossover_angle}")

            if (prev_tick["MA_short"] > prev_tick["MA_long"]) and (
                tick["MA_short"] < tick["MA_long"]
            ):
                if show:
                    print(f"Short term downward crossover")
                if is_bought:
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)
                if short_enabled:
                    if np.abs(crossover_angle) > 40:
                        if np.sum(profits) + capital > tick["high"]:
                            (is_bought, is_short, buy_price) = self.buy_position(
                                tick, short_enabled
                            )
                        else:
                            if show:
                                print(f"Not enough money to buy")
                    else:
                        if show:
                            print(f"Crossover angle to small {crossover_angle}")

            if step == df.shape[0] - 1 and (is_bought or is_short):
                (is_bought, is_short, profits, potential_profits) = self.sell_position(
                    profits, profit
                )

        # FINISHED BACKTEST
        if len(profits) > 0:
            np_profits = np.array(profits)
            lossing_trades = np_profits[np_profits < 0]
            winning_trades = np_profits[np_profits > 0]
            if show:
                print(f"Profits: {np.nansum(profits)}")
                print(
                    f"Winning/Lossing trades: {len(winning_trades)}|{len(lossing_trades)}"
                )
                print(
                    f"Average transaction profit: w:{np.nanmean(winning_trades)} | l: {np.nanmean(lossing_trades)}"
                )
                print("")
                print("")
                print("End test")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        else:
            profits = 0
        return np.sum(profits)

    def sell_position(self, profits, profit):
        is_bought = False
        is_short = False
        profits.append(profit)
        potential_profits = [0]
        return is_bought, is_short, profits, potential_profits

    def buy_position(self, tick, is_short=False):
        is_bought = True
        buy_price = tick["high"]
        return is_bought, is_short, buy_price


def trading_loss(y, y_pred, **kwargs):
    y_true = np.zeros_like(y)
    return np.sum(y_pred - y_true)


class TradingEstimator(BaseEstimator, DensityMixin):
    def __init__(
        self,
        period,
        capital,
        symbol,
        short_enabled,
        short_ma=None,
        long_ma=None,
        show=False,
        fig=False,
    ) -> None:
        super().__init__()
        self.candles = None
        self.period = period
        self.capital = capital
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.short_enabled = short_enabled
        self.show = show
        if not fig:
            self.fig = fig
        else:
            self.fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.01,
                row_heights=[0.5, 0.1, 0.2, 0.2],
            )

    def fit(self, X, y=None, **kwargs):
        self.candles = X
        return self

    def score(self, X, y=None):
        return self.back_test(X)

    def back_test(self, df=None):
        if df is None:
            df = self.candles
        period = self.period
        capital = self.capital
        symbol = self.symbol
        short_ma = self.short_ma
        long_ma = self.long_ma
        trend = 0
        short_enabled = self.short_enabled
        show = self.show
        fig = self.fig

        df = add_rolling_means(df, short=short_ma, long=long_ma)
        if show:
            fig = plot_stock(df, symbol=symbol, return_fig=True, fig=None)
        start_date = df["ctmString"][-period].strftime("%Y-%m-%d")
        is_bought = False
        is_short = False
        profits = []
        profit = 0
        potential_profits = [0]
        if show:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("")
            print("")
            print(f"Start test from {start_date}")
            print(f"    Short selling: {short_enabled}")
            print(f"    Capital at risk: {capital}")
            print(f"    Trend: {trend}")

        for step in range(df.shape[0] - period, df.shape[0]):
            prev_tick = df.iloc[step - 1, :]
            tick = df.iloc[step, :]
            tick_day = str(tick["ctmString"])
            if is_bought:
                diff = tick["low"] - prev_tick["low"]
                profit = tick["low"] - buy_price
                if show:
                    print(f"Tick: {start_date}")
                    print(f"Sell Price {tick['low']}")
                    print(f"Buy price {buy_price}")
                    print(f"Profit {profit}")
                    print(f"Diff {diff}")
                    print(f"Potential profits {potential_profits}")

                if is_short:
                    profit = -1 * profit
                if profit < -(buy_price * 0.05):
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)

                elif profit > 0:
                    potential_profits.append(diff)
                    if is_short:
                        if (prev_tick["low"] - buy_price) < (
                            tick["low"] - buy_price
                        ) * 1.05:
                            (
                                is_bought,
                                is_short,
                                profits,
                                potential_profits,
                            ) = self.sell_position(profits, profit)
                            if show:
                                print("Sold because of upward trend")
                    else:
                        if (prev_tick["low"] - buy_price) > (
                            tick["low"] - buy_price
                        ) * 1.05:
                            (
                                is_bought,
                                is_short,
                                profits,
                                potential_profits,
                            ) = self.sell_position(profits, profit)
                            if show:
                                print("Sold because of downward trend")
                else:
                    potential_profits.append(diff)

            if (
                (prev_tick["MA_short"] < prev_tick["MA_long"])
                and (tick["MA_short"] > tick["MA_long"])
                or (prev_tick["MA_short"] > prev_tick["MA_long"])
                and (tick["MA_short"] < tick["MA_long"])
            ):
                if show:
                    print(f"CROSSOVER at {tick_day}")
                balance = np.sum(profits) + capital
                c_price = tick["high"]

                ma_short_slope = get_line_slope(
                    [
                        (step - 1) / df.shape[0],
                        prev_tick["MA_short"] / df["MA_short"][: step - 1].max(),
                    ],
                    [
                        step / df.shape[0],
                        tick["MA_short"] / df["MA_short"][:step].max(),
                    ],
                )
                ma_long_slope = get_line_slope(
                    [
                        (step - 1) / df.shape[0],
                        prev_tick["MA_long"] / df["MA_long"][: step - 1].max(),
                    ],
                    [step / df.shape[0], tick["MA_long"] / df["MA_long"][:step].max()],
                )
                crossover_angle = get_angle_between_lines(ma_short_slope, ma_long_slope)
                if show:
                    print(f"    Balance: {balance}")
                    print(f"    Price: {c_price}")
                    print(f"    Crossover angle: {crossover_angle}")

            if (prev_tick["MA_short"] < prev_tick["MA_long"]) and (
                tick["MA_short"] > tick["MA_long"]
            ):
                if show:
                    print(f"Short term upward crossover")
                if np.abs(crossover_angle) > 40:
                    if np.sum(profits) + capital > tick["high"]:
                        (is_bought, is_short, buy_price) = self.buy_position(tick)
                    else:
                        if show:
                            print(f"Not enough money to buy")
                else:
                    if show:
                        print(f"Crossover angle to small {crossover_angle}")

            if (prev_tick["MA_short"] > prev_tick["MA_long"]) and (
                tick["MA_short"] < tick["MA_long"]
            ):
                if show:
                    print(f"Short term downward crossover")
                if is_bought:
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)
                if short_enabled:
                    if np.abs(crossover_angle) > 40:
                        if np.sum(profits) + capital > tick["high"]:
                            (is_bought, is_short, buy_price) = self.buy_position(
                                tick, short_enabled
                            )
                        else:
                            if show:
                                print(f"Not enough money to buy")
                    else:
                        if show:
                            print(f"Crossover angle to small {crossover_angle}")

            if step == df.shape[0] - 1 and (is_bought or is_short):
                (is_bought, is_short, profits, potential_profits) = self.sell_position(
                    profits, profit
                )

        # FINISHED BACKTEST
        if len(profits) > 0:
            np_profits = np.array(profits)
            lossing_trades = np_profits[np_profits < 0]
            winning_trades = np_profits[np_profits > 0]
            if show:
                print(f"Profits: {np.nansum(profits)}")
                print(
                    f"Winning/Lossing trades: {len(winning_trades)}|{len(lossing_trades)}"
                )
                print(
                    f"Average transaction profit: w:{np.nanmean(winning_trades)} | l: {np.nanmean(lossing_trades)}"
                )
                print("")
                print("")
                print("End test")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        else:
            profits = 0
        return np.sum(profits)

    def sell_position(self, profits, profit):
        is_bought = False
        is_short = False
        profits.append(profit)
        potential_profits = [0]
        return is_bought, is_short, profits, potential_profits

    def buy_position(self, tick, is_short=False):
        is_bought = True
        buy_price = tick["high"]
        return is_bought, is_short, buy_price
