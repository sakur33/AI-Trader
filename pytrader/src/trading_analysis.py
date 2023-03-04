from xAPIConnector import *
from trading_systems import CrossTrader, BacktestCrossTrader
from trader_utils import *
from trader_db_utils import *
from logger_settings import *
from custom_exceptions import *
import matplotlib.pyplot as plt

from multiprocessing import Process

from tqdm import tqdm
import pickle as pkl
import logging
import time
import os
import sys
import argparse
from multiprocessing import SimpleQueue
from connection_clock import Clock
import warnings
import copy as cp

TRADER_NAME = "ANALYSIS"
test = 1
custom_logger = CustomLogger(test_name=TRADER_NAME)
logger = logging.getLogger(__name__)
logger = custom_logger.setup_logging(logger, TRADER_NAME, console_debug=True)


def load_candles():
    trader = BacktestCrossTrader(
        name=f"trader68709:",
        capital=1000,
        max_risk=0.05,
        trader_type="FX",
        logger=logger,
        test=test,
    )

    logger.info(f"OFFLINE TRADER")
    trader.CLOCK.wait_clock()
    trader.start_api_client()

    if os.path.exists("results/symbols_df.pkl"):
        with open("results/symbols_df.pkl", "rb") as f:
            symbols_df = pkl.load(f)
    else:
        commandResponse = trader.CLIENT.commandExecute(commandName="getAllSymbols")
        symbols_df = return_as_df(commandResponse["returnData"])
        trader.DB.insert_symbols(symbols_df)

        with open("results/symbols_df.pkl", "wb") as f:
            pkl.dump(symbols_df, f, pkl.HIGHEST_PROTOCOL)

    trader.update_stocks(
        symbols_df[symbols_df["categoryName"] == "FX"],
        period=1,
        days=360,
        force=True,
    )


def get_slope(prev_candle, c_candle, short_ma, long_ma):
    min_ctm, max_ctm = short_ma, long_ma
    min_close, max_close = c_candle["min"], c_candle["max"]

    prev_ctm = min_max_norm(prev_candle["index"], min_ctm, max_ctm)
    c_ctm = min_max_norm(c_candle["index"], min_ctm, max_ctm)
    prev_short = min_max_norm(prev_candle["short_ma"], min_close, max_close)
    c_short = min_max_norm(c_candle["short_ma"], min_close, max_close)
    prev_long = min_max_norm(prev_candle["long_ma"], min_close, max_close)
    c_long = min_max_norm(c_candle["long_ma"], min_close, max_close)

    p1 = [prev_ctm, prev_short]
    p2 = [c_ctm, c_short]
    x1, y1 = p1
    x2, y2 = p2
    ma_short_slope = (y2 - y1) / (x2 - x1)

    p1 = [prev_ctm, prev_long]
    p2 = [c_ctm, c_long]
    x1, y1 = p1
    x2, y2 = p2
    ma_long_slope = (y2 - y1) / (x2 - x1)
    return ma_short_slope, ma_long_slope


def get_absolute_angles(prev_candle, c_candle, short_ma, long_ma):
    ma_short_slope, ma_long_slope = get_slope(prev_candle, c_candle, short_ma, long_ma)
    angle_short = np.degrees(
        np.arctan((ma_short_slope - 0) / (1 + (ma_short_slope * 0)))
    )
    angle_long = np.degrees(np.arctan((ma_long_slope - 0) / (1 + (ma_long_slope * 0))))
    return angle_short, angle_long


def get_angle_between_slopes(prev_candle, c_candle, short_ma, long_ma):
    m1, m2 = get_slope(prev_candle, c_candle)
    return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))


def buy_slope(self, prev_candle, c_candle, min_angle, spread):
    buy_short = False
    buy_long = False
    short_a, long_a = get_absolute_angles(prev_candle, c_candle)
    if np.abs(c_candle["short_ma"] - c_candle["long_ma"]) > spread:
        if short_a < -(min_angle * 2) and long_a < -(min_angle / 2):
            # trend is negative
            buy_short = False
            buy_long = True
        elif short_a > (min_angle * 2) and long_a > (min_angle / 2):
            # trend is positive
            buy_short = True
            buy_long = False
        else:
            buy_short = False
            buy_long = False
    else:
        buy_short = False
        buy_long = False
    return buy_short, buy_long


def find_trades(candle):
    index = candle["index"]
    if (
        (candle["angle_short"] < -(min_angle * 2))
        and (candle["angle_long"] > (min_angle / 5))
        and (candle["short_ma"] > candle["long_ma"])
    ):
        short_signals += 2
        buy_signals = 0
        if short_signals > 5 and not short:
            trades[index] = -1
            short_signals = 0
            short = True
            buy = False
        # trades[index] = -1
    elif (candle["angle_short"] < -(min_angle)) and (
        candle["angle_long"] < -(min_angle / 2)
    ):
        short_signals += 2
        buy_signals = 0
        if short_signals > 5 and not short:
            trades[index] = -1
            short_signals = 0
            short = True
            buy = False
        # trades[index] = -1
    elif (
        (candle["angle_short"] > (min_angle * 2))
        and (candle["angle_long"] < -(min_angle / 5))
        and (candle["short_ma"] < candle["long_ma"])
    ):
        short_signals = 0
        buy_signals += 2
        if buy_signals > 5 and not buy:
            trades[index] = 1
            buy_signals = 0
            buy = True
            short = False
        # trades[index] = 1
    elif (candle["angle_short"] > (min_angle)) and (
        candle["angle_long"] > (min_angle / 2)
    ):
        short_signals = 0
        buy_signals += 2
        if buy_signals > 5 and not buy:
            trades[index] = 1
            buy_signals = 0
            buy = True
            short = False
        # trades[index] = 1
    else:
        # short_signals = 0
        # buy_signals = 0
        trades[index] = 0


def run_trade(candles, exit, exit_profit):
    profits = []
    first_candle = candles.iloc[0, :]
    price = np.round(np.average(first_candle[["open", "close", "high", "low"]]), 5)
    first_index = first_candle["index"]
    first_time = first_candle.name
    trade_type = first_candle["trades"]
    for index in range(1, candles.shape[0]):
        candle = candles.iloc[index, :]
        if trade_type == 1:
            # BUY
            order_type = "BUY"
            trigger = candle["short_ma"]
            new_price = np.round(
                np.average(candle[["open", "close", "high", "low"]]), 5
            )
            profits.append(new_price - price)
            if new_price < price:
                close_type = -1
                break

        elif trade_type == -1:
            # SHORT
            order_type = "SHORT"
            trigger = candle["short_ma"]
            new_price = np.round(
                np.average(candle[["open", "close", "high", "low"]]), 5
            )
            profits.append(price - new_price)
            if new_price > price:
                close_type = -1
                break

        if len(profits) > 3:
            if profits[-2] / profits[-1] < exit_profit:
                close_type = 0
                break
            elif profits[-1] > exit:
                close_type = 1
                break
    return (
        profits[-1],
        {
            "Type": order_type,
            "open_index": first_index,
            "open_time": first_time,
            "open_price": price,
            "close_index": candle["index"],
            "close_time": candle.name,
            "close_price": new_price,
            "profit": profits[-1],
            "profits": profits,
        },
        close_type,
    )


if os.path.exists("results/jan2023_forex_candles.pkl"):
    logger.info("Loading Candles of 2022")
    with open("results/jan2023_forex_candles.pkl", "rb") as f:
        fx_candles = pkl.load(f)
else:
    if os.path.exists("results/symbols_df.pkl"):
        logger.info("")
        with open("results/symbols_df.pkl", "rb") as f:
            symbols_df = pkl.load(f)
    else:
        load_candles()

    trader = BacktestCrossTrader(
        name=f"trader68709:",
        capital=1000,
        max_risk=0.05,
        trader_type="FX",
        logger=logger,
        test=test,
    )
    start_date = (datetime.strptime("23-02-2022", "%d-%m-%Y")).strftime("%m-%d-%Y")
    end_date = (datetime.strptime("23-02-2023", "%d-%m-%Y")).strftime("%m-%d-%Y")
    candles = trader.DB.get_candles_range(start_date=start_date, end_date=end_date)
    candles = candles[["symbol", "ctmstring", "open", "high", "low", "close", "vol"]]

    fx_candles = cp.deepcopy(candles)
    fx_symbols = list(symbols_df[symbols_df["categoryName"] == "FX"]["symbol"].values)
    for symbol in candles["symbol"].unique():
        if symbol not in fx_symbols:
            fx_candles = fx_candles[fx_candles["symbol"] != symbol]

    fx_candles["ctmstring"] = pd.to_datetime(fx_candles["ctmstring"])
    fx_candles = fx_candles.set_index("ctmstring")
    fx_candles = fx_candles[fx_candles.index.dayofweek < 5]
    with open("results/last_year_forex_candles.pkl", "wb") as f:
        pkl.dump(fx_candles, f, pkl.HIGHEST_PROTOCOL)

long_ma = 1000
short_ma = 30
spread = 0.00009
min_angle = 30
for symbol in fx_candles["symbol"].unique():
    candles = cp.deepcopy(fx_candles[fx_candles["symbol"] == symbol])
    candles = candles.reset_index()
    candles = candles.drop_duplicates()
    candles = candles.set_index("ctmstring")
    candles["index"] = np.arange(0, candles.shape[0])
    candles["short_ma"] = candles["close"].rolling(60).mean()
    candles["long_ma"] = candles["close"].rolling(long_ma).mean()
    candles["max"] = candles["high"].rolling(long_ma).max()
    candles["min"] = candles["low"].rolling(long_ma).min()

    short_angles = []
    long_angles = []

    candles["prev_ctm"] = 0.999
    candles["c_ctm"] = 1
    candles["prev_short"] = min_max_norm(
        candles["short_ma"].shift(1), candles["min"], candles["max"]
    )
    candles["c_short"] = min_max_norm(
        candles["short_ma"], candles["min"], candles["max"]
    )
    candles["prev_long"] = min_max_norm(
        candles["long_ma"].shift(1), candles["min"], candles["max"]
    )
    candles["c_long"] = min_max_norm(candles["long_ma"], candles["min"], candles["max"])

    candles["ma_short_slope"] = (candles["c_short"] - candles["prev_short"]) / (
        candles["c_ctm"] - candles["prev_ctm"]
    )
    candles["ma_long_slope"] = (candles["c_long"] - candles["prev_long"]) / (
        candles["c_ctm"] - candles["prev_ctm"]
    )

    candles["angle_short"] = np.degrees(
        np.arctan(
            (candles["ma_short_slope"] - 0) / (1 + (candles["ma_short_slope"] * 0))
        )
    )
    candles["angle_long"] = np.degrees(
        np.arctan((candles["ma_long_slope"] - 0) / (1 + (candles["ma_long_slope"] * 0)))
    )

    trades = np.zeros((candles.shape[0]))
    short_signals = 0
    buy_signals = 0
    buy = False
    short = False

    for index in tqdm(range(long_ma, candles.shape[0])):
        # logger.info(f"\nBuy-signals:{buy_signals}\nShort-signals: {short_signals}")
        candle = candles.iloc[index, :]
        if np.abs(candle["short_ma"] - candle["long_ma"] > spread):
            if (
                (candle["angle_short"] < -(min_angle * 2))
                and (candle["angle_long"] > (min_angle / 5))
                and (candle["short_ma"] > candle["long_ma"])
            ):
                short_signals += 1
                buy_signals = 0
                if short_signals < 2 and not short:
                    trades[index] = -1
                    short_signals = 0
                    short = True
                    buy = False
                # trades[index] = -1
                # trades[index] = 0
            elif (candle["angle_short"] < -(min_angle)) and (
                candle["angle_long"] < -(min_angle / 2)
            ):
                short_signals += 1
                buy_signals = 0
                if short_signals < 2 and not short:
                    trades[index] = -1
                    short_signals = 0
                    short = True
                    buy = False
                # trades[index] = -1
                # trades[index] = 0
            elif (
                (candle["angle_short"] > (min_angle * 2))
                and (candle["angle_long"] < -(min_angle / 5))
                and (candle["short_ma"] < candle["long_ma"])
            ):
                short_signals = 0
                buy_signals += 1
                if buy_signals > 2 and not buy:
                    trades[index] = 1
                    buy_signals = 0
                    buy = True
                    short = False
                # trades[index] = 1
                # trades[index] = 0
            elif (candle["angle_short"] > (min_angle)) and (
                candle["angle_long"] > (min_angle / 2)
            ):
                short_signals = 0
                buy_signals += 1
                if buy_signals < 2 and not buy:
                    trades[index] = 1
                    buy_signals = 0
                    buy = True
                    short = False
                # trades[index] = 1
                # trades[index] = 0
            else:
                # short_signals = 0
                # buy_signals = 0
                trades[index] = 0

    candles["trades"] = trades
    plot_stock_simple(
        candles.iloc[:1000, :],
        show=True,
    )
    pass

    indices = list(np.where(candles["trades"] != 0)[0])
    short = False
    buy = False
    last_price = None
    returns = []
    trades = []
    close_types = []
    for index in tqdm(indices):
        ret, trade, close_type = run_trade(
            candles.iloc[index:, :], exit=0.01, exit_profit=0.95
        )
        returns.append(ret)
        trades.append(trade)
        close_types.append(close_type)

    plt.figure()
    plt.bar(range(len(returns)), returns)
    plt.title(f"Result: {np.sum(returns)}")
    plt.figure()
    counts, bins = np.histogram(close_types, bins=3)
    plt.hist(x=["-1", "0", "1"], bins=3, weights=counts, align="mid")
    plt.figure()
    plt.plot(np.cumsum(returns))
    plt.pause(0.1)

    plot_stock_all_trades(candles, trades, show=False)
