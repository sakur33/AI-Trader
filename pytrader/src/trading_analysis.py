from xAPIConnector import *
from trading_systems import CrossTrader, BacktestCrossTrader
from trader_utils import *
from trader_db_utils import *
from logger_settings import *
from custom_exceptions import *

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


def get_slope(self, prev_candle, c_candle, candles, short_ma, long_ma):
    min_ctm, max_ctm = short_ma, long_ma
    min_close, max_close = np.min(candles["low"]), np.max(candles["high"])

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


def get_absolute_angles(self, prev_candle, c_candle):
    ma_short_slope, ma_long_slope = self.get_slope(prev_candle, c_candle)
    angle_short = np.degrees(
        np.arctan((ma_short_slope - 0) / (1 + (ma_short_slope * 0)))
    )
    angle_long = np.degrees(np.arctan((ma_long_slope - 0) / (1 + (ma_long_slope * 0))))
    return angle_short, angle_long


def get_angle_between_slopes(self, prev_candle, c_candle):
    m1, m2 = self.get_slope(prev_candle, c_candle)
    return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))


def buy_slope(self, prev_candle, c_candle):
    buy_short = False
    buy_long = False
    short_a, long_a = self.get_absolute_angles(prev_candle, c_candle)
    if np.abs(c_candle["short_ma"] - c_candle["long_ma"]) > self.spread:
        if short_a < -(self.min_angle * 2) and long_a < -(self.min_angle / 2):
            # trend is negative
            buy_short = False
            buy_long = True
        elif short_a > (self.min_angle * 2) and long_a > (self.min_angle / 2):
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


if os.path.exists("results/last_year_forex_candles.pkl"):
    logger.info("Loading Candles of 2022")
    with open("results/last_year_forex_candles.pkl", "rb") as f:
        fx_candles = pkl.load(f)
else:
    if os.path.exists("results/symbols_df.pkl"):
        logger.info("")
        with open("results/symbols_df.pkl", "rb") as f:
            symbols_df = pkl.load(f)
    else:
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

pass

for symbol in fx_candles["symbol"].unique():
    candles = cp.deepcopy(fx_candles[fx_candles["symbol"] == symbol])

    candles["short_ma"] = candles["close"].rolling(30).mean()
    candles["long_ma"] = candles["close"].rolling(1000).mean()

    plot_stock_simple(
        candles.iloc[: int(candles.shape[0] / 200), :],
        show=True,
    )
