import os
import pandas as pd
from trader_utils import *
from sqlalchemy import create_engine
from xAPIConnector import *
from creds import creds
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import threading
from trader_db_utils import *
from trader_api_utils import *
from tqdm import tqdm
import os
from logger_settings import setup_logging
import logging
from trading_sessions import TradingSession

today = get_today()
todayms, today_int = get_today_ms()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../data/"
symbol_path = curr_path + "../../symbols/"
cluster_path = curr_path + "../../clusters/"
model_path = curr_path + "../../model/"
result_path = curr_path + "../../result/"
docs_path = curr_path + "../../docs/"
database_path = curr_path + "../../database/"
logs_path = curr_path + "../../logs/"

if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")

logger = logging.getLogger(__name__)
logger = setup_logging(logger, logs_path, __name__, console_debug=True)

minimun_trade = 10
minimun_trade = 10


class Trader:
    def __init__(self, name, capital, max_risk, trader_type, tick_queue, clock) -> None:
        self.user = creds[name.split(":")[0]]["user"]
        self.passw = creds[name.split(":")[0]]["passw"]
        self.name = name
        self.capital = capital
        self.max_risk = max_risk
        self.trader_type = trader_type
        traderid = get_trader_id(name)
        if traderid is None:
            traderid = insert_trader(name, capital, max_risk)
            logger.info(f"Trader created id: {traderid}")
        else:
            logger.info(f"Trader exists id: {traderid}")
        self.traderid = traderid

        self.apiSession = ApiSessionManager(self.user, self.passw, clock)
        self.apiSession.set_apiClient()
        self.tick_queue = tick_queue
        self.clock = clock

    def look_for_suitable_symbols_v1(self, df):
        # TODO look for suitable symbols
        # Look for symbols with:
        #   - Volatility bigger than spread [DONE LATER]
        #   - Trader_type products
        #   - Ask comparable to our max_risk

        df = df[df["categoryName"] == self.trader_type]
        df["min_price"] = (
            df["ask"] * df["lotMin"] * df["contractSize"] * (df["leverage"] / 100)
        )
        df = df[df["min_price"] <= (self.capital * self.max_risk)]
        df["spread_percentage"] = (df["ask"] - df["bid"]) / df["ask"]
        df = df.sort_values(by=["spread_percentage"])
        return df

    def update_stocks(self, df, period, days=14):
        start_date = datetime.now() - timedelta(days=days)
        for symbol in tqdm(list(df["symbol"])):
            if not get_candles_today():
                self.apiSession.store_past_candles(symbol, start_date, period)

    def evaluate_stocks(self, date, threshold=25):
        symbols = get_distict_symbols()
        for symbol in symbols:
            symbol = symbol[0]
            symbol_info = get_symbol_info(symbol)
            symbol_stats = get_symbol_stats(symbol, date)
            if symbol_info["spreadRaw"] * (threshold) < symbol_stats["std_close"]:
                score = symbol_stats["std_close"] / symbol_info["spreadRaw"]
                spread = symbol_info["spreadRaw"]
                logger.info(
                    f"CHOSEN\nSymbol {symbol}: {symbol_stats['std_close']} / {symbol_info['spreadRaw']} = {score}"
                )
                insert_params(
                    day=datetime.today(),
                    symbol=symbol,
                    score=score,
                    short_ma=1,
                    long_ma=10,
                    profit_exit=spread,
                    loss_exit=0,
                    min_angle=20,
                )

            elif symbol_info["spreadRaw"] < symbol_stats["std_close"]:
                logger.info(
                    f"GOOD\nSymbol {symbol}: {symbol_stats['std_close']} / {symbol_info['spreadRaw']} = {symbol_stats['std_close'] / symbol_info['spreadRaw']}"
                )
            else:
                logger.info(
                    f"BAD\nSymbol {symbol}: {symbol_stats['std_close']} / {symbol_info['spreadRaw']} = {symbol_stats['std_close'] / symbol_info['spreadRaw']}"
                )

    def start_trading_session(self, params, test=False):
        self.apiSession.load_missed_candles(params["symbol"], params["long_ma"])
        session = TradingSession(
            name=self.name,
            capital=self.capital * self.max_risk,
            symbol=params["symbol"],
            short_ma=params["short_ma"],
            long_ma=params["long_ma"],
            profit_exit=params["profit_exit"],
            loss_exit=params["loss_exit"],
            min_angle=params["min_angle"],
            apiClient=self.apiSession,
            test=test,
            tick_queue=self.tick_queue,
            clock=self.clock,
        )
        self.session = session
        self.apiSession.set_session(session)
        session.start()
