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
from logger_settings import *
import logging
from trading_sessions import TradingSession

if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")

logger = logging.getLogger(__name__)
logger = setup_logging(logger, logs_path, __name__, console_debug=True)

minimun_trade = 10
minimun_trade = 10


class Trader:
    def __init__(
        self,
        name,
        capital,
        max_risk,
        trader_type=None,
        tick_queue=None,
        clock=None,
        candle_queue=None,
    ) -> None:
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

    def start_trading_session(self, params, test=False):
        try:
            self.apiSession.load_missed_candles(params["symbol"], 500)
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
                candle_queue=self.candle_queue,
                clock=self.clock,
            )
            self.session = session
            self.apiSession.set_session(session)
            session.start()
        except Exception as e:
            raise LogicError
