from xAPIConnector import *
from trading_systems import BaseTrader, RandomTrader, BacktestRandomTrader
from sklearn.model_selection import GridSearchCV
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

warnings.simplefilter(action="ignore", category=FutureWarning)

DEFAULT_XAPI_ADDRESS = "xapi.xtb.com"
DEFAULT_XAPI_PORT = 5124
DEFAULT_XAPI_STREAMING_PORT = 5125

parser = argparse.ArgumentParser()
parser.add_argument("--trader_name", type=str, help="Trader name")
parser.add_argument("--force_backtest", type=int, default=0, help="Force Backtesting")
parser.add_argument("--test", type=int, default=0, help="Override trading params")
args = parser.parse_args()

if args.trader_name:
    TRADER_NAME = args.trader_name
else:
    TRADER_NAME = "tests"

force_backtest = args.force_backtest

is_debug_run = os.environ.get("is_debug_run")

if is_debug_run or args.test:
    test = True
    first_run = True
    force_backtest = True
else:
    test = False


def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)


def main():
    global TRADER_NAME
    custom_logger = CustomLogger(test_name=TRADER_NAME)
    logger = logging.getLogger(__name__)
    logger = custom_logger.setup_logging(logger, TRADER_NAME, console_debug=True)
    weekno = datetime.today().weekday()
    if force_backtest:
        trader = RandomTrader(
            name=f"trader:",
            capital=1000,
            max_risk=0.05,
            trader_type="FX",
            logger=logger,
            test=is_debug_run or test,
        )
        trader.start_api_client()
        logger.info(f"OFFLINE TRADER")
        trader.CLOCK.wait_clock()
        logger.info("Loading symbols")
        if not os.path.exists("./data/symbols_today.pkl"):
            commandResponse = trader.CLIENT.commandExecute(commandName="getAllSymbols")
            symbols_df = return_as_df(commandResponse["returnData"])
            trader.DB.insert_symbols(symbols_df)
            with open("./data/symbols_today.pkl", "wb") as f:
                pkl.dump(symbols_df, f, pkl.HIGHEST_PROTOCOL)
        else:
            with open("./data/symbols_today.pkl", "rb") as f:
                symbols_df = pkl.load(f)
        logger.info("Filtering suitable symbols")
        symbols_df = symbols_df[symbols_df["categoryName"] == "FX"]
        logger.info("Updating stocks...")
        # trader.update_stocks(symbols_df, period=1, days=30, force=False)
        logger.info("Evaluating stocks...")
        if hasattr(trader, "evaluate_stocks"):
            symbol_analysis = trader.evaluate_stocks(
                date=(datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d"),
                threshold=40,
            )
            logger.info(symbol_analysis)
            chosen_symbols = []
            for symbol in symbol_analysis.keys():
                logger.info(f"{symbol}: {symbol_analysis[symbol]['Score']}")
                if symbol_analysis[symbol]["Score"] > 1:
                    chosen_symbols.append(symbol)
        else:
            chosen_symbols = symbols_df["symbol"]
        for symbol in chosen_symbols:
            df = trader.DB.get_candles_range(
                start_date=(datetime.now() - timedelta(days=30)).strftime("%m-%d-%Y"),
                end_date=get_today(),
                symbol=symbol,
            )
            params = {
                "max_profit_percentage": np.linspace(1.01, 1.5, num=5),
                "good_profit_percentage": np.linspace(1.01, 1.5, num=5),
                "max_loss_percentage": np.linspace(0.005, 0.1, num=5),
                "max_instant_drawdown_percentage": np.linspace(1.01, 1.5, num=5),
                "max_drawdown_from_max_percentage": np.linspace(1.01, 1.5, num=5),
                "max_loss_from_profitable_max_percentage": np.linspace(
                    0.005, 0.1, num=5
                ),
            }
            clf_trader = BacktestRandomTrader()
            clf = GridSearchCV(clf_trader, params, verbose=3)
            clf.fit(X=df, y=None)

    else:
        trader = RandomTrader(
            name=f"trader:",
            capital=1000,
            max_risk=0.05,
            trader_type="FX",
            logger=logger,
            test=is_debug_run or test,
        )
        logger.info(f"ONLINE TRADER")
        symbol = "GBPUSD"
        # params_df = trader.DB.get_trading_params()
        # iterator = params_df.iterrows()
        # index, row = next(iterator)
        trader.name = f"trader:{symbol}:{TRADER_NAME}"
        trader.test_name = TRADER_NAME
        trader.start_api_client()
        trader.SYMBOl = "GBPUSD"
        trader.VOLUME = 0.01

        trader.SYMBOL_INFO = trader.DB.get_symbol_info(symbol)
        trader.start_stream_clients(symbol, tick=True, candle=True, trade=True)
        trader.set_last_candle()
        logger.info("SYSTEM STARTS")
        trader.recover_trades()
        while True:
            try:
                if not trader.exception_queue.empty():
                    exception = trader.exception_queue.get()
                    raise exception
                trader.set_last_tick()
                trader.set_last_candle()
                trader.set_last_trade()
                trader.set_last_profit(trades=2)
                trader.step()
            except KeyboardInterrupt:
                logger.info(f"Keyboard Interrupt")
                quit()
            except Exception as e:
                logger.info(f"Exception in main loop | {e}")
                time.sleep(30)
                if not test:
                    logger.info(f"Restarting system")
                    restart_program()
                else:
                    logger.info(f"No restart in debug mode")
            pass


if __name__ == "__main__":
    main()
