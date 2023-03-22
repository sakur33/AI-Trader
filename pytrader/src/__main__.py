from xAPIConnector import *
from trading_systems import CrossTrader, BacktestCrossTrader, RandomTrader
from trader_utils import *
from trader_db_utils import *
from logger_settings import *
from custom_exceptions import *

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
        trader = BacktestCrossTrader(
            name=f"trader68709:",
            capital=1000,
            max_risk=0.05,
            trader_type="FX",
            logger=logger,
            test=is_debug_run or test,
        )
        # trader.start_api_client()
        # apiClient = trader.CLIENT
        logger.info(f"OFFLINE TRADER")
        trader.CLOCK.wait_clock()
        # logger.info("Loading symbols")
        # symbols = trader.DB.get_symbol_today()
        # if not isinstance(symbols, pd.DataFrame):
        #     commandResponse = apiClient.commandExecute(commandName="getAllSymbols")
        #     symbols_df = return_as_df(commandResponse["returnData"])
        #     trader.DB.insert_symbols(symbols_df)
        # else:
        #     symbols_df = symbols
        # logger.info("Filtering suitable symbols")
        # symbols_df = trader.look_for_suitable_symbols_v1(
        #     symbols_df, symbol_type="FX", capital=1000, max_risk=0.05
        # )
        # logger.info("Updating stocks...")
        # trader.update_stocks(symbols_df, period=1, days=7, force=True)
        # logger.info("Evaluating stocks...")
        # symbol_analysis = trader.evaluate_stocks(
        #     date=(datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d"),
        #     threshold=40,
        # )
        # logger.info(symbol_analysis)
        # chosen_symbols = []
        # for symbol in symbol_analysis.keys():
        #     logger.info(f"{symbol}: {symbol_analysis[symbol]['Score']}")
        #     if symbol_analysis[symbol]['Score'] > 1:
        #         chosen_symbols.append(symbol)
        chosen_symbols = ["AUDUSD", "GBPUSD", "EURUSD"]
        trader.backTest(symbols=chosen_symbols)
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
