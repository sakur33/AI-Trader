from xAPIConnector import *
from trading_systems import CrossTrader, BacktestCrossTrader
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
parser.add_argument("--test", type=int, default=0, help="Override trading params")
args = parser.parse_args()

if args.trader_name:
    TRADER_NAME = args.trader_name
else:
    TRADER_NAME = "tests"

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
    if not test:
        os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)


def main():
    global TRADER_NAME
    custom_logger = CustomLogger(test_name=TRADER_NAME)
    logger = logging.getLogger(__name__)
    logger = custom_logger.setup_logging(logger, TRADER_NAME, console_debug=True)
    weekno = datetime.today().weekday()
    if weekno > -1:
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
        logger.info(f"ONLINE TRADER")
        params_df = trader.DB.get_trading_params()
        iterator = params_df.iterrows()
        index, row = next(iterator)
        trader.name = f"trader68709:{row['symbol']}:{TRADER_NAME}"
        trader.test_name = TRADER_NAME
        trader.CLIENT = apiClient
        trader.SYMBOl = "AUDUSD"
        trader.VOLUME = 0.01

        if test:
            trader.short_ma = 1
            trader.long_ma = 10
        else:
            trader.short_ma = 5
            trader.long_ma = 200

        date = datetime.now() - timedelta(
            minutes=int(np.max([trader.long_ma, 200]) + 3)
        )
        trader.store_past_candles(row["symbol"], date)

        trader.SYMBOL_INFO = trader.DB.get_symbol_info(row["symbol"])
        trader.start_stream_clients(row["symbol"], tick=True, candle=True, trade=True)
        trader.set_last_candle()
        logger.info("SYSTEM STARTS")
        while True:
            try:
                trader.set_last_tick()
                trader.set_last_candle(margin=200)
                trader.update_trades()
                trader.evaluate_risks()
                trader.step()
                # trader.update_stop_loss()
                # TODO: Calculate left balance
                # TODO: Gather closed trades
                # TODO: Improve positions
                time.sleep(5)
            except Exception as e:
                logger.info(f"Exception in main loop | {e}")
                restart_program()
                quit()
            pass


if __name__ == "__main__":
    main()
