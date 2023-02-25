from xAPIConnector import *
from trader_utils import *
from trader_db_utils import *
from logger_settings import *

import logging
from trading_accounts import Trader
import time
import os
import sys
import argparse
import logging
from multiprocessing import SimpleQueue
from connection_clock import Clock

parser = argparse.ArgumentParser()
parser.add_argument("--trader_name", type=str, help="Trader name")
parser.add_argument("--test", type=int, default=0, help="Override trading params")
args = parser.parse_args()

if args.trader_name:
    trader_name = args.trader_name
else:
    trader_name = "crossover"

if args.test:
    test = True
else:
    test = False

is_debug_run = os.environ.get("is_debug_run")
if is_debug_run:
    test = True

if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")

logger = logging.getLogger(__name__)
logger = setup_logging(logger, logs_path, __name__, console_debug=True)


def main_logic(row):
    logger.info("Start Online trading")
    tick_queue = SimpleQueue()
    candle_queue = SimpleQueue()
    clock = Clock()
    trader = Trader(
        name=f"trader68709:{row['symbol']}:{trader_name}",
        capital=1000,
        max_risk=0.05,
        trader_type="FX",
        tick_queue=tick_queue,
        clock=clock,
        candle_queue=candle_queue,
    )

    trader.apiSession.set_streamClient(
        row["symbol"],
        tick=True,
        candle=True,
        trade=True,
        tick_queue=tick_queue,
        candle_queue=candle_queue,
    )
    trader.start_trading_session(row, test=True)
    while 1:
        pass


def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)


def main():
    logger.info("ONLINE TRADER")
    params_df = get_trading_params()
    for index, row in params_df.iterrows():
        try:
            main_logic(row)
        except Exception as e:
            logger.info(f"Exceptions being catched: {e}")
            restart_program()


if __name__ == "__main__":
    main()
