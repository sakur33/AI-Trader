import os
from logger_settings import *

from trading_accounts import *
from trader_db_utils import *
from trader_utils import *
from trader_api_utils import *
from xAPIConnector import *
from connection_clock import Clock
from multiprocessing import SimpleQueue


if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")

logger = logging.getLogger(__name__)
logger = setup_logging(logger, logs_path, __name__, console_debug=True)


def main():
    tick_queue = SimpleQueue()
    clock = Clock()
    trader = Trader(
        name="trader68709",
        capital=1000,
        max_risk=0.05,
        trader_type="FX",
        tick_queue=tick_queue,
        clock=clock,
    )
    symbols_df = trader.apiSession.get_symbols()
    symbols_df = trader.look_for_suitable_symbols_v1(symbols_df)
    trader.update_stocks(symbols_df, period=1, days=14)

    trader.evaluate_stocks(
        date=(datetime.today() - timedelta(days=9)).strftime("%Y-%m-%d"),
        threshold=40,
    )


if __name__ == "__main__":
    main()
