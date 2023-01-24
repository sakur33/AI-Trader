from xAPIConnector import *
from trader_utils import *
import logging
from trading_accounts import Trader
from trader_db_utils import *
import time
import os


today = get_today()
todayms = get_today_ms()
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
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s : %(levelname)s : %(threadName)s : %(name)s %(message)s",
    filename=f"{logs_path}{__name__}.log",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(threadName)s : %(name)s %(message)s"
)
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logger = logging.getLogger(__name__)
logger.info(f"{__name__}")


def main():
    trader = Trader(
        name="TestTrader",
        capital=1000,
        max_risk=0.05,
        trader_type="FX",
    )

    symbols_df = trader.client.commandExecute("getAllSymbols")
    insert_symbols(symbols_df)
    symbols_df = trader.look_for_suitable_symbols_v1(symbols_df)
    trader.update_stocks(symbols_df, period=1, days=14)

    trader.evaluate_stocks(
        params={
            "repetitions": 500,
            "short_ma": [5, 60, 5],
            "long_ma": [500, 3500, 500],
            "min_angle": [0, 40, 5],
            "train_period": 6 / 7,
            "test_period": 1 / 7,
            "out": [0.01, 0.5, 12],
        },
        verbose=1,
        show=False,
    )


if __name__ == "__main__":
    main()
