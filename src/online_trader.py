from xAPIConnector import *
from trader_utils import *
import logging
from trading_accounts import Trader
import time
import os
import sys


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

logger = logging.getLogger(__name__)
logger.info(f"{__name__}")


def main():
    trader = Trader(
        name="Test-trader",
        trader_name="trader2",
        capital=1000,
        max_risk=0.05,
        trader_type="FX",
    )

    params_df = trader.get_trading_params()

    trader.start_trading_sessions(params_df.iloc[0, :])
    trader.start_streaming()

    # trader.stream_client.subscribePrice(params_df.iloc[0, :]["symbol_name"])
    trader.stream_client.subscribeCandle(params_df.iloc[0, :]["symbol_name"])
    trader.stream_client.subscribeKeepAlive()

    while 1:
        pass


if __name__ == "__main__":
    main()
