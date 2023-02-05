from xAPIConnector import *
from trader_utils import *
from trader_db_utils import *
from logger_settings import setup_logging

import logging
from trading_accounts import Trader
import time
import os
import sys
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--trader_name", type=str, help="Trader name")
parser.add_argument("--test", help="Override trading params")
args = parser.parse_args()
if args.trader_name:
    trader_name = args.trader_name
if args.test:
    test = False
else:
    test = True

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
logger = setup_logging(logger, logs_path, __name__, console_debug=True)


def main():
    logger.info("ONLINE TRADER")
    params_df = get_trading_params()

    for index, row in params_df.iterrows():
        logger.info("Start Online trading")
        trader = Trader(
            name=f"trader68709:{row['symbol']}:{trader_name}",
            capital=1000,
            max_risk=0.05,
            trader_type="FX",
        )
        trader.apiSession.set_streamClient(
            row["symbol"], tick=True, candle=True, trade=True
        )
        trader.start_trading_session(row, test=test)
        while 1:
            pass


if __name__ == "__main__":
    main()
