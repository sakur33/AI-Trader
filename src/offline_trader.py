from xAPIConnector import *
from trader_utils import *
import logging
from trading_accounts import Trader
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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
    handlers=logging.FileHandler(f"{logs_path}python_trader.log"),
)
logger = logging.getLogger()


def main():
    trader = Trader(name="Test-trader", capital=1000, max_risk=0.05, trader_type="FX")

    loginResponse = trader.client.execute(
        loginCommand(userId=trader.user, password=trader.passw)
    )
    status = loginResponse["status"]
    print(f"Login Response: {status}")

    # check if user logged in correctly
    if loginResponse["status"] == False:
        error_code = loginResponse["errorCode"]
        print(f"Login failed. Error code: {error_code}")
        return

    # get ssId from login response
    ssid = loginResponse["streamSessionId"]
    trader.ssid = ssid

    # TODO to run once the market is closed

    commandResponse = trader.client.commandExecute("getAllSymbols")

    if commandResponse["status"] == False:
        error_code = commandResponse["errorCode"]
        print(f"Login failed. Error code: {error_code}")
        symbols_df = None
    else:
        symbols_df = return_as_df(commandResponse["returnData"])

    trader.insert_symbols(symbols_df)
    symbols_df = trader.look_for_suitable_symbols_v1(symbols_df)
    trader.update_stocks(symbols_df, period=1, days=30)

    trader.evaluate_stocks()


if __name__ == "__main__":
    main()
