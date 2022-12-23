from xAPIConnector import *
from utils import *
import logging
from trading_accounts import Trader
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
    handlers=logging.FileHandler("./logs/python_trader.log"),
)
logger = logging.getLogger()


def main():
    trader = Trader(name="Test-trader", capital=250, max_risk=0.05, trade_type="STC")

    loginResponse = trader.client.execute(
        loginCommand(userId=trader.user, password=trader.passw)
    )
    print(f"Login Response: {str(loginResponse)}")

    # check if user logged in correctly
    if loginResponse["status"] == False:
        error_code = loginResponse["errorCode"]
        print(f"Login failed. Error code: {error_code}")
        return

    # get ssId from login response
    ssid = loginResponse["streamSessionId"]
    trader.ssid = ssid

    # second method of invoking commands
    resp = trader.client.commandExecute("getAllSymbols")

    # create & connect to Streaming socket with given ssID
    # and functions for processing ticks, trades, profit and tradeStatus
    trader.start_streaming()

    # subscribe for trades
    trader.stream_client.subscribeTrades()

    # subscribe for prices
    trader.stream_client.subscribePrices(["EURUSD", "EURGBP", "EURJPY"])

    # subscribe for profits
    trader.stream_client.subscribeProfits()

    # this is an example, make it run for 5 seconds
    time.sleep(5)

    # gracefully close streaming socket
    trader.stream_client.disconnect()

    # gracefully close RR socket
    trader.client.disconnect()


if __name__ == "__main__":
    main()
