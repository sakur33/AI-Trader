from xAPIConnector import (
    TransactionSide,
    TransactionType,
    JsonSocket,
    APIClient,
    APIStreamClient,
)
from utils import (
    loginCommand,
    procTickExample,
    procTradeExample,
    procProfitExample,
    procTradeStatusExample,
)
import logging
from creds import user, passw

import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
    handlers=logging.FileHandler("./logs/python_trader.log"),
)
logger = logging.getLogger()


def main():
    # enter your login credentials here
    userId = user
    password = passw

    # create & connect to RR socket
    client = APIClient()

    # connect to RR socket, login
    loginResponse = client.execute(loginCommand(userId=userId, password=password))
    logger.info(str(loginResponse))

    # check if user logged in correctly
    if loginResponse["status"] == False:
        print("Login failed. Error code: {0}".format(loginResponse["errorCode"]))
        return

    # get ssId from login response
    ssid = loginResponse["streamSessionId"]

    # second method of invoking commands
    resp = client.commandExecute("getAllSymbols")

    # create & connect to Streaming socket with given ssID
    # and functions for processing ticks, trades, profit and tradeStatus
    sclient = APIStreamClient(
        ssId=ssid,
        tickFun=procTickExample,
        tradeFun=procTradeExample,
        profitFun=procProfitExample,
        tradeStatusFun=procTradeStatusExample,
    )

    # subscribe for trades
    sclient.subscribeTrades()

    # subscribe for prices
    sclient.subscribePrices(["EURUSD", "EURGBP", "EURJPY"])

    # subscribe for profits
    sclient.subscribeProfits()

    # this is an example, make it run for 5 seconds
    time.sleep(5)

    # gracefully close streaming socket
    sclient.disconnect()

    # gracefully close RR socket
    client.disconnect()


if __name__ == "__main__":
    main()
