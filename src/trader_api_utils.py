import os
import threading
import pandas as pd
from xAPIConnector import *
from trader_utils import *
from trader_db_utils import *
from trader_api_utils import *

curr_path = os.path.dirname(os.path.realpath(__file__))
logs_path = curr_path + "../../logs/"
if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")
logger = logging.getLogger(__name__)

today = get_today()
todayms, today_int = get_today_ms()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../data/"
symbol_path = curr_path + "../../symbols/"
cluster_path = curr_path + "../../clusters/"
model_path = curr_path + "../../model/"
result_path = curr_path + "../../result/"
docs_path = curr_path + "../../docs/"
database_path = curr_path + "../../database/"
logs_path = curr_path + "../../logs/"


class ApiSessionManager:
    def __init__(self, user, passw) -> None:
        self.user = user
        self.passw = passw

    def set_apiClient(self):
        client = APIClient()
        loginResponse = self.login_client(client)
        self.client = client
        self.ssid = loginResponse["streamSessionId"]
        pingFun = threading.Thread(target=self.ping_client)
        pingFun.setDaemon(True)
        pingFun.start()

    def login_client(self, client):
        loginResponse = client.execute(
            loginCommand(userId=self.user, password=self.passw)
        )
        status = loginResponse["status"]
        if status:
            logging.info(f"Login successful")
        else:
            error_code = loginResponse["errorCode"]
            logging.info(f"Login error | {error_code}")
            quit()
        return loginResponse

    def ping_client(self):
        while True:
            commandResponse = self.client.commandExecute(
                commandName="ping",
            )
            time.sleep(60 * 5)

    def set_streamClient(self, symbol=None):
        stream_client = APIStreamClient(
            ssId=self.ssid, tickFun=self.tick_processor, candleFun=self.candle_processor
        )
        if symbol:
            stream_client.subscribeCandle(symbol)
            stream_client.subscribePrice(symbol)
        self.streamClient = stream_client

    def tick_processor(self, msg):
        tick = msg["data"]
        tick["timestamp"] = xtb_time_to_date(int(tick["timestamp"]), local_tz=True)
        inserTick = threading.Thread(target=insert_tick, args=(tick,))
        inserTick.start()

    def candle_processor(self, msg):
        candle = msg["data"]
        candle["ctmString"] = (
            pd.to_datetime(candle["ctmString"], format="%b %d, %Y, %I:%M:%S %p")
            .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            .strftime("%Y-%m-%d %H:%M:%S.%f")
        )
        inserCandle = threading.Thread(target=insert_candle, args=(candle,))
        inserCandle.start()

    def store_past_candles(self, symbol, start_date, period=1):
        CHART_RANGE_INFO_RECORD = {
            "period": period,
            "start": date_to_xtb_time(start_date),
            "symbol": symbol,
        }
        commandResponse = self.client.commandExecute(
            "getChartLastRequest",
            arguments={"info": CHART_RANGE_INFO_RECORD},
            return_df=False,
        )
        if commandResponse["status"] == False:
            error_code = commandResponse["errorCode"]
            logger.info(f"Login failed. Error code: {error_code}")
        else:
            returnData = commandResponse["returnData"]
            digits = returnData["digits"]
            candles = return_as_df(returnData["rateInfos"])
            if not candles is None:
                candles = cast_candles_to_types(candles, digits, dates=True)
                candles = adapt_data(candles)
                candles["symbol"] = symbol
                for index, row in candles.iterrows():
                    insert_candle(row.to_dict())
            else:
                logger.info(f"Symbol {symbol} did not return candles")

    def get_symbols(self):
        commandResponse = self.client.commandExecute("getAllSymbols")
        symbols_df = return_as_df(commandResponse["returnData"])
        insert_symbols(symbols_df)
        return symbols_df
