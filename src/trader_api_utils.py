import os
import pandas as pd
from trader_utils import *
from xAPIConnector import *
from creds import creds
import threading
from trader_db_utils import *
from trader_api_utils import *
from tqdm import tqdm

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
            time.sleep(60 * 5)
            try:
                commandResponse = self.client.commandExecute(
                    "ping",
                    return_df=False,
                )
            except Exception as e:
                logger.info(f"Exception at ping: {e}")

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
