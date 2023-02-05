import os
import threading
import pandas as pd
from xAPIConnector import *
from trader_utils import *
from trader_db_utils import *
from trader_api_utils import *
from ssl import SSLError
from logger_settings import setup_logging
import logging

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


if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")

logger = logging.getLogger(__name__)
logger = setup_logging(logger, logs_path, __name__, console_debug=True)


class ApiSessionManager:
    def __init__(self, user, passw, clock) -> None:

        self.user = user
        self.passw = passw
        self.client = None
        self.stream_clients = {}
        self.session = None
        self.symbol = None
        self.cllast_message_time = time.time()
        self.clock = clock

    def set_apiClient(self):
        client = APIClient()
        loginResponse = self.login_client(client)
        self.client = client
        self.ssid = loginResponse["streamSessionId"]

    def login_client(self, client):
        self.clock.wait_clock()
        loginResponse = client.execute(
            loginCommand(userId=self.user, password=self.passw)
        )
        status = loginResponse["status"]
        if status:
            logger.info(f"Login successful")
        else:
            error_code = loginResponse["errorCode"]
            logger.info(f"Login error | {error_code}")
            quit()
        return loginResponse

    def set_streamClient(self, symbol, tick=False, candle=True, trade=True, queue=None):
        self.stream_clients = {}
        self.symbol = symbol

        if tick:
            stream_client = APIStreamClient(
                ssId=self.ssid,
                tickFun=self.tick_processor,
            )
            stream_client.subscribePrice(symbol, minArrivalTime=5000, maxLevel=3)
            self.stream_clients["tick"] = stream_client

        if candle:
            stream_client = APIStreamClient(
                ssId=self.ssid,
                candleFun=self.candle_processor,
            )
            stream_client.subscribeCandle(symbol)
            self.stream_clients["candle"] = stream_client

        if trade:
            stream_client = APIStreamClient(
                ssId=self.ssid,
                tradeFun=self.trade_processor,
            )
            stream_client.subscribeTrades()
            self.stream_clients["trade"] = stream_client

        if queue is not None:
            self.tick_queue = queue

    def set_session(self, session):
        self.session = session

    def tick_processor(self, msg):
        tick = msg["data"]
        logger.debug(f"TICK:{msg}")
        if self.session is not None:
            self.tick_queue.put(tick)
        inserTick = threading.Thread(target=insert_tick, args=(tick,))
        inserTick.setDaemon(True)
        inserTick.start()

    def candle_processor(self, msg):
        candle = msg["data"]
        logger.debug(f"CANDLE:{msg}")
        inserCandle = threading.Thread(target=insert_candle, args=(candle,))
        inserCandle.setDaemon(True)
        inserCandle.start()
        if self.session is not None:
            session_step = threading.Thread(target=self.session.step())
            session_step.start()

    def trade_processor(self, msg):
        trade = msg["data"]
        logger.debug(f"TRADE:{msg}")
        inserTrade = threading.Thread(target=insert_trade, args=(trade,))
        inserTrade.setDaemon(True)
        inserTrade.start()

    def store_past_candles(self, symbol, start_date, period=1):
        CHART_RANGE_INFO_RECORD = {
            "period": period,
            "start": date_to_xtb_time(start_date),
            "symbol": symbol,
        }
        self.clock.wait_clock()
        commandResponse = self.client.commandExecute(
            "getChartLastRequest", arguments={"info": CHART_RANGE_INFO_RECORD}
        )
        if commandResponse["status"] == False:
            error_code = commandResponse["errorCode"]
            logger.debug(f"Login failed. Error code: {error_code}")

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
                logger.debug(f"Symbol {symbol} did not return candles")

    def get_symbols(self):
        logger.debug("get_symbols")
        self.clock.wait_clock()
        symbols = get_symbol_today()
        if not isinstance(symbols, pd.DataFrame):
            commandResponse = self.client.commandExecute("getAllSymbols")
            symbols_df = return_as_df(commandResponse["returnData"])
            insert_symbols(symbols_df)
        else:
            symbols_df = symbols
        return symbols_df

    def get_ticks(self, timestamp, symbol):
        try:
            self.clock.wait_clock()
            tick_df = self.client.commandExecute(
                commandName="getTickPrices",
                arguments={
                    "level": 0,
                    "symbols": [symbol],
                    "timestamp": timestamp,
                },
            )
            tick_df = return_as_df(tick_df["returnData"]["quotations"])
        except Exception as e:
            logger.debug(f"Exception at get ticks: {e}")
            self.relaunch_system()
            tick_df = self.get_ticks(timestamp, symbol)

        return tick_df

    def get_trades(self):
        self.clock.wait_clock()
        commandResponse = self.client.commandExecute(
            "getTrades", arguments={"openedOnly": True}
        )
        if commandResponse["status"] == True:
            trade_records = commandResponse["returnData"]
            return trade_records
        else:
            return None

    def get_trade(self, opened=True, order_n=None):
        while 1:
            try:
                if opened:
                    if order_n is not None:
                        if self.is_trade_closed(order_n):
                            return None
                    self.clock.wait_clock()
                    commandResponse = self.client.commandExecute(
                        "getTrades", arguments={"openedOnly": True}
                    )
                    trade_records = commandResponse["returnData"]

                    if isinstance(trade_records, list):
                        for record in trade_records:
                            if record["order2"] == order_n:
                                return record
                    else:
                        if record["order2"] == order_n:
                            return record
                else:
                    commandResponse = self.client.commandExecute(
                        "getTradesHistory", arguments={"end": 0, "start": 0}
                    )
                    trade_records = commandResponse["returnData"]

                    if isinstance(trade_records, list):
                        for record in trade_records:
                            if record["order2"] == order_n:
                                return record
                    else:
                        if record["order2"] == order_n:
                            return record
            except Exception as e:
                logger.debug(f"Exception at get trade: {e}")
                self.relaunch_system()
                record = self.get_trade(opened=True, order_n=order_n)
                return record

    def get_trade_status(self, order_n):
        self.clock.wait_clock()
        commandResponse = self.client.commandExecute(
            "tradeTransactionStatus", arguments={"order": order_n}
        )
        if commandResponse["status"] == True:
            if commandResponse["returnData"]["requestStatus"] == 3:
                return True
        return False

    def get_max_values(self, symbol, long_ma):
        try:
            self.clock.wait_clock()
            commandResponse = self.client.commandExecute(
                commandName="getSymbol", arguments={"symbol": symbol}
            )
            min_close = commandResponse["returnData"]["low"]
            max_close = commandResponse["returnData"]["high"]
            min_ctm = 0
            max_ctm = long_ma
        except Exception as e:
            logger.debug(f"Exception at get max values: {e}")
            self.relaunch_system()
            min_ctm, max_ctm, min_close, max_close = self.get_max_values(
                symbol, long_ma
            )
        return min_ctm, max_ctm, min_close, max_close

    def buy(self, name, buy_trans, c_tick, volume, sl):
        try:
            order_status = False
            while order_status == False:
                self.clock.wait_clock()
                commandResponse = self.client.commandExecute(
                    "tradeTransaction",
                    arguments={
                        "tradeTransInfo": {
                            "cmd": buy_trans,
                            "customComment": name,
                            "expiration": 0,
                            "offset": 0,
                            "order": 0,
                            "price": c_tick["bid"],
                            "sl": sl,
                            "symbol": c_tick["bid"],
                            "tp": 0.0,
                            "type": TransactionType.ORDER_OPEN,
                            "volume": volume,
                        }
                    },
                )

                returnData = commandResponse["returnData"]
                if len(returnData) == 0:
                    order_status = False
                else:
                    order_status = self.get_trade_status(order_n)
                if order_status == False:
                    if buy_trans == 0:
                        logger.info(
                            f"Attempt to buy failed | sl {sl} -> {sl * (1 - 0.01)}"
                        )
                        sl = float(np.round(sl * (1 - 0.01), 5))
                    else:
                        logger.info(
                            f"Attempt to buy failed | sl {sl} -> {sl * (1 + 0.01)}"
                        )
                        sl = float(np.round(sl * (1 + 0.01), 5))
                    logger.info(f"Attempt to buy failed | sl {sl}")
        except Exception as e:
            logger.info(f"Exception at buy position: {e}")
            self.relaunch_system()
            order_n = self.buy(name, buy_trans, c_tick, volume, sl)
        return order_n

    def sell(self, name, buy_trans, position, c_tick, volume):
        try:
            self.clock.wait_clock()
            commandResponse = self.client.commandExecute(
                "tradeTransaction",
                arguments={
                    "tradeTransInfo": {
                        "cmd": buy_trans,
                        "customComment": name,
                        "expiration": 0,
                        "offset": 0,
                        "order": position["order"],
                        "price": c_tick["ask"],
                        "sl": 0.0,
                        "symbol": c_tick["symbol"],
                        "tp": 0.0,
                        "type": TransactionType.ORDER_CLOSE,
                        "volume": volume,
                    }
                },
            )
            sell_order_n = commandResponse["returnData"]["order"]
        except Exception as e:
            logger.info(f"Exception at sell position: {e}")
            self.relaunch_system()
            sell_order_n = self.sell(name, buy_trans, position, c_tick, volume)

        return sell_order_n

    def get_balance(self):
        try:
            self.clock.wait_clock()
            commandResponse = self.client.commandExecute(
                "getTrades", arguments={"openedOnly": True}
            )
            balance = commandResponse["returnData"]
        except Exception as e:
            logger.info(f"Exception at get trade: {e}")
            self.relaunch_system()
            balance = self.get_balance()
        return balance

    def relaunch_system(self):
        logger.info("\n**********\nSYSTEM RELAUNCH\n**********\n")
        if self.client is not None:
            self.client.disconnect()
        list_of_streams = list(self.stream_clients.keys())
        if len(list_of_streams) > 0:
            for key in list_of_streams:
                self.stream_clients[key].disconnect()

        self.set_apiClient()
        self.set_streamClient(
            self.symbol,
            tick=("tick" in list_of_streams),
            candle=("candle" in list_of_streams),
            trade=("trade" in list_of_streams),
        )

    def load_missed_candles(self, symbol, long_ma):
        # TODO: get last date and load from there
        date = datetime.now() - timedelta(minutes=long_ma)
        self.store_past_candles(symbol, date)

    def is_trade_closed(self, order_n):
        return find_closed_trade(order_n)
