import os
import json
import socket
from logger_settings import *
import logging
import time
import ssl
from trader_utils import baseCommand
from threading import Thread
from custom_exceptions import LoginError, SocketError, ApiException
from trader_db_utils import *
from json import JSONDecodeError


# set to true on debug environment only
DEBUG = False

# default connection properites
DEFAULT_XAPI_ADDRESS = "xapi.xtb.com"
DEFAULT_XAPI_PORT = 5124
DEFAULT_XAPI_STREAMING_PORT = 5125

# wrapper name and version
WRAPPER_NAME = "python"
WRAPPER_VERSION = "2.5.0"

# API inter-command timeout (in ms)
API_SEND_TIMEOUT = 100

# max connection tries
API_MAX_CONN_TRIES = 3


class TransactionSide(object):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5


class TransactionType(object):
    ORDER_OPEN = 0
    ORDER_CLOSE = 2
    ORDER_MODIFY = 3
    ORDER_DELETE = 4


class JsonSocket(object):
    def __init__(
        self,
        address,
        port,
        encrypt=False,
        logger=logging.getLogger(__name__),
        exception_com=None,
    ):
        self._ssl = encrypt
        if self._ssl != True:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket = ssl.wrap_socket(sock)
        self.conn = self.socket
        self._timeout = None
        self._address = address
        self._port = port
        self._decoder = json.JSONDecoder()
        self._receivedData = ""
        self.logger = logger
        self.exception_com = exception_com

    def connect(self):
        for i in range(API_MAX_CONN_TRIES):
            try:
                self.socket.connect((self.address, self.port))
            except socket.error as msg:
                self.logger.error("SockThread Error: %s" % msg)
                time.sleep(0.25)
                continue
            # self.logger.debug("\nSocket connected")
            return True
        return False

    def _sendObj(self, obj):
        msg = json.dumps(obj)
        self._waitingSend(msg)

    def _waitingSend(self, msg):
        if self.socket:
            sent = 0
            msg = msg.encode("utf-8")
            while sent < len(msg):
                sent += self.conn.send(msg[sent:])
                self.logger.debug("Sent: " + str(msg))
                time.sleep(API_SEND_TIMEOUT / 1000)

    def _read(self, bytesSize=20000):
        while True:
            if not self.socket:
                self.logger.info("socket connection broken")
                self.exception_com.put(RuntimeError("socket connection broken"))
                quit()
            char = self.conn.recv(bytesSize).decode()
            if char == "":
                self.logger.info("Socket disconnection")
                self.exception_com.put(RuntimeError("Socket disconnection"))
                self.close()
            self._receivedData += char
            # if self._receivedData[-1] == "}":
            try:
                (resp, size) = self._decoder.raw_decode(self._receivedData)
                if size == len(self._receivedData):
                    self._receivedData = ""
                    break
                elif size < len(self._receivedData):
                    self._receivedData = self._receivedData[size:].strip()
                    break
            except (ValueError, JSONDecodeError) as e:
                continue
        self.logger.debug("Received: " + str(resp))
        return resp

    def _readObj(self):
        msg = self._read()
        return msg

    def close(self):
        self.logger.debug("Closing socket")
        self._closeSocket()
        if self.socket is not self.conn:
            self.logger.debug("Closing connection socket")
            self._closeConnection()

    def _closeSocket(self):
        self.socket.close()

    def _closeConnection(self):
        self.conn.close()

    def _get_timeout(self):
        return self._timeout

    def _set_timeout(self, timeout):
        self._timeout = timeout
        self.socket.settimeout(timeout)

    def _get_address(self):
        return self._address

    def _set_address(self, address):
        pass

    def _get_port(self):
        return self._port

    def _set_port(self, port):
        pass

    def _get_encrypt(self):
        return self._ssl

    def _set_encrypt(self, encrypt):
        pass

    def _keep_alive(self):
        pass

    timeout = property(_get_timeout, _set_timeout, doc="Get/set the socket timeout")
    address = property(
        _get_address, _set_address, doc="read only property socket address"
    )
    port = property(_get_port, _set_port, doc="read only property socket port")
    encrypt = property(_get_encrypt, _set_encrypt, doc="read only property socket port")


class APIClient(JsonSocket):
    def __init__(
        self,
        address=DEFAULT_XAPI_ADDRESS,
        port=DEFAULT_XAPI_PORT,
        encrypt=True,
        logger=None,
        exception_com=None,
    ):
        super(APIClient, self).__init__(address, port, encrypt, logger, exception_com)
        self.LAST_COMMAND_EXECUTED = time.time()
        self.kill_ping = False
        self.pingFun = Thread(target=self.ping)
        self.pingFun.setDaemon(True)
        self.pingFun.start()
        if not self.connect():
            raise Exception(
                "Cannot connect to "
                + address
                + ":"
                + str(port)
                + " after "
                + str(API_MAX_CONN_TRIES)
                + " retries"
            )

    def execute(self, dictionary):
        self._sendObj(dictionary)
        return self._readObj()

    def disconnect(self):
        self.close()

    def commandExecute(self, commandName, arguments=None):
        try:
            commandResponse = self.execute(baseCommand(commandName, arguments))
            if commandResponse["status"] == False:
                error_code = commandResponse["errorCode"]
                error_description = commandResponse["errorDescr"]
                self.logger.error(f"\nLogin failed. Error code: {error_code}")
                raise LoginError(
                    f"Error code: {error_code} | Error description: {error_description}"
                )
            self.LAST_COMMAND_EXECUTED = time.time()
            return commandResponse
        except Exception as e:
            self.disconnect()
            self.exception_com.put(SocketError(e))
            quit()

    def ping(self):
        while not self.kill_ping:
            if self.socket:
                try:
                    if (time.time() - self.LAST_COMMAND_EXECUTED) > (60):
                        commandResponse = self.execute(
                            baseCommand(
                                commandName="ping",
                            )
                        )
                        self.LAST_COMMAND_EXECUTED = time.time()
                        self.logger.debug("ping")
                except Exception as e:
                    self.kill_ping = 1
                    self.exception_com.put(SocketError(e))
                    self.disconnect()

    def login(self, user, passw, appName=""):
        try:
            commandResponse = self.commandExecute(
                commandName="login",
                arguments=dict(userId=user, password=passw, appName=appName),
            )
            self.logger.info("Login successfull")
            return commandResponse["streamSessionId"]
        except Exception as e:
            self.logger.error(f"{e}")
            self.exception_com.put(SocketError(e))
            self.disconnect()


class APIStreamClient(JsonSocket):
    def __init__(
        self,
        address=DEFAULT_XAPI_ADDRESS,
        port=DEFAULT_XAPI_STREAMING_PORT,
        encrypt=True,
        ssId=None,
        tickFun=None,
        tradeFun=None,
        balanceFun=None,
        tradeStatusFun=None,
        profitFun=None,
        newsFun=None,
        candleFun=None,
        keepAliveFun=None,
        candle_queue=None,
        tick_queue=None,
        trade_queue=None,
        profit_queue=None,
        db_obj=None,
        logger=None,
        exception_com=None,
    ):
        super(APIStreamClient, self).__init__(
            address, port, encrypt, logger, exception_com
        )
        self._ssId = ssId

        self._tickFun = self.tick_processor
        self._tradeFun = self.trade_processor
        self._balanceFun = balanceFun
        self._tradeStatusFun = tradeStatusFun
        self._profitFun = self.profit_processor
        self._newsFun = newsFun
        self._candleFun = self.candle_processor
        self.last_tick_time = None
        self.candle_queue = candle_queue
        self.tick_queue = tick_queue
        self.trade_queue = trade_queue
        self.profit_queue = profit_queue
        self.ticks = []
        self.candles = []
        self.trades = []

        if not self.connect():
            raise Exception(
                "Cannot connect to streaming on "
                + address
                + ":"
                + str(port)
                + " after "
                + str(API_MAX_CONN_TRIES)
                + " retries"
            )
        self.DB = db_obj
        self._running = True
        self._t = Thread(target=self._readStream, args=())
        self._t.setDaemon(True)
        self._t.start()
        self.subscribeKeepAlive()

    def _readStream(self):
        try:
            while self._running:
                msg = self._readObj()
                if msg["command"] == "tickPrices":
                    self._tickFun(msg)
                elif msg["command"] == "candle":
                    self._candleFun(msg)
                elif msg["command"] == "trade":
                    self._tradeFun(msg)
                elif msg["command"] == "balance":
                    self._balanceFun(msg)
                elif msg["command"] == "tradeStatus":
                    self._tradeStatusFun(msg)
                elif msg["command"] == "profit":
                    self._profitFun(msg)
                elif msg["command"] == "news":
                    self._newsFun(msg)
                elif msg["command"] == "keepAlive":
                    # self.logger.info("KeepAlive")
                    pass
        except Exception as e:
            self.logger.info(f"Exception at read_stream | {e}")
            self.exception_com.put(e)
            self._running = False
            self.disconnect()

    def disconnect(self):
        self._running = False
        self.close()

    def execute(self, dictionary):
        self._sendObj(dictionary)

    def subscribePrice(self, symbol, minArrivalTime=10 * 1000, maxLevel=1):
        self.execute(
            dict(
                command="getTickPrices",
                symbol=symbol,
                streamSessionId=self._ssId,
                minArrivalTime=minArrivalTime,
                maxLevel=maxLevel,
            )
        )

    def subscribeMultiplePrices(self, symbols, minArrivalTime=1):
        for symbolX in symbols:
            self.subscribePrice(symbolX, minArrivalTime)

    def subscribeTrades(self):
        self.execute(dict(command="getTrades", streamSessionId=self._ssId))

    def subscribeBalance(self):
        self.execute(dict(command="getBalance", streamSessionId=self._ssId))

    def subscribeTradeStatus(self):
        self.execute(dict(command="getTradeStatus", streamSessionId=self._ssId))

    def subscribeProfits(self):
        self.execute(dict(command="getProfits", streamSessionId=self._ssId))

    def subscribeNews(self):
        self.execute(dict(command="getNews", streamSessionId=self._ssId))

    def subscribeMultipleCandles(self, symbols):
        for symbol in symbols:
            self.subscribeCandles(symbol)

    def subscribeCandle(self, symbol):
        self.execute(
            dict(command="getCandles", streamSessionId=self._ssId, symbol=symbol)
        )

    def subscribeKeepAlive(self):
        self.execute(dict(command="getKeepAlive", streamSessionId=self._ssId))

    def unsubscribePrice(self, symbol):
        self.execute(
            dict(command="stopTickPrices", symbol=symbol, streamSessionId=self._ssId)
        )

    def unsubscribeMultiplePrices(self, symbols):
        for symbolX in symbols:
            self.unsubscribePrice(symbolX)

    def unsubscribeTrades(self):
        self.execute(dict(command="stopTrades", streamSessionId=self._ssId))

    def unsubscribeBalance(self):
        self.execute(dict(command="stopBalance", streamSessionId=self._ssId))

    def unsubscribeTradeStatus(self):
        self.execute(dict(command="stopTradeStatus", streamSessionId=self._ssId))

    def unsubscribeProfits(self):
        self.execute(dict(command="stopProfits", streamSessionId=self._ssId))

    def unsubscribeNews(self):
        self.execute(dict(command="stopNews", streamSessionId=self._ssId))

    def unsubscribeCandles(self, symbol):
        self.execute(dict(command="stopCandles", symbol=symbol))

    def unsubscribeMultipleCandles(self, symbols):
        for symbolX in symbols:
            self.unsubscribeCandles(symbolX)

    def tick_processor(self, msg):
        try:
            tick = msg["data"]
            self.logger.debug(f"TICK:{msg}")
            self.tick_queue.put(tick)
            if len(self.ticks) < 10:
                self.ticks.append(tick)
            else:
                self.ticks.append(tick)
                inserTick = threading.Thread(target=self.DB.insert_tick, args=(tick,))
                inserTick.start()
                self.ticks = []
        except Exception as e:
            self.exception_com.put(ApiException(e))
            self._running = False

    def candle_processor(self, msg):
        try:
            candle = msg["data"]
            self.logger.debug(f"CANDLE:{msg}")
            self.candle_queue.put(candle)
            # inserCandle = threading.Thread(target=self.DB.insert_candle, args=(candle,))
            # inserCandle.start()
        except Exception as e:
            self.exception_com.put(ApiException(e))
            self._running = False

    def trade_processor(self, msg):
        try:
            trade = msg["data"]
            self.logger.debug(f"TRADE:{msg}")
            self.trade_queue.put(trade)
            # if len(self.trades) < 10:
            #     self.trades.append(trade)
            # else:
            #     self.trades.append(trade)
            #     inserTrade = threading.Thread(
            #         target=self.DB.insert_trade, args=(self.trades,)
            #     )
            #     inserTrade.start()
            #     self.trades = []

        except Exception as e:
            self.exception_com.put(ApiException(e))
            self._running = False

    def profit_processor(self, msg):
        try:
            profit = msg["data"]
            self.logger.debug(f"PROFIT:{msg}")
            self.profit_queue.put(profit)
        except Exception as e:
            self.exception_com.put(ApiException(e))
            self._running = False
