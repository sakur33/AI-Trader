import os
import json
import socket
from logger_settings import setup_logging
import logging
import time
import ssl
from trader_utils import baseCommand
from threading import Thread

# set to true on debug environment only
DEBUG = False

# default connection properites
DEFAULT_XAPI_ADDRESS = "xapi.xtb.com"
DEFAULT_XAPI_PORT = 5124
DEFUALT_XAPI_STREAMING_PORT = 5125

# wrapper name and version
WRAPPER_NAME = "python"
WRAPPER_VERSION = "2.5.0"

# API inter-command timeout (in ms)
API_SEND_TIMEOUT = 100

# max connection tries
API_MAX_CONN_TRIES = 3

# logger properties
curr_path = os.path.dirname(os.path.realpath(__file__))
logs_path = curr_path + "../../logs/"
if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")

logger = logging.getLogger(__name__)
logger = setup_logging(logger, logs_path, __name__, console_debug=True)


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
    def __init__(self, address, port, encrypt=False):
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

    def connect(self):
        for i in range(API_MAX_CONN_TRIES):
            try:
                self.socket.connect((self.address, self.port))
            except socket.error as msg:
                logger.error("SockThread Error: %s" % msg)
                time.sleep(0.25)
                continue
            # logger.debug("\nSocket connected")
            return True
        return False

    def relaunch_socket(self):
        logger.info("\n***********\nSOCKET RELAUNCH\n***********\n")
        self.__init__(self._address, self._port, encrypt=True)

    def _sendObj(self, obj):
        msg = json.dumps(obj)
        self._waitingSend(msg)

    def _waitingSend(self, msg):
        if self.socket:
            total_sent = 0
            msg = msg.encode("utf-8")

            while total_sent < len(msg):
                try:
                    sent = self.conn.send(msg[total_sent:])
                    if sent == 0:
                        self.relaunch_socket()
                    total_sent = total_sent + sent
                    # logger.debug("\nSent: " + str(msg))
                    time.sleep(API_SEND_TIMEOUT / 1000)
                except (OSError, Exception) as e:
                    self.relaunch_socket()

    def _read(self, bytesSize=4096):
        if not self.socket:
            logger.error("Socket connection broken")
            raise RuntimeError("socket connection broken")
        while True:
            try:
                char = self.conn.recv(bytesSize).decode()
                self._receivedData += char
                (resp, size) = self._decoder.raw_decode(self._receivedData)
                if size == len(self._receivedData):
                    self._receivedData = ""
                    break
                elif size < len(self._receivedData):
                    self._receivedData = self._receivedData[size:].strip()
                    break
            except ValueError as e:
                # logger.error(f"ValueError: {e}")
                continue
            except (OSError, Exception) as e:
                self.relaunch_socket()
        return resp

    def _readObj(self):
        msg = self._read()
        return msg

    def close(self):
        logger.debug("Closing socket")
        self._closeSocket()
        if self.socket is not self.conn:
            logger.debug("Closing connection socket")
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
        self, address=DEFAULT_XAPI_ADDRESS, port=DEFAULT_XAPI_PORT, encrypt=True
    ):
        super(APIClient, self).__init__(address, port, encrypt)
        self.LAST_COMMAND_EXECUTED = time.time()
        pingFun = Thread(target=self.ping)
        pingFun.setDaemon(True)
        pingFun.start()
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
        self.LAST_COMMAND_EXECUTED = time.time()
        commandResponse = self.execute(baseCommand(commandName, arguments))
        if commandResponse["status"] == False:
            error_code = commandResponse["errorCode"]
            logger.error(f"\nLogin failed. Error code: {error_code}")

        return commandResponse

    def ping(self):
        while True:
            if (time.time() - self.LAST_COMMAND_EXECUTED) > (60):
                commandResponse = self.commandExecute(
                    commandName="ping",
                )
                logger.debug("ping")


class APIStreamClient(JsonSocket):
    def __init__(
        self,
        address=DEFAULT_XAPI_ADDRESS,
        port=DEFUALT_XAPI_STREAMING_PORT,
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
    ):
        super(APIStreamClient, self).__init__(address, port, encrypt)
        self._ssId = ssId

        self._tickFun = tickFun
        self._tradeFun = tradeFun
        self._balanceFun = balanceFun
        self._tradeStatusFun = tradeStatusFun
        self._profitFun = profitFun
        self._newsFun = newsFun
        self._candleFun = candleFun
        self.last_tick_time = None

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

        self._running = True
        self._t = Thread(target=self._readStream, args=())
        self._t.setDaemon(True)
        self._t.start()
        self.subscribeKeepAlive()

    def _readStream(self):
        while self._running:
            msg = self._readObj()
            if msg["command"] == "tickPrices":
                self._tickFun(msg)
            elif msg["command"] == "candle":
                self._candleFun(msg)
            elif msg["command"] == "trade":
                logger.debug("trade-call")
                self._tradeFun(msg)
            elif msg["command"] == "balance":
                self._balanceFun(msg)
            elif msg["command"] == "tradeStatus":
                self._tradeStatusFun(msg)
            elif msg["command"] == "profit":
                self._profitFun(msg)
            elif msg["command"] == "news":
                self._newsFun(msg)

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
