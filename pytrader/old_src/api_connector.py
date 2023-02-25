import os
import json
import socket
from logger_settings import *
import logging
import time
import ssl
from trader_utils import baseCommand
from threading import Thread
from custom_exceptions import LoginError, SocketError
from json import JSONDecodeError

# set to true on debug environment only
DEBUG = False

# wrapper name and version
WRAPPER_NAME = "python"
WRAPPER_VERSION = "2.5.0"

# API inter-command timeout (in ms)
API_SEND_TIMEOUT = 100

# max connection tries
API_MAX_CONN_TRIES = 3

# logger properties
if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")

logger = logging.getLogger(__name__)
logger = setup_logging(logger, logs_path, __name__, console_debug=True)


class apiSocket:
    def __init__(self, address, port, packet_size=4096) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket = ssl.wrap_socket(sock)
        self._timeout = None
        self._address = address
        self._port = port
        self._decoder = json.JSONDecoder()
        self._receivedData = ""
        self.is_connected = False
        self.is_streaming = False
        self.packet_size = packet_size

    def connect(self):
        for i in range(API_MAX_CONN_TRIES):
            try:
                self.socket.connect((self._address, self._port))
            except socket.error as msg:
                logger.error("SockThread Error: %s" % msg)
                time.sleep(0.25)
                continue
            # logger.debug("\nSocket connected")
            self.is_connected = True
            return True
        self.is_connected = False
        return False

    def sendObj(self, obj):
        try:
            msg = json.dumps(obj)
            msg = msg.encode("utf-8")
        except Exception as e:
            logger.error(f"{e}")
            self.is_connected = False
            raise Exception
        try:
            if self.is_connected:
                total_sent = 0
                while total_sent < len(msg):
                    sent = self.socket.send(msg[total_sent:])
                    if sent == 0:
                        raise Exception
                    total_sent = total_sent + sent
                    # logger.debug("\nSent: " + str(msg))
                    time.sleep(API_SEND_TIMEOUT / 1000)
        except Exception as e:
            logger.error(f"{e}")
            self.is_connected = False
            raise SocketError(e)

    def readObj(self, bytesSize=4096):
        attempts = 0
        if not self.socket:
            logger.error("Socket connection broken")
            raise RuntimeError("socket connection broken")
        while True:
            try:
                char = self.socket.recv(bytesSize).decode()
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
                attempts += 1
                if attempts > 20:
                    raise SocketError
        return resp

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

    def close(self):
        logger.debug("Closing socket")
        self.is_connected = False
        self.socket.close()

    def _keep_alive(self):
        pass


class APIClient(apiSocket):
    def __init__(self, address, port, packet_size=4096) -> None:
        super().__init__(address, port, packet_size)
        self.LAST_COMMAND_EXECUTED = time.time()
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

        self.pingFun = Thread(target=self.ping)
        self.pingFun.setDaemon(True)
        self.pingFun.start()

    def execute(self, dictionary):
        self.sendObj(dictionary)
        return self.readObj()

    def disconnect(self):
        self.close()

    def commandExecute(self, commandName, arguments=None):
        try:
            commandResponse = self.execute(baseCommand(commandName, arguments))
            if commandResponse["status"] == False:
                error_code = commandResponse["errorCode"]
                error_description = commandResponse["errorDescr"]
                logger.error(f"\nLogin failed. Error code: {error_code}")
                raise LoginError(
                    f"Error code: {error_code} | Error description: {error_description}"
                )
            self.LAST_COMMAND_EXECUTED = time.time()
            return commandResponse
        except Exception as e:
            self.disconnect()
            raise SocketError(e)

    def ping(self):
        while self.is_connected:
            try:
                if (time.time() - self.LAST_COMMAND_EXECUTED) > (60):
                    commandResponse = self.execute(
                        baseCommand(
                            commandName="ping",
                        )
                    )
                    logger.debug("ping")
            except Exception as e:
                self.kill_ping = 1
                self.disconnect()
                raise SocketError(e)

    def login(self, user, passw, appName=""):
        try:
            commandResponse = self.commandExecute(
                commandName="login",
                arguments=dict(userId=user, password=passw, appName=appName),
            )
            logger.info("Login successfull")
            return commandResponse["streamSessionId"]
        except Exception as e:
            logger.error(f"{e}")
            raise SocketError(e)


class APIStreamClient(apiSocket):
    def __init__(
        self,
        address,
        port,
        packet_size=4096,
        ssId=None,
        tickFun=None,
        tradeFun=None,
        balanceFun=None,
        tradeStatusFun=None,
        profitFun=None,
        newsFun=None,
        candleFun=None,
    ) -> None:
        super().__init__(address, port, packet_size)
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
        self.is_streaming = True
        self.stream_thread = Thread(target=self.readStream, args=())
        self.stream_thread.setDaemon(True)
        self.stream_thread.start()

    def readStream(self):
        if self.is_connected:
            while self.is_streaming:
                msg = self.readObj()
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

    def disconnect(self):
        self.is_streaming = False
        self.close()

    def execute(self, dictionary):
        self.sendObj(dictionary)

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
