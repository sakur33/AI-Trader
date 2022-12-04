import json
from datetime import datetime, timedelta
import websocket
import asyncio
from websocket._exceptions import WebSocketConnectionClosedException
from utils import get_time, to_milliseconds, time_conversion


class XTB:
    __version__ = "1.0"

    def __init__(self, ID, PSW):
        self.ID = ID
        self.PSW = PSW
        self.ws = 0
        self.exec_start = get_time()
        self.connect()
        self.login()

    def login(self):
        login = {
            "command": "login",
            "arguments": {"userId": self.ID,
                        "password": self.PSW,
                        "appId": "test",
                        "appName": "test"},
        }
        login_json = json.dumps(login)
        # Sending Login Request
        result = self.send(login_json)
        result = json.loads(result)
        status = result["status"]
        self.stream_id = result["streamSessionId"]
        if str(status) == "True":
            # Success
            return True
        else:
            # Error
            return False

    def logout(self):
        logout = {"command": "logout"}
        logout_json = json.dumps(logout)
        # Sending Logout Request
        result = self.send(logout_json)
        result = json.loads(result)
        status = result["status"]
        self.disconnect()
        if str(status) == "True":
            # Success
            return True
        else:
            # Error
            return False

    def get_AllSymbols(self):
        """get_AllSymbols
        Returns array of all symbols available for the user.

        Returns:
            _type_: _description_
        """
        allsymbols = {"command": "getAllSymbols"}
        allsymbols_json = json.dumps(allsymbols)
        result = self.send(allsymbols_json)
        result = json.loads(result)
        return result

    def get_STCSymbols(self):
        allsymbols = self.get_AllSymbols()
        STC = []
        for symbol in allsymbols["returnData"]:
            if symbol["categoryName"] == "STC":
                STC.append(symbol)
        return STC

    def get_Candles(self, period, symbol, days=0, hours=0, minutes=0, qty_candles=0):
        """get_Candles Subscribes for and unsubscribes from API chart candles.
        The interval of every candle is 1 minute. A new candle arrives every minute.

        Args:
            period (_type_): _description_
            symbol (_type_): _description_
            days (int, optional): _description_. Defaults to 0.
            hours (int, optional): _description_. Defaults to 0.
            minutes (int, optional): _description_. Defaults to 0.
            qty_candles (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if period == "M1":
            minutes += qty_candles
            period = 1
        elif period == "M5":
            minutes += qty_candles * 5
            period = 5
        elif period == "M15":
            minutes += qty_candles * 15
            period = 15
        elif period == "M30":
            minutes += qty_candles * 30
            period = 30
        elif period == "H1":
            minutes += qty_candles * 60
            period = 60
        elif period == "H4":
            minutes += qty_candles * 240
            period = 240
        elif period == "D1":
            minutes += qty_candles * 1440
            period = 1440
        elif period == "W1":
            minutes += qty_candles * 10080
            period = 10080
        elif period == "MN1":
            minutes += qty_candles * 43200
            period = 43200
        if qty_candles != 0:
            minutes = minutes * 2
        start = self.get_ServerTime() - to_milliseconds(
            days=days, hours=hours, minutes=minutes
        )
        CHART_LAST_INFO_RECORD = {"period": period, "start": start, "symbol": symbol}
        candles = {
            "command": "getChartLastRequest",
            "arguments": {"info": CHART_LAST_INFO_RECORD},
        }
        candles = {
            "command": "getCandles",
            "streamSessionId": self.stream_id,
            "symbol": symbol,
        }
        candles_json = json.dumps(candles)
        result = self.send(candles_json)
        result = json.loads(result)
        candles = []
        candle = {}
        qty = len(result["returnData"]["rateInfos"])
        candle["digits"] = result["returnData"]["digits"]
        if qty_candles == 0:
            candle["qty_candles"] = qty
        else:
            candle["qty_candles"] = qty_candles
        candles.append(candle)
        if qty_candles == 0:
            start_qty = 0
        else:
            start_qty = qty - qty_candles
        if qty == 0:
            start_qty = 0

        for i in range(start_qty, qty):
            candle = {}
            candle["datetime"] = result["returnData"]["rateInfos"][i]["ctmString"]
            candle["open"] = result["returnData"]["rateInfos"][i]["open"]
            candle["close"] = result["returnData"]["rateInfos"][i]["close"]
            candle["high"] = result["returnData"]["rateInfos"][i]["high"]
            candle["low"] = result["returnData"]["rateInfos"][i]["low"]
            candles.append(candle)
        if len(candles) == 1:
            return False
        return candles
        """
        LIMITS:
        PERIOD_M1 --- <0-1) month, i.e. one month time
        PERIOD_M30 --- <1-7) month, six months time
        PERIOD_H4 --- <7-13) month, six months time
        PERIOD_D1 --- 13 month, and earlier on
        ##############################################
        PERIOD_M1	1	1 minute
        PERIOD_M5	5	5 minutes
        PERIOD_M15	15	15 minutes
        PERIOD_M30	30	30 minutes
        PERIOD_H1	60	60 minutes (1 hour)
        PERIOD_H4	240	240 minutes (4 hours)
        PERIOD_D1	1440	1440 minutes (1 day)
        PERIOD_W1	10080	10080 minutes (1 week)
        PERIOD_MN1	43200	43200 minutes (30 days)
        ##############################################
        close   Value of close price (shift from open price)
        ctm     Candle start time in CET / CEST time zone (see Daylight Saving Time)
        ctmString     String representation of the 'ctm' field
        high    Highest value in the given period (shift from open price)
        low     Lowest value in the given period (shift from open price)
        open    Open price (in base currency * 10 to the power of digits)
        vol     Volume in lots
        """

    def get_CandlesRange(self, period, symbol, start=0, end=0, days=0, qty_candles=0):
        if period == "M1":
            period = 1
        elif period == "M5":
            period = 5
        elif period == "M15":
            period = 15
        elif period == "M30":
            period = 30
        elif period == "H1":
            period = 60
        elif period == "H4":
            period = 240
        elif period == "D1":
            period = 1440
        elif period == "W1":
            period = 10080
        elif period == "MN1":
            period = 43200

        if end == 0:
            end = get_time()
            end = end.strftime("%m/%d/%Y %H:%M:%S")
            if start == 0:
                if qty_candles == 0:
                    temp = datetime.strptime(end, "%m/%d/%Y %H:%M:%S")
                    start = temp - timedelta(days=days)
                    start = start.strftime("%m/%d/%Y %H:%M:%S")
                else:
                    start = datetime.strptime(end, "%m/%d/%Y %H:%M:%S")
                    minutes = period * qty_candles
                    start = start - timedelta(minutes=minutes)
                    start = start.strftime("%m/%d/%Y %H:%M:%S")

        start = time_conversion(start)
        end = time_conversion(end)

        CHART_RANGE_INFO_RECORD = {
            "end": end,
            "period": period,
            "start": start,
            "symbol": symbol,
            "ticks": 0,
        }
        candles = {
            "command": "getChartRangeRequest",
            "arguments": {"info": CHART_RANGE_INFO_RECORD},
        }
        candles_json = json.dumps(candles)
        result = self.send(candles_json)
        result = json.loads(result)
        candles = []
        candle = {}
        qty = len(result["returnData"]["rateInfos"])
        candle["digits"] = result["returnData"]["digits"]
        if qty_candles == 0:
            candle["qty_candles"] = qty
        else:
            candle["qty_candles"] = qty_candles
        candles.append(candle)
        if qty_candles == 0:
            start_qty = 0
        else:
            start_qty = qty - qty_candles
        if qty == 0:
            start_qty = 0
        for i in range(start_qty, qty):
            candle = {}
            candle["datetime"] = str(result["returnData"]["rateInfos"][i]["ctmString"])
            candle["open"] = result["returnData"]["rateInfos"][i]["open"]
            candle["close"] = result["returnData"]["rateInfos"][i]["close"]
            candle["high"] = result["returnData"]["rateInfos"][i]["high"]
            candle["low"] = result["returnData"]["rateInfos"][i]["low"]
            candles.append(candle)
        if len(candles) == 1:
            return False
        return candles
        """
        LIMITS:
        PERIOD_M1 --- <0-1) month, i.e. one month time
        PERIOD_M30 --- <1-7) month, six months time
        PERIOD_H4 --- <7-13) month, six months time
        PERIOD_D1 --- 13 month, and earlier on
        ##############################################
        PERIOD_M1	1	1 minute
        PERIOD_M5	5	5 minutes
        PERIOD_M15	15	15 minutes
        PERIOD_M30	30	30 minutes
        PERIOD_H1	60	60 minutes (1 hour)
        PERIOD_H4	240	240 minutes (4 hours)
        PERIOD_D1	1440	1440 minutes (1 day)
        PERIOD_W1	10080	10080 minutes (1 week)
        PERIOD_MN1	43200	43200 minutes (30 days)
        ##############################################
        close   Value of close price (shift from open price)
        ctm     Candle start time in CET / CEST time zone (see Daylight Saving Time)
        ctmString     String representation of the 'ctm' field
        high    Highest value in the given period (shift from open price)
        low     Lowest value in the given period (shift from open price)
        open    Open price (in base currency * 10 to the power of digits)
        vol     Volume in lots
        """

    def get_ServerTime(self):
        time = {"command": "getServerTime"}
        time_json = json.dumps(time)
        result = self.send(time_json)
        result = json.loads(result)
        time = result["returnData"]["time"]
        return time

    def get_Balance(self):
        balance = {"command": "getMarginLevel"}
        balance_json = json.dumps(balance)
        result = self.send(balance_json)
        result = json.loads(result)
        balance = result["returnData"]["balance"]
        return balance

    def get_Margin(self, symbol, volume):
        margin = {
            "command": "getMarginTrade",
            "arguments": {"symbol": symbol, "volume": volume},
        }
        margin_json = json.dumps(margin)
        result = self.send(margin_json)
        result = json.loads(result)
        margin = result["returnData"]["margin"]
        return margin

    def get_Profit(self, open_price, close_price, transaction_type, symbol, volume):
        if transaction_type == 1:
            # buy
            cmd = 0
        else:
            # sell
            cmd = 1
        profit = {
            "command": "getProfitCalculation",
            "arguments": {
                "closePrice": close_price,
                "cmd": cmd,
                "openPrice": open_price,
                "symbol": symbol,
                "volume": volume,
            },
        }
        profit_json = json.dumps(profit)
        result = self.send(profit_json)
        result = json.loads(result)
        profit = result["returnData"]["profit"]
        return profit

    def get_Symbol(self, symbol):
        symbol = {"command": "getSymbol", "arguments": {"symbol": symbol}}
        symbol_json = json.dumps(symbol)
        result = self.send(symbol_json)
        result = json.loads(result)
        symbol = result["returnData"]
        return symbol

    def make_Trade(
        self,
        symbol,
        cmd,
        transaction_type,
        volume,
        comment="",
        order=0,
        sl=0,
        tp=0,
        days=0,
        hours=0,
        minutes=0,
    ):
        price = self.get_Candles("M1", symbol, qty_candles=1)
        price = price[1]["open"] + price[1]["close"]

        delay = to_milliseconds(days=days, hours=hours, minutes=minutes)
        if delay == 0:
            expiration = self.get_ServerTime() + to_milliseconds(minutes=1)
        else:
            expiration = self.get_ServerTime() + delay

        TRADE_TRANS_INFO = {
            "cmd": cmd,
            "customComment": comment,
            "expiration": expiration,
            "offset": -1,
            "order": order,
            "price": price,
            "sl": sl,
            "symbol": symbol,
            "tp": tp,
            "type": transaction_type,
            "volume": volume,
        }
        trade = {
            "command": "tradeTransaction",
            "arguments": {"tradeTransInfo": TRADE_TRANS_INFO},
        }
        trade_json = json.dumps(trade)
        result = self.send(trade_json)
        result = json.loads(result)
        if result["status"] == True:
            # success
            return True, result["returnData"]["order"]
        else:
            # error
            return False, 0
        """
        format TRADE_TRANS_INFO:
        cmd	        Number	            Operation code
        customComment	String	            The value the customer may provide in order to retrieve it later.
        expiration	Time	            Pending order expiration time
        offset	        Number	            Trailing offset
        order	        Number	            0 or position number for closing/modifications
        price	        Floating number	    Trade price
        sl	        Floating number	    Stop loss
        symbol	        String	            Trade symbol
        tp	        Floating number	    Take profit
        type	        Number	            Trade transaction type
        volume	        Floating number	    Trade volume

        values cmd:
        BUY	        0	buy
        SELL	        1	sell
        BUY_LIMIT	2	buy limit
        SELL_LIMIT	3	sell limit
        BUY_STOP	4	buy stop
        SELL_STOP	5	sell stop
        BALANCE	        6	Read only. Used in getTradesHistory  for manager's deposit/withdrawal operations (profit>0 for deposit, profit<0 for withdrawal).
        CREDIT	        7	Read only

        values transaction_type:
        OPEN	    0	    order open, used for opening orders
        PENDING	    1	    order pending, only used in the streaming getTrades  command
        CLOSE	    2	    order close
        MODIFY	    3	    order modify, only used in the tradeTransaction  command
        DELETE	    4	    order delete, only used in the tradeTransaction  command
        """

    def check_Trade(self, order):
        """
        ERROR 	0	error
        PENDING	1	pending
        ACCEPTED	3	The transaction has been executed successfully
        REJECTED	4	The transaction has been rejected

        """
        trade = {"command": "tradeTransactionStatus", "arguments": {"order": order}}
        trade_json = json.dumps(trade)
        result = self.send(trade_json)
        result = json.loads(result)
        status = result["returnData"]["requestStatus"]
        return status

    def get_History(self, start=0, end=0, days=0, hours=0, minutes=0):
        if start != 0:
            start = time_conversion(start)
        if end != 0:
            end = time_conversion(end)

        if days != 0 or hours != 0 or minutes != 0:
            if end == 0:
                end = self.get_ServerTime()
            start = end - to_milliseconds(days=days, hours=hours, minutes=minutes)

        history = {
            "command": "getTradesHistory",
            "arguments": {"end": end, "start": start},
        }
        history_json = json.dumps(history)
        result = self.send(history_json)
        result = json.loads(result)
        history = result["returnData"]
        return history

    def ping(self):
        ping = {"command": "ping"}
        ping_json = json.dumps(ping)
        result = self.send(ping_json)
        result = json.loads(result)
        return result["status"]

    def connect(self):
        try:
            self.ws = websocket.create_connection("wss://ws.xtb.com/demoStream")
            # Success
            return True
        except:
            # Error
            return False

    def disconnect(self):
        try:
            self.ws.close()
            # Success
            return True
        except:
            return False

    def send(self, msg):
        self.is_on()
        try:
            self.ws.send(msg)
            result = self.ws.recv()
            return result + "\n"
        except WebSocketConnectionClosedException as e:
            print(e)
        return None

    def is_on(self):
        temp1 = self.exec_start
        temp2 = get_time()
        temp = temp2 - temp1
        temp = temp.total_seconds()
        temp = float(temp)
        if temp >= 8.0:
            self.connect()
            self.login()
        self.exec_start = get_time()

    def is_open(self, symbol):
        candles = self.get_Candles("M1", symbol, qty_candles=1)
        if len(candles) == 1:
            return False
        else:
            return True
