from xAPIConnector import *
from trader_utils import *
from trader_db_utils import *
from logger_settings import *
from custom_exceptions import *
import time
import os
import copy as cp
from multiprocessing import SimpleQueue
from connection_clock import Clock
import pickle as pkl
import shutil
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class BaseTrader:
    def __init__(
        self,
        name,
        capital,
        max_risk,
        trader_type,
        logger=logging.getLogger(__name__),
        test=0,
    ) -> None:
        self.test = test
        self.logger = logger
        # if c_logger is not None:
        #     self.logger = c_logger.setup_logging(self.logger, __name__, console_debug=True)

        self.user = creds[name.split(":")[0]]["user"]
        self.passw = creds[name.split(":")[0]]["passw"]
        self.name = name
        self.CAPITAL = capital
        self.max_risk = max_risk
        self.trader_type = trader_type
        self.DB = TraderDB(logger=self.logger)
        traderid = self.DB.get_trader_id(name)
        if traderid is None:
            traderid = self.DB.insert_trader(name, capital, max_risk)
            self.logger.info(f"Trader created id: {traderid}")
        else:
            self.logger.info(f"Trader exists id: {traderid}")
        self.traderid = traderid

        self.CLIENT = None
        self.STREAM_CLIENTS = {}
        self.TICK_QUEUE = SimpleQueue()
        self.CANDLE_QUEUE = SimpleQueue()
        self.TRADE_QUEUE = SimpleQueue()
        self.CLOCK = Clock()

        self.LAST_TICK = None
        self.LAST_TICKS = []
        self.LAST_CANDLE = None
        self.LAST_CANDLES = None
        self.new_candle = False

        self.last_risk_update_time = 0
        self.live_risks = 0

    def start_api_client(self):
        apiClient = APIClient(
            address=DEFAULT_XAPI_ADDRESS, port=DEFAULT_XAPI_PORT, logger=self.logger
        )
        ssid = apiClient.login(user=self.user, passw=self.passw)
        self.set_api_client(client=apiClient, ssid=ssid)

    def set_api_client(self, client, ssid):
        self.CLIENT = client
        self.ssid = ssid

    def start_stream_clients(self, symbol, tick=False, candle=True, trade=True):
        try:
            symbol = symbol
            if self.CLIENT is None:
                self.start_api_client()
            if tick:
                stream_client = APIStreamClient(
                    ssId=self.ssid,
                    address=DEFAULT_XAPI_ADDRESS,
                    port=DEFAULT_XAPI_STREAMING_PORT,
                    tick_queue=self.TICK_QUEUE,
                    db_obj=self.DB,
                    logger=self.logger,
                )
                stream_client.subscribePrice(symbol, minArrivalTime=5000, maxLevel=3)
                self.set_stream_clients(name="tick", stream_client=stream_client)

            if candle:
                stream_client = APIStreamClient(
                    ssId=self.ssid,
                    address=DEFAULT_XAPI_ADDRESS,
                    port=DEFAULT_XAPI_STREAMING_PORT,
                    candle_queue=self.CANDLE_QUEUE,
                    db_obj=self.DB,
                    logger=self.logger,
                )
                stream_client.subscribeCandle(symbol)
                self.set_stream_clients(name="candle", stream_client=stream_client)

            if trade:
                stream_client = APIStreamClient(
                    ssId=self.ssid,
                    address=DEFAULT_XAPI_ADDRESS,
                    port=DEFAULT_XAPI_STREAMING_PORT,
                    db_obj=self.DB,
                    logger=self.logger,
                )
                stream_client.subscribeTrades()
                self.set_stream_clients(name="trade", stream_client=stream_client)

        except Exception as e:
            raise ApiException

    def set_stream_clients(self, name, stream_client):
        self.STREAM_CLIENTS[name] = stream_client

    def store_past_candles(self, symbol, start_date, end_date=datetime.now(), period=1):
        try:
            CHART_RANGE_INFO_RECORD = {
                "period": period,
                "start": date_to_xtb_time(start_date),
                "end": date_to_xtb_time(end_date),
                "symbol": symbol,
                "ticks": 0,
            }
            self.CLOCK.wait_clock()
            commandResponse = self.CLIENT.commandExecute(
                "getChartRangeRequest", arguments={"info": CHART_RANGE_INFO_RECORD}
            )
            if commandResponse["status"] == False:
                error_code = commandResponse["errorCode"]
                self.logger.debug(f"Login failed. Error code: {error_code}")

            else:
                returnData = commandResponse["returnData"]
                digits = returnData["digits"]
                if len(returnData["rateInfos"]) != 0:
                    candles = return_as_df(returnData["rateInfos"])
                    if not candles is None:
                        candles = cast_candles_to_types(candles, digits, dates=True)
                        candles = adapt_data(candles)
                        self.LAST_CANDLES = candles.reset_index(drop=True)
                        candles["symbol"] = symbol
                        self.DB.insert_candle_batch(candles)
                    else:
                        self.logger.debug(
                            f"Symbol {symbol} did not return candles for dates {start_date.strftime('%Y-%m-%d')} | {end_date.strftime('%Y-%m-%d')}"
                        )
        except Exception as e:
            raise ApiException

    def update_stocks(self, df, period, days=14, force=False):
        try:
            if days > 30 and period == 1:
                for day_range in range(1, days // 15):
                    start_date = datetime.now() - timedelta(days=days * day_range)
                    end_date = start_date + timedelta(days=15)
                    self.logger.info(
                        f"Inserting {start_date.strftime('%Y-%m-%d')} | {end_date.strftime('%Y-%m-%d')} ..."
                    )
                    for symbol in tqdm(list(df["symbol"])):
                        self.store_past_candles(symbol, start_date, end_date, period)

            else:
                start_date = datetime.now() - timedelta(days=days)
                for symbol in list(df["symbol"]):
                    self.logger.info(f"Inserting symbol: {symbol}")
                    if not self.DB.get_candles_today() or force:
                        self.store_past_candles(symbol, start_date, period)
        except Exception as e:
            raise LogicError

    def get_trade_status(self, order_n):
        try:
            self.CLOCK.wait_clock()
            commandResponse = self.CLIENT.commandExecute(
                "tradeTransactionStatus", arguments={"order": order_n}
            )
            if commandResponse["returnData"]["requestStatus"] == 3:
                return True
            else:
                return False
        except Exception as e:
            raise SessionError

    def get_trades(self):
        try:
            self.CLOCK.wait_clock()
            commandResponse = self.CLIENT.commandExecute(
                "getTrades", arguments={"openedOnly": True}
            )

            trade_records = commandResponse["returnData"]
            if len(trade_records) == 0:
                if isinstance(trade_records, dict):
                    return [trade_records]
                else:
                    return None
            else:
                if isinstance(trade_records, list):
                    return trade_records
                else:
                    return [trade_records]
        except Exception as e:
            self.logger.debug(f"Exception at get trades: {e}")
            raise ApiException

    def get_trade(self, opened=True, order_n=None):
        attempts = 0
        while attempts < 20:
            self.CLOCK.wait_clock()
            try:
                if opened:
                    if order_n is not None:
                        if self.is_trade_closed(order_n):
                            return None
                    commandResponse = self.CLIENT.commandExecute(
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
                    commandResponse = self.CLIENT.commandExecute(
                        "getTradesHistory", arguments={"end": 0, "start": 0}
                    )
                    trade_records = commandResponse["returnData"]

                    if isinstance(trade_records, list):
                        for record in trade_records:
                            if record["position"] == order_n:
                                return record
                    else:
                        if record["position"] == order_n:
                            return record
                attempts += 1
            except Exception as e:
                self.logger.debug(f"Exception at get trade: {e}")
                raise ApiException
        return None

    def is_trade_closed(self, order_n):
        try:
            found = self.DB.find_closed_trade(order_n)
            return found
        except Exception as e:
            raise DbError

    def trade(
        self,
        order_cmd,
        order_type,
        name,
        c_tick,
        sl=0.0,
        position=0,
        price=0,
        volume=0,
        trailing=False,
    ):
        try:
            order_status = False
            while order_status == False:
                self.CLOCK.wait_clock()
                if price == 0:
                    price = c_tick["ask"]
                if volume == 0:
                    volume = self.VOLUME
                if trailing:
                    trailing = int(self.calculate_trailing(sl, c_tick["ask"]))
                    self.logger.info(f"Added trailing of {trailing/10}")
                else:
                    trailing = 0
                if order_type == 0 or order_type == 3:
                    added_risk = self.test_risk(price, sl, volume)
                    if np.sum(self.live_risks) + added_risk > (
                        self.CAPITAL * self.max_risk
                    ):
                        self.logger.info(
                            "\n\n********\nNOT ENOUGH RISK SPACE\n********\n\n"
                        )
                        return None
                commandResponse = self.CLIENT.commandExecute(
                    "tradeTransaction",
                    arguments={
                        "tradeTransInfo": {
                            "cmd": order_cmd,
                            "customComment": name,
                            "expiration": 0,
                            "offset": trailing,
                            "order": position,
                            "price": price,
                            "sl": sl,
                            "symbol": c_tick["symbol"],
                            "tp": 0.0,
                            "type": order_type,
                            "volume": volume,
                        }
                    },
                )
                returnData = commandResponse["returnData"]
                if isinstance(returnData, dict):
                    if "order" in list(returnData.keys()):
                        order_status = False
                        order_n = returnData["order"]
                        order_status = self.get_trade_status(order_n)
                    else:
                        order_status = False
                else:
                    order_status = False
                if order_status == False and (
                    (order_type == TransactionType.ORDER_OPEN)
                    or (order_type == TransactionType.ORDER_MODIFY)
                ):
                    if order_cmd == 0:
                        self.logger.info(
                            f"Attempt to trade failed | sl {sl} -> {sl * (1 - 0.0001)}"
                        )
                        sl = float(np.round(sl * (1 - 0.0001), 5))
                    else:
                        self.logger.info(
                            f"Attempt to trade failed | sl {sl} -> {sl * (1 + 0.0001)}"
                        )
                        sl = float(np.round(sl * (1 + 0.0001), 5))
                    self.logger.info(f"Attempt to buy failed | sl {sl}")
            return order_n
        except Exception as e:
            self.logger.error(f"Exception trading: {e}")
            raise SessionError(e)

    def get_profit(self, buy_position):
        try:
            trade = self.get_trade(opened=False, order_n=buy_position)
            return trade["profit"]
        except Exception as e:
            raise SessionError(e)

    def enough_money(self, c_tick):
        position = self.SYMBOL_INFO["contractSize"] * self.VOLUME * c_tick["ask"]
        margin = position * (self.SYMBOL_INFO["leverage"] / 100)
        if (np.sum(self.HIST_PROFITS) + self.CAPITAL) * self.max_risk > (margin):
            return True
        else:
            return False

    def enough_money_live(self):
        if np.sum(self.live_risks) < (self.CAPITAL * self.max_risk):
            return True
        else:
            return False

    def calculate_stop_loss(self, short, price, multiple=4):
        if short:
            sl = np.round(
                price + (self.SYMBOL_INFO["tickSize"] * multiple),
                self.SYMBOL_INFO["precision"],
            )
        else:
            sl = np.round(
                price - (self.SYMBOL_INFO["tickSize"] * multiple),
                self.SYMBOL_INFO["precision"],
            )
        return sl

    def calculate_trailing(self, sl, price):
        return (
            np.abs(sl - price)
            // (1.0 / np.power(10, self.SYMBOL_INFO["pipsPrecision"]))
            * 10
        )

    def test_risk(self, price, stop_loss, volume):
        pip_size = 1.0 / np.power(10, self.SYMBOL_INFO["pipsPrecision"])
        contract_size = self.SYMBOL_INFO["contractSize"]
        if self.SYMBOL_INFO["currencyProfit"] == "USD":
            pip_price = 10 * volume
        else:
            pip_price = (pip_size / price) * contract_size * volume
        max_pip_drop = np.abs(stop_loss - price) // pip_size
        max_loss = max_pip_drop * pip_price
        return max_loss

    def evaluate_risks(self):
        if (time.time() - self.last_risk_update_time) > 50:
            trades = self.get_trades()
            self.CLOCK.wait_clock()
            max_losses = []
            if trades is not None:
                for trade in trades:
                    open_price = trade["open_price"]
                    stop_loss = trade["sl"]
                    volume = trade["volume"]
                    max_loss = self.test_risk(open_price, stop_loss, volume)
                    max_losses.append(max_loss)
                self.live_risks = max_losses
            else:
                self.live_risks = 0
            self.last_risk_update_time = time.time()

    def step(self):
        self.logger.info("Base logger step function has to be overriden")


class AngleTrader(BaseTrader):
    def __init__(
        self, name, capital, max_risk, trader_type, logger=None, test=0
    ) -> None:
        super().__init__(name, capital, max_risk, trader_type, logger=logger, test=test)

        self.SYMBOl = None
        self.VOLUME = None
        self.HIST_PROFITS = [0]
        self.ORDER_N = None
        self.POTENTIAL_PROFITS = []
        self.SYMBOL_INFO = None
        self.MAX_RISK = None
        self.IN_POSITION = False
        self.TRANS_TYPE = None

        self.short_ma = None
        self.long_ma = None
        self.profit_exit = None
        self.loss_exit = None
        self.min_angle = None

        self.last_update_time = 0

    def sell_position(self):
        # TODO: Implement the function to call self.trade and sell the position
        pass

    def buy_position(self):
        # TODO: Implement the function to call self.trade and buy a position
        pass

    def enter_position(self):
        # TODO: Implement the logic to enter a trade.

        """
        Rules to enter:
            1st: Separation between short_ma and long_ma is greater than the spread
            2nd: Enter long:
                ((angle_short > 60) and (angle long < -6) and (short_ma < long_ma))
                OR ((angle_short > 30) and (angle_long > 15))
            3rd: Enter short:
                ((angle_short < -60) and (angle long > 6) and (short_ma > long_ma))
                OR ((angle_short < -30) and (angle_long < -15))
        """
        pass

    def exit_position(self):
        # TODO: Implement the logit to exit a position

        """
        Rules to exit:
            1st: Profit is negative (not counting entry spread)
            2nd: If last profit / current profit < 0.95
            3rd: Profit is > 0.01
        """


class CrossTrader(BaseTrader):
    def __init__(
        self, name, capital, max_risk, trader_type, logger=None, test=0
    ) -> None:
        super(CrossTrader, self).__init__(
            name, capital, max_risk, trader_type, logger=logger, test=test
        )

        self.SYMBOl = None
        self.VOLUME = None
        self.HIST_PROFITS = [0]
        self.ORDER_N = None
        self.POTENTIAL_PROFITS = []
        self.SYMBOL_INFO = None
        self.MAX_RISK = None
        self.IN_POSITION = False
        self.TRANS_TYPE = None

        self.short_ma = None
        self.long_ma = None
        self.profit_exit = None
        self.loss_exit = None
        self.min_angle = None

        self.last_update_time = 0

    def look_for_suitable_symbols_v1(self, df, symbol_type, capital, max_risk):
        # Look for symbols with:
        #   - Trader_type products
        #   - Minimal stake is less than our max risk
        try:
            df = df[df["categoryName"] == symbol_type]
            df = df[
                (df["ask"] * df["lotMin"] * df["contractSize"] * (df["leverage"] / 100))
                <= (capital * max_risk)
            ]
            df["spread_percentage"] = (df["ask"] - df["bid"]) / df["ask"]
            df = df.sort_values(by=["spread_percentage"])
            return df
        except Exception as e:
            raise LogicError

    def evaluate_stocks(self, date, threshold):
        """evaluate_stocks _summary_ Function to select the proper stocks to trade this strategy.
        This test will be done over the already filtered suitable stocks.
        For this strategy to apply, a couple of characteristics must apply:
          - Volatility has to be way bigger (x40) than the spread

        Args:
            date (_type_): date from which to test the characteristics (last week)
            threshold (_type_): volatility multiplier required
        """

        candles = self.DB.get_candles_range(date, datetime.today().strftime("%Y-%m-%d"))
        symbol_analysis = {}
        for symbol in list(candles["symbol"].unique()):
            symbol_info = self.DB.get_symbol_info(symbol=symbol)
            symbol_candle = candles[candles["symbol"] == symbol]
            symbol_volatility = symbol_candle["close"].std()
            symbol_hourly_volatility = symbol_candle["close"].rolling(window=60).std()
            symbol_spread = symbol_info["spreadRaw"]
            symbol_score = symbol_volatility / (symbol_spread * threshold)
            symbol_analysis[symbol] = {
                "Score": symbol_score,
                "Volatity": symbol_volatility,
                "Hourly-Volatility": symbol_hourly_volatility,
                "Spread": symbol_spread,
            }
        return symbol_analysis

    def up_crossover(self, prev_candle, c_candle):

        if (prev_candle["short_ma"] < prev_candle["long_ma"]) and (
            c_candle["short_ma"] > c_candle["long_ma"]
        ):
            return True
        else:
            return False

    def down_crossover(self, prev_candle, c_candle):

        if (prev_candle["short_ma"] > prev_candle["long_ma"]) and (
            c_candle["short_ma"] < c_candle["long_ma"]
        ):
            return True
        else:
            return False

    def sell_position(self, position, short):
        self.logger.info("\n******\nSELL\n******")
        c_tick = self.LAST_TICK
        if short:
            order_cmd = TransactionSide.BUY
        else:
            order_cmd = TransactionSide.SELL
        order_type = TransactionType.ORDER_CLOSE
        self.POTENTIAL_PROFITS = []
        if position is not None:
            sell_order_n = self.trade(
                name=self.name,
                order_cmd=order_cmd,
                order_type=order_type,
                position=position,
                c_tick=c_tick,
            )
            self.HIST_PROFITS.append(self.get_profit(position))
        self.ORDER_N = None
        self.IN_POSITION = False
        self.logger.info(f"Hist. Profits: {self.HIST_PROFITS}")

    def buy_position(self, short=False):
        self.CLOCK.wait_clock()
        c_tick = self.LAST_TICK
        if short:
            trade_type = "SHORT"
            order_cmd = TransactionSide.SELL
            maxes = self.LAST_CANDLES["max"].values
            sl = maxes[~np.isnan(maxes)][-1]
        else:
            trade_type = "BUY"
            order_cmd = TransactionSide.BUY
            sl = self.LAST_CANDLES
            mins = self.LAST_CANDLES["min"].values
            sl = mins[~np.isnan(mins)][-1]

        order_type = TransactionType.ORDER_OPEN
        order_n = self.trade(
            name=self.name,
            order_cmd=order_cmd,
            order_type=order_type,
            c_tick=c_tick,
            sl=sl,
            trailing=True,
        )
        if order_n is not None:
            trade = self.get_trade(opened=True, order_n=order_n)
        else:
            trade = None
        if trade is None:
            self.logger.info(f"Not Positioned")
            return None, None
        else:
            self.logger.info(
                f"\n******\n{trade_type}\n******\nBUY PRICE: {trade['open_price']} | SL: {trade['sl']}"
            )
            self.IN_POSITION = True
            return trade["position"], trade["cmd"]

    def set_last_candle(self, margin=5):
        # self.LAST_CANDLE = self.DB.get_last_candle(self.SYMBOl)
        if not self.CANDLE_QUEUE.empty():
            new_candle = self.CANDLE_QUEUE.get()
            if self.LAST_CANDLES is None:
                self.LAST_CANDLES = pd.DataFrame.from_dict(new_candle)
            else:
                self.LAST_CANDLES = self.LAST_CANDLES.append(
                    new_candle, ignore_index=True
                )
                if self.LAST_CANDLES.shape[0] > (self.long_ma + 3):
                    self.LAST_CANDLES["short_ma"] = (
                        self.LAST_CANDLES["close"].rolling(window=self.short_ma).mean()
                    )
                    self.LAST_CANDLES["long_ma"] = (
                        self.LAST_CANDLES["close"].rolling(window=self.long_ma).mean()
                    )
                    self.LAST_CANDLES["min"] = self.LAST_CANDLES.iloc[
                        argrelextrema(
                            self.LAST_CANDLES["low"].values,
                            np.less_equal,
                            order=margin,
                        )[0]
                    ]["low"]
                    self.LAST_CANDLES["max"] = self.LAST_CANDLES.iloc[
                        argrelextrema(
                            self.LAST_CANDLES["high"].values,
                            np.greater_equal,
                            order=margin,
                        )[0]
                    ]["high"]
                    self.LAST_CANDLES.drop([0])
                    self.LAST_CANDLES = self.LAST_CANDLES.reset_index(drop=True)
                    self.new_candle = True

    def set_last_tick(self):
        if not self.TICK_QUEUE.empty():
            self.LAST_TICK = self.TICK_QUEUE.get()

    def update_trades(self):
        trades = self.get_trades()
        self.CLOCK.wait_clock()
        if trades is None:
            self.IN_POSITION = False

    def get_slope(self, prev_candle, c_candle):
        try:
            min_ctm, max_ctm, min_close, max_close = self.apiClient.get_max_values(
                self.symbol, self.long_ma
            )
            prev_ctm = min_max_norm(self.step_count - 1, min_ctm, max_ctm)
            c_ctm = min_max_norm(self.step_count, min_ctm, max_ctm)
            prev_short = min_max_norm(prev_candle["short_ma"], min_close, max_close)
            c_short = min_max_norm(c_candle["short_ma"], min_close, max_close)
            prev_long = min_max_norm(prev_candle["long_ma"], min_close, max_close)
            c_long = min_max_norm(c_candle["long_ma"], min_close, max_close)

            p1 = [prev_ctm, prev_short]
            p2 = [c_ctm, c_short]
            x1, y1 = p1
            x2, y2 = p2
            ma_short_slope = (y2 - y1) / (x2 - x1)

            p1 = [prev_ctm, prev_long]
            p2 = [c_ctm, c_long]
            x1, y1 = p1
            x2, y2 = p2
            ma_long_slope = (y2 - y1) / (x2 - x1)
            return ma_short_slope, ma_long_slope
        except Exception as e:
            raise SessionError(e)

    def get_angle_between_slopes(self, prev_candle, c_candle):
        try:
            m1, m2 = self.get_slope(prev_candle, c_candle)
            return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))
        except Exception as e:
            raise SessionError(e)

    def step(self):
        try:
            candles = self.LAST_CANDLES
            c_candle = candles.iloc[-1, :].to_dict()
            prev_candle = candles.iloc[-2, :].to_dict()
            if self.new_candle:
                self.new_candle = False
                in_down = self.down_crossover(prev_candle, c_candle)
                in_up = self.up_crossover(prev_candle, c_candle)
                if (in_up) or (in_down):
                    if self.IN_POSITION:
                        self.sell_position(position=self.ORDER_N, short=self.TRANS_TYPE)

                    if self.enough_money_live():
                        self.ORDER_N, self.TRANS_TYPE = self.buy_position(short=in_down)
                    else:
                        self.logger.info(f"Not enough money")
                        # TODO: NOT ENOUGH MONEY (END SESSION?)
                self.logger.info(
                    f"\nCrossover: {in_up}/{in_down}\nHist. Profits: {self.HIST_PROFITS}\nMax risks: {self.live_risks}"
                )
            else:
                # self.logger.info("NO STEP")
                pass

        except Exception as e:
            raise SessionError(e)


class BacktestCrossTrader(CrossTrader):
    def __init__(
        self, name, capital, max_risk, trader_type, logger=None, test=0, spread=0
    ) -> None:
        super().__init__(name, capital, max_risk, trader_type, logger=logger, test=test)

        self.spread = spread
        self.c_trade = None
        self.profits = []
        self.trades = []
        self.results = {}

    def fit(self, X, y=None, **kwargs):
        self.candles = X
        return self

    def score(self, X, y=None):
        return self.back_test(X)

    def test(self, X, parameters, test_period="1D", show=False):
        self.short_ma = parameters["short_ma"]
        self.long_ma = parameters["long_ma"]
        self.min_angle = parameters["min_angle"]
        self.out = parameters["out"]
        return self.back_test(X, test_period=test_period, show=show)

    def backTest(self, symbol_candle, sl_multiplier=0.5, tp_multiplier=2, min_angle=0):
        self.logger.info("Starting Backtest")
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.min_angle = min_angle

        start_date = (datetime.strptime("06-01-2023", "%d-%m-%Y")).strftime("%m-%d-%Y")
        end_date = (datetime.strptime("12-02-2023", "%d-%m-%Y")).strftime("%m-%d-%Y")
        candles = self.DB.get_candles_range(start_date=start_date, end_date=end_date)
        candles = candles[
            ["symbol", "ctmstring", "open", "high", "low", "close", "vol"]
        ]
        # if os.path.exists(f"results/synth_candles_02-12-2023.pkl"):
        #     with open(f"results/synth_candles_02-12-2023.pkl", "rb") as f:
        #         synth_candles = pkl.load(f)
        # else:
        #     synth_candles = generate_random_symbols(symbol_count=10, days=7)
        #     with open(f"results/synth_candles_02-12-2023.pkl", "wb") as f:
        #         pkl.dump(synth_candles, f, pkl.HIGHEST_PROTOCOL)
        # synth_symbols = list(synth_candles["symbol"].unique())
        normal_symbols = list(candles["symbol"].unique())
        # candles = candles.append(synth_candles)
        all_symbols = normal_symbols  # + synth_symbols
        total_profits = []

        if symbol[:5] != "SYNTH":
            self.SYMBOL_INFO = self.DB.get_symbol_info(symbol=symbol)
        else:
            self.SYMBOL_INFO = self.DB.get_symbol_info(symbol="EURUSD")
        self.symbol = symbol
        self.pip_size = 1.0 / np.power(10, self.SYMBOL_INFO["pipsPrecision"])
        symbol_candle = symbol_candle.reset_index(drop=True)
        self.spread = self.SYMBOL_INFO["spreadRaw"]
        self.long_ma = long_ma
        self.short_ma = short_ma
        self.live_risks = 0
        self.live_profits = 0
        self.c_trade = {}
        self.profits = [0]
        self.trades = []
        self.crossovers = []
        self.experiment = f"{symbol}-s{short_ma}-l{long_ma}"
        self.trade_count = 0

        if not os.path.exists(f"results/{self.symbol}/trades/"):
            os.makedirs(f"results/{self.symbol}/trades/")
        else:
            shutil.rmtree(f"results/{self.symbol}")
            os.makedirs(f"results/{self.symbol}/trades/")

        self.LAST_CANDLES = symbol_candle.reset_index()
        self.set_last_candle(margin=self.short_ma)
        for index in tqdm(range(self.long_ma, symbol_candle.shape[0], 15)):
            # for index in range(self.long_ma, symbol_candle.shape[0]):
            #     self.logger.info(
            #         f"\nSymbol: {symbol}\nClosed Trades: {len(self.trades)}\nOpen Trades: {len(self.c_trade.keys())}\nProfits: {np.sum(self.profits)}\nRisks: {np.sum(self.live_risks)}\nLive profits: {np.sum(self.live_profits)}"
            #     )
            if index > self.long_ma + 3:
                self.CURR_CANDLES = cp.deepcopy(
                    self.LAST_CANDLES.iloc[index - self.long_ma - 4 : index, :]
                ).reset_index()
                self.evaluate_risks()
                self.step_slopes()
                self.sell_by_stop_loss()
                self.sell_by_take_profit()
                self.update_stop_loss()
                self.sell_by_stop_loss()
                self.sell_by_take_profit()

        result = {
            "Symbol": symbol,
            "Profit": np.sum(self.profits),
            "Closed_trades": len(self.trades),
            "Avg_profit": np.mean(self.profits),
            "Min_profit": np.min(self.profits),
            "Max_profit": np.max(self.profits),
            "Std_profit": np.std(self.profits),
            "Median_profit": np.median(self.profits),
            "95p_profit": np.percentile(self.profits, q=0.95),
            "05p_profit": np.percentile(self.profits, q=0.05),
            "Trades": self.trades,
            "Crossover": self.crossovers,
            "candles": self.LAST_CANDLES,
        }
        total_profits.append(np.sum(self.profits))
        log_dict(result, self.logger)
        self.logger.info(f"\nTotal Profits: {total_profits} | {np.sum(total_profits)}")
        plot_stock_all_trades(
            self.LAST_CANDLES,
            self.trades,
            self.crossovers,
            symbol=f"{symbol}",
            params=f"s{short_ma}-l{long_ma}",
            profit=str(result["Profit"]),
            show=False,
        )
        if not os.path.exists(f"results/{symbol}"):
            os.mkdir(f"results/{symbol}")
        with open(
            f"results/{symbol}/{symbol}-s{short_ma}-l{long_ma}_results.pkl",
            "wb",
        ) as f:
            pkl.dump(result, f, pkl.HIGHEST_PROTOCOL)
        self.results[symbol] = result
        with open("results/full_results.pkl", "wb") as f:
            pkl.dump(self.results, f, pkl.HIGHEST_PROTOCOL)

    def randomTest(self):
        pass

    def buy_position(self, time, candle, short=False):
        max_std = self.LAST_CANDLES.iloc[: candle["index"], :]["close"].std()
        # maxes = self.CURR_CANDLES["max"].values
        # maxes = np.append(maxes, 0)
        # mins = self.CURR_CANDLES["min"].values
        # mins = np.append(mins, 100000)

        if short:
            order_type = "SHORT"
            price = candle["low"]
            # sl = np.max([np.nanmax(maxes[~np.isnan(maxes)]), price + (1 * max_std)])
            sl = price + (self.sl_multiplier * max_std)
            take_profit = np.min(
                [price - (sl - price), price - (self.tp_multiplier * max_std)]
            )
            price = price - self.spread

        else:
            order_type = "BUY"
            price = candle["high"]
            # sl = np.min([np.nanmin(mins[~np.isnan(mins)]), price - (1 * max_std)])
            sl = price - (self.sl_multiplier * max_std)
            take_profit = np.max(
                [price + (price - sl), price + (self.tp_multiplier * max_std)]
            )
            price = price + self.spread

        trailing = self.calculate_trailing(sl, price, multiplier=1)
        id = hash(time.strftime("%Y-%m-%d %H:%M:%S.%f"))
        self.c_trade[id] = {
            "Type": order_type,
            "open_index": candle["index"],
            "open_time": time,
            "open_price": price,
            "sl": sl,
            "tp": take_profit,
            "trailing": trailing,
            "profit": -self.spread,
            "all_sl": [],
        }
        pass

    def sell_position(self, time, candle, id, force_price=None, save=True, show=False):
        if self.c_trade[id]["Type"] == "SHORT":
            price = candle["high"]
            self.c_trade[id]["close_time"] = time
            if force_price is None:
                self.c_trade[id]["close_price"] = price
            else:
                self.c_trade[id]["close_price"] = force_price
            diff = self.c_trade[id]["open_price"] - self.c_trade[id]["close_price"]
        elif self.c_trade[id]["Type"] == "BUY":
            price = candle["low"]
            self.c_trade[id]["close_time"] = time
            if force_price is None:
                self.c_trade[id]["close_price"] = price
            else:
                self.c_trade[id]["close_price"] = force_price
            diff = self.c_trade[id]["close_price"] - self.c_trade[id]["open_price"]
        self.c_trade[id]["profit"] = self.calculate_profit(profit=diff)
        self.c_trade[id]["close_index"] = candle["index"]
        self.profits.append(self.c_trade[id]["profit"])
        self.trades.append(self.c_trade[id])
        if save:
            start_index = int(self.c_trade[id]["open_index"] - 100)
            end_index = int(self.c_trade[id]["close_index"] + 100)
            plot_stock_trade(
                df=self.LAST_CANDLES.iloc[start_index:end_index, :],
                trade=self.c_trade[id],
                id=id,
                symbol=self.experiment,
                profit=self.c_trade[id]["profit"],
                path=f"{self.symbol}/trades/trade_{self.trade_count}",
                show=show,
            )
            pass
        self.trade_count += 1
        del self.c_trade[id]

    def sell_by_stop_loss(self):
        candles = self.CURR_CANDLES
        c_candle = candles.iloc[-1, :].to_dict()
        # Sell by stop_loss
        if len(self.c_trade.keys()) != 0:
            key_list = list(self.c_trade.keys())
            for trade_id in key_list:
                trade = self.c_trade[trade_id]
                if trade["Type"] == "SHORT":
                    if c_candle["low"] >= trade["sl"]:
                        self.sell_position(
                            time=c_candle["ctmstring"],
                            candle=c_candle,
                            id=trade_id,
                            force_price=trade["sl"],
                        )
                elif trade["Type"] == "BUY":
                    if c_candle["high"] <= trade["sl"]:
                        self.sell_position(
                            time=c_candle["ctmstring"],
                            candle=c_candle,
                            id=trade_id,
                            force_price=trade["sl"],
                        )

    def sell_by_take_profit(self):
        candles = self.CURR_CANDLES
        c_candle = candles.iloc[-1, :].to_dict()

        if len(self.c_trade.keys()) != 0:
            key_list = list(self.c_trade.keys())
            for trade_id in key_list:
                trade = self.c_trade[trade_id]
                if trade["Type"] == "SHORT":
                    short = True
                    if c_candle["low"] < trade["tp"]:
                        self.sell_position(
                            time=c_candle["ctmstring"],
                            candle=c_candle,
                            id=trade_id,
                            force_price=trade["tp"],
                        )
                elif trade["Type"] == "BUY":
                    short = False
                    if c_candle["high"] > trade["tp"]:
                        self.sell_position(
                            time=c_candle["ctmstring"],
                            candle=c_candle,
                            id=trade_id,
                            force_price=trade["tp"],
                        )

    def update_stop_loss(self):
        candles = self.CURR_CANDLES
        c_candle = candles.iloc[-1, :].to_dict()

        if len(self.c_trade.keys()) != 0:
            for trade_id in self.c_trade.keys():
                if self.c_trade[trade_id]["Type"] == "SHORT":
                    sl = self.c_trade[trade_id]["sl"]
                    price = c_candle["high"]
                    cur_trailing = self.calculate_trailing(sl=sl, price=price)
                    trailing = self.c_trade[trade_id]["trailing"]
                    self.c_trade[trade_id]["profit"] = self.calculate_profit(
                        profit=(self.c_trade[trade_id]["open_price"] - price)
                    )
                    if cur_trailing > trailing and price < sl:
                        new_sl = price + (trailing * self.pip_size)
                        self.c_trade[trade_id]["all_sl"].append(sl)
                        self.c_trade[trade_id]["sl"] = new_sl

                if self.c_trade[trade_id]["Type"] == "BUY":
                    sl = self.c_trade[trade_id]["sl"]
                    price = c_candle["low"]
                    cur_trailing = self.calculate_trailing(sl=sl, price=price)
                    trailing = self.c_trade[trade_id]["trailing"]
                    self.c_trade[trade_id]["profit"] = self.calculate_profit(
                        profit=(price - self.c_trade[trade_id]["open_price"])
                    )
                    if cur_trailing > trailing and price > sl:
                        new_sl = price - (trailing * self.pip_size)
                        self.c_trade[trade_id]["all_sl"].append(sl)
                        self.c_trade[trade_id]["sl"] = new_sl

    def calculate_trailing(self, sl, price, multiplier=1):
        return int(
            (
                np.abs(sl - price)
                // (1.0 / np.power(10, self.SYMBOL_INFO["pipsPrecision"]))
            )
            / multiplier
        )

    def set_min_max(self, margin=5):
        self.LAST_CANDLES["min"] = self.LAST_CANDLES.iloc[
            argrelextrema(
                self.LAST_CANDLES["low"].values,
                np.less_equal,
                order=margin,
            )[0]
        ]["low"]
        self.LAST_CANDLES["max"] = self.LAST_CANDLES.iloc[
            argrelextrema(
                self.LAST_CANDLES["high"].values,
                np.greater_equal,
                order=margin,
            )[0]
        ]["high"]
        self.LAST_CANDLES["min"] = pd.to_numeric(self.LAST_CANDLES["min"])
        self.LAST_CANDLES["max"] = pd.to_numeric(self.LAST_CANDLES["max"])

    def set_rolling_means(self):
        self.LAST_CANDLES["short_ma"] = (
            self.LAST_CANDLES["close"].rolling(window=self.short_ma).mean()
        )
        self.LAST_CANDLES["long_ma"] = (
            self.LAST_CANDLES["close"].rolling(window=self.long_ma).mean()
        )

    def set_last_candle(self, margin=None):
        self.set_rolling_means()
        # self.set_min_max(margin)
        self.LAST_CANDLES = self.LAST_CANDLES.reset_index(drop=True)
        self.new_candle = True

    def evaluate_risks(self):
        max_losses = []
        live_wins = []
        if self.c_trade is not None:
            for trade_id in self.c_trade.keys():
                trade = self.c_trade[trade_id]
                open_price = trade["open_price"]
                stop_loss = trade["sl"]
                volume = 0.01
                max_loss = self.test_risk(open_price, stop_loss, volume)
                max_losses.append(max_loss)
                live_wins.append(trade["profit"])
            self.live_risks = max_losses
            self.live_profits = live_wins
        else:
            self.live_risks = 0
            self.live_profits = 0

    def enough_money_live(self):
        if (
            (np.sum(self.live_risks) - np.sum(self.live_profits))
            < ((self.CAPITAL + np.sum(self.profits)) * self.max_risk)
            and (np.sum(self.profits) > -25)
            and (np.sum(self.live_profits) > -25)
        ):
            return True
        else:
            return False

    def test_risk(self, price, stop_loss, volume):
        pip_size = 1.0 / np.power(10, self.SYMBOL_INFO["pipsPrecision"])
        contract_size = self.SYMBOL_INFO["contractSize"]
        if self.SYMBOL_INFO["currencyProfit"] == "USD":
            pip_price = 10 * volume
        else:
            pip_price = (pip_size / price) * contract_size * volume
        max_pip_drop = np.abs(stop_loss - price) // pip_size
        max_loss = max_pip_drop * pip_price
        return max_loss

    def calculate_profit(self, profit, volume=0.01):
        pip_size = 1.0 / np.power(10, self.SYMBOL_INFO["pipsPrecision"])
        pip_price = 10 * volume
        real_profit = (profit // pip_size) * pip_price
        return real_profit

    def find_min_max(self, margin=5, dist=0):
        if (
            self.CURR_CANDLES.iloc[-1, :]["ctmstring"].strftime("%Y-%m-%d %H:%M")
            == "2023-02-13 13:24"
        ):
            pass
        mins = self.CURR_CANDLES.iloc[
            argrelextrema(
                self.CURR_CANDLES["short_ma"].values,
                np.less_equal,
                order=margin,
            )[0]
        ]["short_ma"]
        maxes = self.CURR_CANDLES.iloc[
            argrelextrema(
                self.CURR_CANDLES["short_ma"].values,
                np.greater_equal,
                order=margin,
            )[0]
        ]["short_ma"]
        short_ma = self.CURR_CANDLES["short_ma"].values[-1]
        long_ma = self.CURR_CANDLES["long_ma"].values[-1]
        if (5 < (self.CURR_CANDLES.shape[0] - mins.index[-1]) <= dist) and (
            short_ma < long_ma
        ):
            return True, False
        elif (5 < (self.CURR_CANDLES.shape[0] - maxes.index[-1]) <= dist) and (
            short_ma > long_ma
        ):
            return False, True
        else:
            return False, False

    def get_slope(self, prev_candle, c_candle):
        min_ctm, max_ctm = 0, self.long_ma
        min_close, max_close = np.min(self.CURR_CANDLES["low"]), np.max(
            self.CURR_CANDLES["high"]
        )

        prev_ctm = min_max_norm(prev_candle["index"], min_ctm, max_ctm)
        c_ctm = min_max_norm(c_candle["index"], min_ctm, max_ctm)
        prev_short = min_max_norm(prev_candle["short_ma"], min_close, max_close)
        c_short = min_max_norm(c_candle["short_ma"], min_close, max_close)
        prev_long = min_max_norm(prev_candle["long_ma"], min_close, max_close)
        c_long = min_max_norm(c_candle["long_ma"], min_close, max_close)

        p1 = [prev_ctm, prev_short]
        p2 = [c_ctm, c_short]
        x1, y1 = p1
        x2, y2 = p2
        ma_short_slope = (y2 - y1) / (x2 - x1)

        p1 = [prev_ctm, prev_long]
        p2 = [c_ctm, c_long]
        x1, y1 = p1
        x2, y2 = p2
        ma_long_slope = (y2 - y1) / (x2 - x1)
        return ma_short_slope, ma_long_slope

    def get_absolute_angles(self, prev_candle, c_candle):
        ma_short_slope, ma_long_slope = self.get_slope(prev_candle, c_candle)
        angle_short = np.degrees(
            np.arctan((ma_short_slope - 0) / (1 + (ma_short_slope * 0)))
        )
        angle_long = np.degrees(
            np.arctan((ma_long_slope - 0) / (1 + (ma_long_slope * 0)))
        )
        return angle_short, angle_long

    def get_angle_between_slopes(self, prev_candle, c_candle):
        m1, m2 = self.get_slope(prev_candle, c_candle)
        return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))

    def buy_slope(self, prev_candle, c_candle):
        buy_short = False
        buy_long = False
        short_a, long_a = self.get_absolute_angles(prev_candle, c_candle)
        if np.abs(c_candle["short_ma"] - c_candle["long_ma"]) > self.spread:
            if short_a < -(self.min_angle * 2) and long_a < -(self.min_angle / 2):
                # trend is negative
                buy_short = False
                buy_long = True
            elif short_a > (self.min_angle * 2) and long_a > (self.min_angle / 2):
                # trend is positive
                buy_short = True
                buy_long = False
            else:
                buy_short = False
                buy_long = False
        else:
            buy_short = False
            buy_long = False
        return buy_short, buy_long

    def step(self):
        candles = self.CURR_CANDLES
        c_candle = candles.iloc[-1, :].to_dict()
        prev_candle = candles.iloc[-2, :].to_dict()
        if self.new_candle:
            in_down = self.down_crossover(prev_candle, c_candle)
            in_up = self.up_crossover(prev_candle, c_candle)
            if (in_up) or (in_down):
                id_list = list(self.c_trade.keys())
                if len(id_list) != 0:
                    for trade_id in id_list:
                        self.sell_position(
                            time=c_candle["ctmstring"], candle=c_candle, id=trade_id
                        )
                if self.enough_money_live():
                    self.buy_position(
                        time=c_candle["ctmstring"], candle=c_candle, short=in_down
                    )
                else:
                    pass
            self.crossovers.append({"time": c_candle["ctmstring"], "cross-type": in_up})
        else:
            pass

    def step_slopes(self):
        candles = self.CURR_CANDLES
        c_candle = candles.iloc[-1, :].to_dict()
        prev_candle = candles.iloc[-2, :].to_dict()
        if self.new_candle:
            in_up, in_down = self.buy_slope(prev_candle, c_candle)
            if (in_up) or (in_down):
                id_list = list(self.c_trade.keys())
                if len(id_list) != 0:
                    for trade_id in id_list:
                        if (self.c_trade[trade_id]["Type"] == "BUY" and in_down) or (
                            (self.c_trade[trade_id]["Type"] == "SHORT" and in_up)
                        ):
                            self.sell_position(
                                time=c_candle["ctmstring"],
                                candle=c_candle,
                                id=trade_id,
                                show=False,
                            )

                if self.enough_money_live():  # and (len(self.c_trade.keys()) < 10):
                    id_list = list(self.c_trade.keys())
                    if len(id_list) == 0:
                        self.buy_position(
                            time=c_candle["ctmstring"], candle=c_candle, short=in_down
                        )
                else:
                    id_list = list(self.c_trade.keys())
                    if len(id_list) != 0:
                        for trade_id in id_list:
                            self.sell_position(
                                time=c_candle["ctmstring"],
                                candle=c_candle,
                                id=trade_id,
                                show=False,
                            )
                            if self.enough_money_live():
                                break
            self.crossovers.append({"time": c_candle["ctmstring"], "cross-type": in_up})
        else:
            pass

    def step_max_min(self):
        candles = self.CURR_CANDLES
        c_candle = candles.iloc[-1, :].to_dict()
        prev_candle = candles.iloc[-2, :].to_dict()
        if self.new_candle:
            in_up, in_down = self.find_min_max(
                margin=int(self.long_ma / 5), dist=self.short_ma * 2
            )

            if (in_up) or (in_down):
                id_list = list(self.c_trade.keys())
                if len(id_list) != 0:
                    for trade_id in id_list:
                        self.sell_position(
                            time=c_candle["ctmstring"], candle=c_candle, id=trade_id
                        )
                if self.enough_money_live():
                    self.buy_position(
                        time=c_candle["ctmstring"], candle=c_candle, short=in_down
                    )
                else:
                    pass
            self.crossovers.append({"time": c_candle["ctmstring"], "cross-type": in_up})
        else:
            pass


class TeamOfTraders:
    def __init__(
        self, team_name, capital, max_risk, trader_type, c_logger=None
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.c_logger = c_logger
        if c_logger is not None:
            self.logger = c_logger.setup_logging(self.logger, __name__)
        self.team_name = team_name
        self.team_capital = capital
        self.team_max_risk = max_risk
        self.team_trader_type = trader_type
        self.trader_team = {}

    def add_trader(self, trader):
        self.trader_team[trader.name] = trader

    def del_trader(self, trader):
        removed = self.trader_team.pop(trader.name, None)
        if removed is None:
            self.logger.info(f"Trader removed: {trader.name}")
        else:
            self.logger.info(f"Trader not found")

    def list_traders(self):
        list_of_traders_str = "\n"
        for cont, key in enumerate(self.trader_team.keys()):
            list_of_traders_str += f"{cont}: {key}\n"
        self.logger.info(list_of_traders_str + "\n")

    def get_live_risk(self):
        live_risk = 0
        for key in self.trader_team.keys():
            live_risk += self.trader_team[key].live_risk
        return live_risk
