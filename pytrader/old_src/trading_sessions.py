import logging
import os
import threading

import pandas as pd
from creds import creds
from logger_settings import *
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sqlalchemy import create_engine
from tqdm import tqdm
from trader_api_utils import *
from trader_db_utils import *
from trader_utils import *
from xAPIConnector import *


class TradingSession:
    def __init__(
        self,
        name,
        capital,
        symbol,
        short_ma,
        long_ma,
        profit_exit,
        loss_exit,
        min_angle,
        apiClient,
        offline=False,
        test=False,
        tick_queue=None,
        candle_queue=None,
        clock=None,
    ) -> None:

        # Session params
        self.name = name
        self.capital = capital
        self.symbol = symbol
        if test:
            self.short_ma = 1
            self.long_ma = 10
            self.profit_exit = 0.0001
            self.loss_exit = 0
            self.min_angle = 0
            self.step_count = self.long_ma
        else:
            self.short_ma = short_ma
            self.long_ma = long_ma
            self.profit_exit = profit_exit
            self.loss_exit = loss_exit
            self.min_angle = min_angle
            self.step_count = long_ma

        self.offline = offline
        self.apiClient = apiClient
        self.tick_queue = tick_queue
        self.candle_queue = candle_queue
        self.clock = clock

        # Symbol information
        symbol_info = get_symbol_info(self.symbol)
        self.short_enabled = symbol_info["shortSelling"]
        self.leverage = symbol_info["leverage"]
        self.contractSize = symbol_info["contractSize"]
        self.lotStep = symbol_info["lotStep"]
        self.lotMin = symbol_info["lotMin"]
        self.currency = symbol_info["currency"]
        self.volume = symbol_info["lotMin"]
        self.symbol_info = symbol_info

        self.last_tick = None
        self.last_ticks = []

        self.in_down_crossover = False
        self.in_up_crossover = False
        self.is_bought = False
        self.is_short = False
        self.profit = None
        self.prev_profit = None
        self.hist_profits = []
        self.potential_profits = {}
        self.order_numbers = []
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        self.in_position = False
        # insert_trade_session(self.name, self.symbol, -1)

    def start(self):
        # watchDog = threading.Thread(target=self.watchdog, args=("ask",))
        # watchDog.setDaemon(True)
        # watchDog.start()

        tick_reader = threading.Thread(target=self.read_last_tick)
        tick_reader.setDaemon(True)
        tick_reader.start()

    def step(self):
        try:
            self.step_count += 1
            if self.step_count > self.long_ma and self.last_tick is not None:
                candles = get_last_two_candles(self.symbol, self.short_ma, self.long_ma)
                c_tick = self.last_tick
                c_candle = candles.iloc[0, :]
                prev_candle = candles.iloc[-1, :]
                angle_diff = self.get_angle_between_slopes(prev_candle, c_candle)
                calculate_volume = threading.Thread(
                    target=self.get_max_volume, args=(c_tick["ask"],)
                )
                calculate_volume.start()

                in_down = self.down_crossover(prev_candle, c_candle)
                in_up = self.down_crossover(prev_candle, c_candle)

                if (in_up) or (in_down):
                    if self.in_position:
                        self.sell_position(self.order_n, self.trans_type)
                        if self.enough_money(c_tick):
                            self.order_n, self.trans_type = self.buy_position(
                                short=in_down
                            )
                            self.in_position = True
                        else:
                            logger.info(f"Not enough money")
                            # TODO: NOT ENOUGH MONEY (END SESSION?)
                    else:
                        if self.enough_money(c_tick):
                            self.order_n, self.trans_type = self.buy_position(
                                short=in_down
                            )
                            self.in_position = True
                        else:
                            logger.info(f"Not enough money")
                            # TODO: NOT ENOUGH MONEY (END SESSION?)
                else:
                    logger.info(f"Angle to small: {angle_diff} <> {self.min_angle}")

                logger.info(
                    f"\nCrossover: {in_up}/{in_down}\nAngle: {angle_diff} <> {self.min_angle}\nHist. Profits: {self.hist_profits}"
                )
        except Exception as e:
            raise SessionError(e)

    def store_vars(self):
        vars = json.dumps(self.__dict__, cls=CustomJSONizer)
        if not os.path.exists(data_path + self.symbol):
            os.mkdir(data_path + self.symbol)
        with open(
            data_path
            + self.symbol
            + "/"
            + "class_attrs_s"
            + str(self.step_count)
            + ".json",
            "w",
        ) as f:
            f.write(vars)

    def load_vars(self):
        with open(data_path + self.symbol + "class_attrs.json") as f:
            vars = json.load(f)
        for key, value in vars:
            print(f"{key}: {value}")

    def sell_position(self, position, trans_type):
        try:
            logger.info("\n******\nSELL\n******")
            c_tick = self.last_tick
            if trans_type:
                buy_trans = TransactionSide.BUY
            else:
                buy_trans = TransactionSide.SELL
            try:
                del self.potential_profits[position]
            except KeyError:
                pass
            if position is not None:
                self.sell_order_n = self.apiClient.sell(
                    name=self.name,
                    buy_trans=buy_trans,
                    position=position,
                    c_tick=c_tick,
                    volume=self.volume,
                )
            self.order_n = None
            self.hist_profits.append(self.get_profit(position))
            logger.info(f"Hist. Profits: {self.hist_profits}")
        except Exception as e:
            raise SessionError(e)

    def buy_position(self, short=False):
        try:
            c_tick = self.last_tick
            if short:
                self.is_short = short
                self.trade_type = "SHORT"
                buy_trans = TransactionSide.SELL
                sl = np.round(
                    c_tick["ask"] + (self.symbol_info["tickSize"] * 1),
                    self.symbol_info["precision"],
                )
                logger.info("\n******\nSHORT\n******")
            else:
                self.is_bought = True
                self.trade_type = "BUY"
                buy_trans = TransactionSide.BUY
                sl = np.round(
                    c_tick["ask"] - (self.symbol_info["tickSize"] * 1),
                    self.symbol_info["precision"],
                )
                logger.info("\n******\nBUY\n******")

            logger.info(f"\n\nBUY PRICE: {c_tick['ask']} | SL: {sl}")

            order_n = self.apiClient.buy(
                name=self.name,
                buy_trans=buy_trans,
                c_tick=c_tick,
                volume=self.volume,
                sl=sl,
            )
            trade = self.apiClient.get_trade(opened=True, order_n=order_n)
            return trade["position"], trade["cmd"]
        except Exception as e:
            raise SessionError(e)

    def enough_money(self, c_tick):
        try:
            if np.sum(self.hist_profits) + self.capital > (
                self.calculate_position(price=c_tick["ask"], vol=self.volume)
            ):
                return True
            else:
                return False
        except Exception as e:
            raise SessionError(e)

    def crossover(self, prev_candle, c_candle):
        try:
            if self.up_crossover(prev_candle, c_candle):
                return True
            elif self.down_crossover(prev_candle, c_candle):
                return False
        except Exception as e:
            raise SessionError(e)

    def up_crossover(self, prev_candle, c_candle):
        try:
            if (prev_candle["short_ma"] < prev_candle["long_ma"]) and (
                c_candle["short_ma"] > c_candle["long_ma"]
            ):
                return True
            else:
                return False
        except Exception as e:
            raise SessionError(e)

    def down_crossover(self, prev_candle, c_candle):
        try:
            if (prev_candle["short_ma"] > prev_candle["long_ma"]) and (
                c_candle["short_ma"] < c_candle["long_ma"]
            ):
                return True
            else:
                return False
        except Exception as e:
            raise SessionError(e)

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

    def get_max_volume(self, price):
        try:
            n = -1
            buy_price = 0
            while buy_price < (np.sum(self.hist_profits) + self.capital):
                n += 1
                buy_price = self.calculate_position(
                    price=price, vol=self.lotMin + n * self.lotStep
                )
            volume = self.lotMin + ((n - 1) * self.lotStep)
            self.volume = volume
        except Exception as e:
            raise SessionError(e)

    def calculate_position(self, price, vol):
        try:
            position = self.contractSize * vol * price
            margin = position * (self.leverage / 100)
            # convert_currency(margin, self.currency)
            return margin
        except Exception as e:
            raise SessionError(e)

    def get_profit(self, buy_position):
        try:
            trade = self.apiClient.get_trade(opened=False, order_n=buy_position)
            return trade["profit"]
        except Exception as e:
            raise SessionError(e)

    def evaluate_profit(self, timestamp):
        try:
            if self.profit is not None:
                prev_profit = self.profit
            else:
                prev_profit = 0
            if self.is_short or self.is_bought:
                trade = self.apiClient.get_trade(opened=True, order_n=self.order_n)
                if trade is not None:
                    self.profit = trade["profit"]
                    self.prev_profit = prev_profit
                else:
                    self.profit = 0
                    self.prev_profit = 0
                    self.profits = []
                    self.is_short = False
                    self.is_bought = False

            else:
                prev_profit = 0
                self.profit = 0
                self.prev_profit = 0
            return self.profit, prev_profit
        except Exception as e:
            raise SessionError(e)

    def read_last_tick(self):
        try:
            while 1:
                if not self.tick_queue.empty():
                    tick = self.tick_queue.get()
                    self.last_tick = tick
                    self.set_last_ticks(tick)
                time.sleep(0.1)
        except Exception as e:
            raise SessionError(e)

    def set_last_ticks(self, tick):
        try:
            if len(self.last_ticks) > 30:
                self.last_ticks.pop(0)
                self.last_ticks.append(tick)
            else:
                self.last_ticks.append(tick)
        except Exception as e:
            raise SessionError(e)

    def watchdog(self, key):
        try:
            logger.info("Watchdog")
            while 1:
                key_val_list = []
                trades = self.apiClient.get_trades()
                if self.last_ticks is not None and trades is not None:
                    if len(self.last_ticks) > 9:
                        for tick in self.last_ticks:
                            key_val_list.append(tick[key])
                        mean_key_val = np.sum(key_val_list) / len(key_val_list)
                        for trade in trades:
                            if trade["open_time"] < self.last_ticks[0]["timestamp"]:
                                if (
                                    (mean_key_val < trade["open_price"])
                                    and trade["cmd"] == 0
                                ) or (
                                    (mean_key_val > trade["open_price"])
                                    and trade["cmd"] == 1
                                ):
                                    logger.info(
                                        f"\n***********************\nWATCHDOG\n***********************"
                                    )
                                    self.sell_position(trade["position"], trade["cmd"])
                                else:
                                    if (
                                        str(trade["position"])
                                        not in self.potential_profits.keys()
                                    ):
                                        self.potential_profits[trade["position"]] = []
                                    self.potential_profits[trade["position"]].append(
                                        trade["profit"]
                                    )
                                    list_of_profits = self.potential_profits[
                                        trade["position"]
                                    ]
                                    if len(list_of_profits) > 0:
                                        last_percentage = (
                                            self.calculate_last_percentage(
                                                list_of_profits,
                                                trade["open_price"],
                                                trade["cmd"],
                                            )
                                        )
                                        if last_percentage > self.profit_exit:
                                            logger.info(
                                                f"\nEvolution list ({last_percentage}): {list_of_profits}"
                                            )
                                            self.sell_position(
                                                trade["position"], trade["cmd"]
                                            )
                                        time.sleep(0.2)

                time.sleep(0.2)
        except Exception as e:
            raise SessionError(e)

    def calculate_last_percentage(self, list_of_profits, open_price, trans_type):
        try:
            if len(list_of_profits) == 1:
                last = list_of_profits[0]
                before_last = open_price
            else:
                last = list_of_profits[-1]
                before_last = np.max(list_of_profits[:-1])

            diff = before_last - last
            percentage = diff / before_last
            return percentage
        except Exception as e:
            raise SessionError(e)


class TradingLiveSession(TradingSession):
    pass


class TradingOflineSession(TradingSession):
    pass
