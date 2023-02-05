import os
import pandas as pd
from trader_utils import *
from sqlalchemy import create_engine
from xAPIConnector import *
from creds import creds
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import threading
from trader_db_utils import *
from trader_api_utils import *
from tqdm import tqdm
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

minimun_trade = 10
minimun_trade = 10


class Trader:
    def __init__(self, name, capital, max_risk, trader_type) -> None:
        self.user = creds[name.split(":")[0]]["user"]
        self.passw = creds[name.split(":")[0]]["passw"]
        self.name = name
        self.capital = capital
        self.max_risk = max_risk
        self.trader_type = trader_type
        traderid = get_trader_id(name)
        if traderid is None:
            traderid = insert_trader(name, capital, max_risk)
            logger.info(f"Trader created id: {traderid}")
        else:
            logger.info(f"Trader exists id: {traderid}")
        self.traderid = traderid

        self.apiSession = ApiSessionManager(self.user, self.passw)
        self.apiSession.set_apiClient()

    def look_for_suitable_symbols_v1(self, df):
        # TODO look for suitable symbols
        # Look for symbols with:
        #   - Volatility bigger than spread [DONE LATER]
        #   - Trader_type products
        #   - Ask comparable to our max_risk

        df = df[df["categoryName"] == self.trader_type]
        df["min_price"] = (
            df["ask"] * df["lotMin"] * df["contractSize"] * (df["leverage"] / 100)
        )
        df = df[df["min_price"] <= (self.capital * self.max_risk)]
        df["spread_percentage"] = (df["ask"] - df["bid"]) / df["ask"]
        df = df.sort_values(by=["spread_percentage"])
        return df

    def update_stocks(self, df, period, days=14):
        start_date = datetime.now() - timedelta(days=days)
        for symbol in tqdm(list(df["symbol"])):
            if not get_candles_today():
                self.apiSession.store_past_candles(symbol, start_date, period)

    def evaluate_stocks(self, date, threshold=25):
        symbols = get_distict_symbols()
        for symbol in symbols:
            symbol = symbol[0]
            symbol_info = get_symbol_info(symbol)
            symbol_stats = get_symbol_stats(symbol, date)
            if symbol_info["spreadRaw"] * (threshold) < symbol_stats["std_close"]:
                score = symbol_stats["std_close"] / symbol_info["spreadRaw"]
                spread = symbol_info["spreadRaw"]
                logger.info(
                    f"CHOSEN\nSymbol {symbol}: {symbol_stats['std_close']} / {symbol_info['spreadRaw']} = {score}"
                )
                insert_params(
                    day=datetime.today(),
                    symbol=symbol,
                    score=score,
                    short_ma=1,
                    long_ma=10,
                    profit_exit=spread,
                    loss_exit=0,
                    min_angle=20,
                )

            elif symbol_info["spreadRaw"] < symbol_stats["std_close"]:
                logger.info(
                    f"GOOD\nSymbol {symbol}: {symbol_stats['std_close']} / {symbol_info['spreadRaw']} = {symbol_stats['std_close'] / symbol_info['spreadRaw']}"
                )
            else:
                logger.info(
                    f"BAD\nSymbol {symbol}: {symbol_stats['std_close']} / {symbol_info['spreadRaw']} = {symbol_stats['std_close'] / symbol_info['spreadRaw']}"
                )

    def start_trading_session(self, params, test=False):
        self.apiSession.load_missed_candles(params["symbol"], params["long_ma"])
        session = TradingSession(
            name=self.name,
            capital=self.capital * self.max_risk,
            symbol=params["symbol"],
            short_ma=params["short_ma"],
            long_ma=params["long_ma"],
            profit_exit=params["profit_exit"],
            loss_exit=params["loss_exit"],
            min_angle=params["min_angle"],
            apiClient=self.apiSession,
            test=test,
        )
        self.session = session
        self.apiSession.set_session(session)


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
        offline=False,
        apiClient=None,
        test=False,
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
            self.step_count = long_ma
        else:
            self.short_ma = short_ma
            self.long_ma = long_ma
            self.profit_exit = profit_exit
            self.loss_exit = loss_exit
            self.min_angle = min_angle
            self.step_count = long_ma * 2
        self.offline = offline
        self.apiClient = apiClient

        # Symbol information
        symbol_info = get_symbol_info(self.symbol)
        self.short_enabled = symbol_info["shortSelling"]
        self.leverage = symbol_info["leverage"]
        self.contractSize = symbol_info["contractSize"]
        self.lotStep = symbol_info["lotStep"]
        self.lotMin = symbol_info["lotMin"]
        self.currency = symbol_info["currency"]
        self.symbol_info = symbol_info

        self.last_tick = None
        self.last_ticks = []

        self.in_down_crossover = True
        self.in_up_crossover = False
        self.is_bought = False
        self.is_short = False
        self.profit = None
        self.prev_profit = None
        self.hist_profits = []
        self.potential_profits = []
        self.buy_price = None
        self.open_price = None
        self.current_position = None
        self.current_price = None
        self.buy_volume = None
        self.order_n = None
        self.sell_order_n = None
        self.volume = symbol_info["lotMin"]
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        insert_trade_session(self.name, self.symbol, self.is_bought, self.is_short)
        watchDog = threading.Thread(target=self.get_last_ticks_mean, args=("ask",))
        watchDog.setDaemon(True)
        watchDog.start()

    def step(self):
        self.step_count += 1
        if self.step_count > self.long_ma:
            candles = get_last_two_candles(self.symbol, self.short_ma, self.long_ma)
            timestamp = int(candles["ctm"].values[0])
            c_tick = self.apiClient.get_ticks(timestamp, self.symbol)
            c_candle = candles.iloc[0, :]
            prev_candle = candles.iloc[-1, :]
            angle_diff = self.get_angle_between_slopes(prev_candle, c_candle)
            calculate_volume = threading.Thread(
                target=self.get_max_volume, args=(c_tick["ask"][0],)
            )
            calculate_volume.start()
            profit, prev_profit = self.evaluate_profit(timestamp)

            if c_candle["short_ma"] < c_candle["long_ma"]:
                self.in_down_crossover = True
                self.in_up_crossover = False
            elif c_candle["short_ma"] > c_candle["long_ma"]:
                self.in_up_crossover = True
                self.in_down_crossover = False

            if self.is_bought:
                diff = c_candle["low"] - prev_candle["low"]
                profit, prev_profit = self.evaluate_profit(timestamp)
                self.potential_profits.append(
                    np.round(profit, self.symbol_info["precision"])
                )

                logger.info(
                    f"Is Bought\nBuy price {self.buy_price}\nProfit {profit}\nDiff {diff}\nPotential profits {self.potential_profits})"
                )

                if profit < 0:
                    logger.info(
                        f"\nProfit ({profit} < {-(self.buy_price * self.loss_exit)})"
                    )
                    self.sell_position()

                elif profit > 0:
                    if (prev_profit) > ((profit) * (1 + self.profit_exit)):
                        logger.info(
                            f"\nProfit ({profit} > {(profit) * (1 + self.profit_exit)})"
                        )
                        self.sell_position()

            if self.is_short:
                diff = prev_candle["low"] - c_candle["low"]
                profit, prev_profit = self.evaluate_profit(timestamp)
                self.potential_profits.append(
                    np.round(profit, self.symbol_info["precision"])
                )

                logger.info(
                    f"Is Short\nBuy price {self.buy_price}\nProfit {profit}\nDiff {diff}\nPotential profits {self.potential_profits}"
                )

                if profit < 0:
                    logger.info(
                        f"\nProfit ({profit} < {-(self.buy_price * self.loss_exit)})"
                    )
                    self.sell_position()

                elif profit > 0:
                    self.potential_profits.append(diff)
                    if (prev_profit) > ((profit) * (1 + self.profit_exit)):
                        logger.info(
                            f"\nProfit ({profit} > {(profit) * (1 + self.profit_exit)})"
                        )
                        self.sell_position()

            if self.crossover(prev_candle, c_candle):
                if angle_diff > self.min_angle:
                    if self.enough_money(c_tick):
                        if not self.is_bought and not self.is_short:
                            self.buy_position()
                        else:
                            pass
                            # TODO: implement multitrade
                    else:
                        logger.info(f"Not enough money")
                        pass
                        # TODO: NOT ENOUGH MONEY (END SESSION?)
            else:
                if self.short_enabled:
                    if angle_diff < -self.min_angle:

                        if self.enough_money(c_tick):
                            if not self.is_bought and not self.is_short:
                                self.buy_position(short=True)
                            else:
                                pass
                                # TODO: implement multitrade
                        else:
                            logger.info(f"Not enough money")
                            pass
                            # TODO: NOT ENOUGH MONEY (END SESSION?)

            logger.info(
                f"\n**************************\nStep count ({self.step_count}) > long_ma ({self.long_ma})\n Is bought: {self.is_bought}\n Is short: {self.is_short}\n Up cross: {self.in_up_crossover}\n Down cross: {self.in_down_crossover}\n Previous:\n     Time: {prev_candle['ctmstring']}\n      Short_ma: {prev_candle['short_ma']}\n       Long_ma: {prev_candle['long_ma']}\n Current:\n      Time: {c_candle['ctmstring']}\n            Short_ma: {c_candle['short_ma']}\n            Long_ma: {c_candle['long_ma']}\n Angle: {angle_diff}\n Min_angle: {self.min_angle}\n Entry Price: {self.open_price}\n Entry Position: {self.buy_price}\n Current Price: {self.current_price}\n Current Position: {self.current_position}\n Profit: {self.profit}\n Prev. Profit: {self.prev_profit}\n Pot. Profits: {self.potential_profits}\n\nHistoric Profits: {self.hist_profits}\n**************************"
            )
        self.store_vars()

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

    def sell_position(self):
        logger.info("\n******\nSELL\n******")
        c_tick = self.last_tick
        if self.is_short:
            buy_trans = TransactionSide.BUY
        else:
            buy_trans = TransactionSide.SELL
        self.is_bought = False
        self.is_short = False
        self.potential_profits = [0]
        self.buy_volume = None
        self.buy_price = None

        position = self.apiClient.get_trade(opened=True, order_n=self.order_n)
        if position is not None:
            self.sell_order_n = self.apiClient.sell(
                name=self.name,
                buy_trans=buy_trans,
                position=position,
                c_tick=c_tick,
                volume=self.volume,
            )
        self.order_n = None
        self.hist_profits.append(self.get_profit())

    def buy_position(self, short=False):
        c_tick = self.last_tick
        if short:
            self.is_short = short
            self.trade_type = "SHORT"
            buy_trans = TransactionSide.SELL
            sl = np.round(
                c_tick["ask"] + (self.symbol_info["tickSize"] * 5),
                self.symbol_info["precision"],
            )
            logger.info("\n******\nSHORT\n******")
        else:
            self.is_bought = True
            self.trade_type = "BUY"
            buy_trans = TransactionSide.BUY
            sl = np.round(
                c_tick["ask"] - (self.symbol_info["tickSize"] * 5),
                self.symbol_info["precision"],
            )
            logger.info("\n******\nBUY\n******")

        logger.info(f"\n\nBUY PRICE: {c_tick['ask']} | SL: {sl}")

        self.order_n = self.apiClient.buy(
            name=self.name,
            buy_trans=buy_trans,
            c_tick=c_tick,
            volume=self.volume,
            sl=sl,
        )

        self.get_open_price(c_tick)

    def enough_money(self, c_tick):
        if np.sum(self.hist_profits) + self.capital > (
            self.calculate_position(price=c_tick["ask"][0], vol=self.volume)
        ):
            return True
        else:
            return False

    def crossover(self, prev_candle, c_candle):
        if self.up_crossover(prev_candle, c_candle):
            return True
        elif self.down_crossover(prev_candle, c_candle):
            return False

    def up_crossover(self, prev_candle, c_candle):
        if (
            (prev_candle["short_ma"] < prev_candle["long_ma"])
            and (c_candle["short_ma"] > c_candle["long_ma"])
        ) or (self.in_up_crossover and not self.is_bought):
            return True
        else:
            return False

    def down_crossover(self, prev_candle, c_candle):
        if (
            (prev_candle["short_ma"] > prev_candle["long_ma"])
            and (c_candle["short_ma"] < c_candle["long_ma"])
            or (self.in_down_crossover and not self.is_short)
        ):
            return True
        else:
            return False

    def get_slope(self, prev_candle, c_candle):
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

    def get_angle_between_slopes(self, prev_candle, c_candle):
        m1, m2 = self.get_slope(prev_candle, c_candle)
        return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))

    def get_max_volume(self, price):
        n = -1
        buy_price = 0
        while buy_price < (np.sum(self.hist_profits) + self.capital):
            n += 1
            buy_price = self.calculate_position(
                price=price, vol=self.lotMin + n * self.lotStep
            )
        volume = self.lotMin + ((n - 1) * self.lotStep)
        self.volume = volume

    def calculate_position(self, price, vol):
        position = self.contractSize * vol * price
        margin = position * (self.leverage / 100)
        # convert_currency(margin, self.currency)
        return margin

    def get_open_price(self, c_tick):
        trade = self.apiClient.get_trade(opened=True, order_n=self.order_n)
        if trade is not None:
            open_price = trade["open_price"]
            self.open_price = open_price
            open_position = self.calculate_position(price=open_price, vol=self.volume)
            self.entry_slipage = c_tick["ask"] - open_price
            self.buy_price = open_position
        else:
            self.buy_price = None
            self.entry_slipage = None

    def get_profit(self):
        trade = self.apiClient.get_trade(opened=False, order_n=self.sell_order_n)
        return trade["profit"]

    def evaluate_profit(self, timestamp):
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

    def set_last_tick(self, msg):
        self.last_tick = msg
        self.set_last_ticks(msg)

    def set_last_ticks(self, msg):
        if len(self.last_ticks) > 30:
            self.last_ticks.pop(0)
            self.last_ticks.append(msg)
        else:
            self.last_ticks.append(msg)

    def get_last_ticks_mean(self, key):
        logger.info("Watchdog")
        while 1:
            key_val_list = []
            if self.last_ticks is not None and self.buy_price is not None:
                if len(self.last_ticks) > 9:
                    for tick in self.last_ticks:
                        key_val_list.append(tick[key])
                    mean_key_val = np.sum(key_val_list) / len(key_val_list)
                    if mean_key_val < self.buy_price:
                        logger.info(
                            f"\n***********************\nWATCHDOG\n***********************"
                        )
                        self.sell_position()
            time.sleep(0.1)

    # def get_running_trades(self):
    #     trades = self.apiClient.get_trades()
    #     if trades is not None:


class TradingLiveSession(TradingSession):
    pass


class TradingOflineSession(TradingSession):
    pass
