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

minimun_trade = 10
minimun_trade = 10


class Trader:
    def __init__(self, name, capital, max_risk, trader_type) -> None:
        self.user = creds[name]["user"]
        self.passw = creds[name]["passw"]
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

    def start_trading_session(self, params):
        session = TradingSession(
            name=self.name,
            capital=self.capital * self.max_risk,
            symbol=params["symbol_name"],
            short_ma=params["short_ma"],
            long_ma=params["long_ma"],
            profit_exit=params["profit_exit"],
            loss_exit=params["loss_exit"],
            min_angle=params["min_angle"],
        )
        return session

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
            self.apiSession.store_past_candles(symbol, start_date, period)

    def evaluate_stocks(self, date):
        symbols = get_distict_symbols()
        for symbol in tqdm(symbols):
            symbol = symbol[0]
            symbol_info = get_symbol_info(symbol)
            symbol_stats = get_symbol_stats(symbol, date)
            if symbol_info["spreadRaw"] * 1.25 < symbol_stats["std_close"]:
                logger.info(
                    f"GOOD\nSymbol {symbol}: {symbol_stats['std_close']} / symbol_info['spreadRaw']"
                )
            elif symbol_info["spreadRaw"] < symbol_stats["std_close"]:
                logger.info(
                    f"GREATER\nSymbol {symbol}: {symbol_stats['std_close']} / symbol_info['spreadRaw']"
                )
            else:
                logger.info(
                    f"LESS\nSymbol {symbol}: {symbol_stats['std_close']} / symbol_info['spreadRaw']"
                )


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
    ) -> None:

        # Session params
        self.name = name
        self.capital = capital
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.profit_exit = profit_exit
        self.loss_exit = loss_exit
        self.min_angle = min_angle
        self.offline = offline

        # Symbol information
        symbol_info = get_symbol_info(self.symbol)
        self.short_enabled = symbol_info["shortSelling"]
        self.leverage = symbol_info["leverage"]
        self.contractSize = symbol_info["contractSize"]
        self.lotStep = symbol_info["lotStep"]
        self.lotMin = symbol_info["lotMin"]

        self.is_bought = False
        self.is_short = False
        self.profit = None
        self.prev_profit = None
        self.profits = []
        self.potential_profits = []
        self.buy_price = None
        self.open_price = None
        self.current_position = None
        self.current_price = None
        self.buy_volume = None
        self.start_time = None
        self.insert_trade_session()

    def step(self):
        self.step_count += 1
        if self.step_count > self.long_ma:

            candles = get_last_two_candles(self.symbol, self.short_ma, self.long_ma)
            timestamp = int(candles["ctm"].values[0])
            c_tick = self.get_ticks(timestamp)
            c_candle = candles.iloc[0, :]
            prev_candle = candles.iloc[-1, :]
            angle_diff = self.get_angle_between_slopes(prev_candle, c_candle)

            if c_candle["short_ma"] < c_candle["long_ma"]:
                self.in_down_crossover = True
            elif c_candle["short_ma"] > c_candle["long_ma"]:
                self.in_up_crossover = True

            if self.is_bought:
                diff = c_candle["low"] - prev_candle["low"]
                profit, prev_profit = self.evaluate_profit(timestamp)

                logger.info(
                    f"Is Bought\nBuy price {self.buy_price}\nProfit {profit}\nDiff {diff}\nPotential profits {self.potential_profits}"
                )

                if profit < -(self.buy_price * self.exit):
                    self.sell_position(profit=profit, c_tick=c_tick)

                elif profit > 0:
                    self.potential_profits.append(diff)
                    if (prev_profit) > ((profit) * (1 + self.out)):
                        self.sell_position(profit=profit, c_tick=c_tick)

                else:
                    self.potential_profits.append(diff)

            if self.is_short:
                diff = prev_candle["low"] - c_candle["low"]
                profit, prev_profit = self.evaluate_profit(timestamp)

                logger.info(
                    f"Is Short\nBuy price {self.buy_price}\nProfit {profit}\nDiff {diff}\nPotential profits {self.potential_profits}"
                )

                if profit < -(self.buy_price * self.out):
                    self.sell_position(profit=profit, c_tick=c_tick)

                elif profit > 0:
                    self.potential_profits.append(diff)
                    if (prev_profit) > ((profit) * (1 + self.out)):
                        self.sell_position(profit=profit, c_tick=c_tick)
                else:
                    self.potential_profits.append(diff)

            if self.crossover(prev_candle, c_candle):
                self.in_down_crossover = False
                logger.info(f"Up Crossover")
                if angle_diff > self.min_angle:
                    if self.enough_money(c_tick):
                        if not self.is_bought and not self.is_short:
                            self.buy_position(c_tick=c_tick)
                        else:
                            logger.info("Already positioned")
                            # TODO: implement multitrade
                    else:
                        logger.info(f"Not enough money")
                        pass
                        # TODO: NOT ENOUGH MONEY (END SESSION?)
            else:
                logger.info(f"Down Crossover")
                self.in_up_crossover = False
                if self.short_enabled:
                    if angle_diff < -self.min_angle:

                        if self.enough_money(c_tick):
                            if not self.is_bought and not self.is_short:
                                self.buy_position(c_tick=c_tick, short=True)
                            else:
                                logger.info("Already positioned")
                        else:
                            logger.info(f"Not enough money")
                            pass
                            # TODO: NOT ENOUGH MONEY (END SESSION?)

            logger.info(
                f"\n**************************\nStep count ({self.step_count}) > long_ma ({self.long_ma})\n Is bought: {self.is_bought}\n Is short: {self.is_short}\n Up cross: {self.in_up_crossover}\n Down cross: {self.in_down_crossover}\n Previous:\n     Time: {prev_candle['ctmstring']}\n      Short_ma: {prev_candle['short_ma']}\n       Long_ma: {prev_candle['long_ma']}\n Current:\n      Time: {c_candle['ctmstring']}\n            Short_ma: {c_candle['short_ma']}\n            Long_ma: {c_candle['long_ma']}\n Angle: {angle_diff}\n Min_angle: {self.min_angle}\n Entry Price: {self.open_price}\n Entry Position: {self.buy_price}\n Current Price: {self.current_price}\n Current Position: {self.current_position}\n Profit: {self.profit}\n Prev. Profit: {self.prev_profit}\n**************************"
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

    def sell_position(self, profit, c_tick, short=False):
        logger.info("\n******\nSELL\n******")
        if short:
            buy_trans = TransactionSide.BUY
        else:
            buy_trans = TransactionSide.SELL
        self.is_bought = False
        self.is_short = False
        self.profits.append(profit)
        self.potential_profits = [0]
        self.buy_volume = None
        self.buy_price = None
        self.in_down_crossover = False
        self.in_up_crossover = False
        position = self.get_trade(opened=True)
        try:
            commandResponse = self.apiClient.commandExecute(
                "tradeTransaction",
                arguments={
                    "tradeTransInfo": {
                        "cmd": buy_trans,
                        "customComment": self.name,
                        "expiration": 0,
                        "offset": 0,
                        "order": position["order"],
                        "price": c_tick["ask"][0],
                        "sl": 0.0,
                        "symbol": c_tick["symbol"][0],
                        "tp": 0.0,
                        "type": TransactionType.ORDER_CLOSE,
                        "volume": self.volume,
                    }
                },
                return_df=False,
            )
            self.sell_order_n = commandResponse["returnData"]["order"]
        except Exception as e:
            logger.info(f"Exception at sell position: {e}")
            self.apiClientLogin()
            commandResponse = self.apiClient.commandExecute(
                "tradeTransaction",
                arguments={
                    "tradeTransInfo": {
                        "cmd": buy_trans,
                        "customComment": self.name,
                        "expiration": 0,
                        "offset": 0,
                        "order": position["order"],
                        "price": c_tick["bid"][0],
                        "sl": 0.0,
                        "symbol": c_tick["symbol"][0],
                        "tp": 0.0,
                        "type": TransactionType.ORDER_CLOSE,
                        "volume": self.volume,
                    }
                },
                return_df=False,
            )
            self.sell_order_n = commandResponse["returnData"]["order"]
        self.order_n = None
        self.sell_price = c_tick["ask"][0]
        t = Thread(target=self.insert_trade_info)
        t.start()

    def buy_position(self, c_tick, short=False):
        if short:
            self.is_short = short
            self.trade_type = "SHORT"
            buy_trans = TransactionSide.SELL
            logger.info("\n******\nSHORT\n******")
        else:
            self.is_bought = True
            self.trade_type = "BUY"
            buy_trans = TransactionSide.BUY
            logger.info("\n******\nBUY\n******")

        self.in_down_crossover = False
        self.in_up_crossover = False
        try:
            commandResponse = self.apiClient.commandExecute(
                "tradeTransaction",
                arguments={
                    "tradeTransInfo": {
                        "cmd": buy_trans,
                        "customComment": "No comment",
                        "expiration": 0,
                        "offset": 0,
                        "order": 0,
                        "price": c_tick["bid"][0],
                        "sl": 0.0,
                        "symbol": c_tick["symbol"][0],
                        "tp": 0.0,
                        "type": TransactionType.ORDER_OPEN,
                        "volume": self.get_max_volume(c_tick["ask"][0]),
                    }
                },
                return_df=False,
            )
            self.order_n = commandResponse["returnData"]["order"]
        except Exception as e:
            logger.info(f"Exception at buy position: {e}")
            self.apiClientLogin()
            commandResponse = self.apiClient.commandExecute(
                "tradeTransaction",
                arguments={
                    "tradeTransInfo": {
                        "cmd": buy_trans,
                        "customComment": "No comment",
                        "expiration": date_to_xtb_time(
                            datetime.now() + timedelta(seconds=5)
                        ),
                        "offset": 0,
                        "order": 0,
                        "price": c_tick["ask"][0],
                        "sl": 0.0,
                        "symbol": c_tick["symbol"][0],
                        "tp": 0.0,
                        "type": TransactionType.ORDER_OPEN,
                        "volume": self.get_max_volume(c_tick["ask"][0]),
                    }
                },
                return_df=False,
            )
            self.order_n = commandResponse["returnData"]["order"]
        self.buy_price, open_price = self.get_open_price()
        self.entry_slipage = c_tick["ask"][0] - open_price

    def enough_money(self, c_tick):
        if np.sum(self.profits) + self.capital > (
            self.calculate_position(
                price=c_tick["ask"][0], vol=self.get_max_volume(price=c_tick["ask"][0])
            )
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

    def get_max_values(self):
        try:
            commandResponse = self.apiClient.commandExecute(
                commandName="getSymbol",
                arguments={"symbol": self.symbol},
                return_df=False,
            )
            min_close = commandResponse["returnData"]["low"]
            max_close = commandResponse["returnData"]["high"]
        except Exception as e:
            logger.info(f"Exception at get max values: {e}")
            self.apiClientLogin()
            commandResponse = self.apiClient.commandExecute(
                commandName="getSymbol",
                arguments={"symbol": self.symbol},
                return_df=False,
            )
            min_close = commandResponse["returnData"]["low"]
            max_close = commandResponse["returnData"]["high"]
        min_ctm = 0
        max_ctm = self.long_ma
        return min_ctm, max_ctm, min_close, max_close

    def get_slope(self, prev_candle, c_candle):
        min_ctm, max_ctm, min_close, max_close = self.get_max_values()
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
        while buy_price < self.capital:
            n += 1
            buy_price = self.calculate_position(
                price=price, vol=self.lotMin + n * self.lotStep
            )
        volume = self.lotMin + ((n - 1) * self.lotStep)
        self.volume = volume
        return volume

    def calculate_position(self, price, vol):
        position = self.contractSize * vol * price
        margin = position * (self.leverage / 100)
        return convert_currency(margin, self.currency)

    def get_ticks(self, timestamp):
        try:
            tick_df = self.apiClient.commandExecute(
                commandName="getTickPrices",
                arguments={
                    "level": 0,
                    "symbols": [self.symbol],
                    "timestamp": timestamp,
                },
                return_df=False,
            )
            tick_df = return_as_df(tick_df["returnData"]["quotations"])
        except Exception as e:
            logger.info(f"Exception at get ticks: {e}")
            self.apiClientLogin()
            tick_df = self.apiClient.commandExecute(
                commandName="getTickPrices",
                arguments={
                    "level": 0,
                    "symbols": [self.symbol],
                    "timestamp": timestamp,
                },
                return_df=False,
            )
            tick_df = return_as_df(tick_df["returnData"]["quotations"])
        return tick_df

    def get_trade(self, opened=True):
        while 1:
            if opened:
                try:
                    commandResponse = self.apiClient.commandExecute(
                        "getTrades",
                        arguments={"openedOnly": True},
                        return_df=False,
                    )
                    trade_records = commandResponse["returnData"]
                except Exception as e:
                    logger.info(f"Exception at get trade: {e}")
                    self.apiClientLogin()
                    commandResponse = self.apiClient.commandExecute(
                        "getTrades",
                        arguments={"openedOnly": True},
                        return_df=False,
                    )
                    trade_records = commandResponse["returnData"]

                if isinstance(trade_records, list):
                    for record in trade_records:
                        if record["order2"] == self.order_n:
                            return record
                else:
                    if record["order2"] == self.order_n:
                        return record
            else:
                try:
                    commandResponse = self.apiClient.commandExecute(
                        "getTradesHistory",
                        arguments={"end": 0, "start": 0},
                        return_df=False,
                    )
                    trade_records = commandResponse["returnData"]
                except Exception as e:
                    logger.info(f"Exception at get trade: {e}")
                    self.apiClientLogin()
                    commandResponse = self.apiClient.commandExecute(
                        "getTradesHistory",
                        arguments={"end": 0, "start": 0},
                        return_df=False,
                    )
                    trade_records = commandResponse["returnData"]
                if isinstance(trade_records, list):
                    for record in trade_records:
                        if record["order2"] == self.sell_order_n:
                            return record
                else:
                    if record["order2"] == self.sell_order_n:
                        return record

            time.sleep(0.1)

    def get_open_price(self):
        trade = self.get_trade(opened=True)
        open_price = trade["open_price"]
        self.open_price = open_price
        open_position = self.calculate_position(price=open_price, vol=self.volume)
        return open_position, open_price

    def evaluate_profit(self, timestamp):
        tick = self.get_ticks(timestamp=timestamp)
        if self.profit is not None:
            prev_profit = self.profit
        else:
            prev_profit = 0
        self.current_position = self.calculate_position(
            price=tick["ask"][0], vol=self.volume
        )
        self.current_price = tick["ask"][0]
        profit = (
            (self.current_price - self.open_price) * self.contractSize * self.volume
        )

        if self.is_short:
            self.profit = -1 * profit
        else:
            self.profit = profit
        self.prev_profit = prev_profit
        return self.profit, prev_profit

    def insert_trade_info(self):
        trade = self.get_trade(opened=False)
        entry_position = self.calculate_position(trade["open_price"], trade["volume"])
        out_position = self.calculate_position(trade["close_price"], trade["volume"])
        out_slipage = (
            (self.sell_price - trade["close_price"]) * self.contractSize * self.volume
        )
        self.entry_slipage = self.entry_slipage * self.contractSize * self.volume
        open_time = xtb_time_to_date(trade["open_time"])
        close_time = xtb_time_to_date(trade["close_time"])
        insert_trade(
            self.name,
            self.symbol,
            trade["cmd"],
            open_time,
            close_time,
            trade["volume"],
            trade["open_price"],
            entry_position,
            trade["close_price"],
            out_position,
            self.entry_slipage,
            out_slipage,
            trade["profit"],
        )
