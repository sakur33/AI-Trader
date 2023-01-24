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

    def start_trading_sessions(self, params_df):
        sessions = {}
        if isinstance(params_df, pd.DataFrame):
            for params in params_df.iterrows():
                sessions[params["symbol"]] = self.start_trading_session(params)
            self.sessions = sessions
        else:
            self.sessions[params_df["symbol_name"]] = self.start_trading_session(
                params_df
            )

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
            # get candles
            CHART_RANGE_INFO_RECORD = {
                "period": period,
                "start": date_to_xtb_time(start_date),
                "symbol": symbol,
            }
            commandResponse = self.client.commandExecute(
                "getChartLastRequest",
                arguments={"info": CHART_RANGE_INFO_RECORD},
                return_df=False,
            )
            if commandResponse["status"] == False:
                error_code = commandResponse["errorCode"]
                logger.info(f"Login failed. Error code: {error_code}")
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
                    logger.info(f"Symbol {symbol} did not return candles")

    def evaluate_stocks(
        self,
        params=None,
        verbose=0,
        show=False,
    ):
        if params is None:
            logger.info("No params to evaluate stocks")
            quit()
        symbols = get_distict_symbols()

        parameters = {
            "short_ma": list(
                range(
                    params["short_ma"][0], params["short_ma"][1], params["short_ma"][2]
                )
            ),
            "long_ma": list(
                range(params["long_ma"][0], params["long_ma"][1], params["long_ma"][2])
            ),
            "min_angle": list(
                range(
                    params["min_angle"][0],
                    params["min_angle"][1],
                    params["min_angle"][2],
                )
            ),
            "out": list(
                np.linspace(
                    params["out"][0],
                    params["out"][1],
                    params["out"][2],
                )
            ),
        }

        train_period = params["train_period"]
        test_period = params["test_period"]
        profits = []
        for symbol in symbols:
            candles = pd.read_sql(
                f"SELECT ctm, ctmString, low, high, [open], [close], vol FROM stocks where symbol_name = '{symbol[0]}'",
                self.db_conn,
            )
            candles = cast_candles_to_types(candles, digits=None, dates=False)
            cur.execute(
                f"SELECT TOP(1) shortSelling, leverage, contractSize, lotStep FROM symbols WHERE symbol = '{symbol[0]}' ORDER BY time"
            )
            short_enabled, leverage, contractSize, lotStep = cur.fetchall()[0]
            trading_estimator = TradingEstimator(
                period=train_period,
                capital=self.capital * self.max_risk,
                symbol=symbol[0],
                short_enabled=short_enabled,
                leverage=leverage,
                contractSize=contractSize,
                lotStep=lotStep,
            )
            dates = candles[["ctmString", "ctm"]]
            dates = dates.set_index(dates["ctmString"])
            n_days = dates.resample("D").first().shape[0]
            if n_days >= (train_period + test_period) * 7:
                logger.info("Tesing parameters")
                estimator = trading_estimator.fit(candles)
                clf = RandomizedSearchCV(
                    estimator,
                    parameters,
                    n_iter=params["repetitions"],
                    verbose=verbose,
                    cv=2,
                )
                clf.fit(candles)
                score = trading_estimator.test(
                    candles, clf.best_params_, test_period, show=show
                )
                logger.info(" Results from Grid Search ")
                logger.info(
                    f"    The best estimator across ALL searched params: {clf.best_estimator_}"
                )
                logger.info(
                    f"    The best score across ALL searched params: {clf.best_score_}"
                )
                logger.info(
                    f"    The best parameters across ALL searched params: {clf.best_params_}"
                )
                gain = (self.capital * self.max_risk) + (
                    (score * contractSize / ((1 / leverage) * 100)) * lotStep
                )
                logger.info(f"    Score in the last day: {score}")
                logger.info(f"    Balance of the last day: {gain}")

                if score > 0:
                    self.insert_params(
                        day=todayms,
                        symbol=symbol[0],
                        score=score,
                        short_ma=clf.best_params_["short_ma"],
                        long_ma=clf.best_params_["long_ma"],
                        out=clf.best_params_["out"],
                        min_angle=clf.best_params_["min_angle"],
                    )

            else:
                logger.info(f"Not enough values in candles {candles.shape[0]}")
        logger.info(f"Profits: {np.nansum(profits)}")

    def trade_logic(self, symbol):
        trade_params = pd.read_sql(
            f"select * from trading_params where symbol_name = '{symbol[0]}');",
            self.db_conn,
        )
        candles = pd.read_sql(
            f"SELECT ctm,ctmstring,low,high,open,close,vol,AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {trade_params['short_ma']} PRECEDING AND CURRENT ROW) AS short_ma, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {trade_params['long_ma']} PRECEDING AND CURRENT ROW) AS long_ma FROM candles where symbol = '{symbol[0]}' ORDER BY ctmstring DESC;",
            self.ts_conn,
        )

    def buy_position(self, c_tick):
        commandResponse = self.client.commandExecute(
            "tradeTransaction",
            arguments={
                "tradeTransInfo": {
                    "cmd": TransactionSide.BUY,
                    "customComment": "No comment",
                    "expiration": date_to_xtb_time(
                        datetime.now() + timedelta(seconds=10)
                    ),
                    "offset": 0,
                    "order": 0,
                    "price": c_tick["ask"][0],
                    "sl": 0.0,
                    "symbol": c_tick["symbol"][0],
                    "tp": 0.0,
                    "type": TransactionType.ORDER_OPEN,
                    "volume": 0.1,
                }
            },
            return_df=False,
        )
        logger.info(commandResponse)
        return commandResponse["returnData"]["order"]

    def sell_position(self, c_tick, order_n):
        commandResponse = self.client.commandExecute(
            "tradeTransaction",
            arguments={
                "tradeTransInfo": {
                    "cmd": TransactionSide.SELL,
                    "customComment": "No comment",
                    "expiration": 0,
                    "offset": 0,
                    "order": order_n,
                    "price": c_tick["bid"][0],
                    "sl": 0.0,
                    "symbol": c_tick["symbol"][0],
                    "tp": 0.0,
                    "type": TransactionType.ORDER_CLOSE,
                    "volume": 0.1,
                }
            },
            return_df=False,
        )
        logger.info(commandResponse)


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
