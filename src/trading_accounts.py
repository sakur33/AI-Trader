import os
import pandas as pd
from trader_utils import *
from sqlalchemy import create_engine
import urllib
from xAPIConnector import *
from creds import creds
import pyodbc
import psycopg2
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import threading

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

minimun_trade = 10

minimun_trade = 10


class Trader:
    def __init__(self, name, trader_name, capital, max_risk, trader_type) -> None:
        self.user = creds[trader_name]["user"]
        self.passw = creds[trader_name]["passw"]
        self.name = name
        self.capital = capital
        self.max_risk = max_risk
        self.db_conn = self.create_db_conn()
        self.db_engine = self.create_db_engine()
        self.ts_conn = self.create_ts_conn()
        self.create_trader()

        self.client = APIClient()
        loginResponse = self.client.execute(
            loginCommand(userId=self.user, password=self.passw)
        )
        status = loginResponse["status"]
        if status:
            logging.info(f"Login successful")

        else:
            error_code = loginResponse["errorCode"]
            logging.info(f"Login error | {error_code}")
            quit()

        self.ssid = loginResponse["streamSessionId"]
        self.client_session_start = datetime.now()
        pingFun = threading.Thread(target=self.ping_client)
        pingFun.setDaemon(True)
        pingFun.start()

        self.streaming_start_time = None
        self.stream_client = None
        self.trader_type = trader_type
        self.sessions = {}
        self.operation_count = 0
        self.order_n = None
        self.session_status = False

    def make_trade(self, symbol_info, current_price, stop_loss=None, take_profit=None):
        # TODO implement make trade:
        pass

    def create_db_conn(self):
        conn = None
        # Trusted Connection to Named Instance
        try:
            conn = pyodbc.connect(
                driver="{ODBC Driver 17 for SQL Server}",
                server="localhost",
                port=1433,
                database="master",
                user="sa",
                password="yourStrong(!)Password",
            )
        except Exception as e:
            logger.info(f"Exception | create_db_conn | {e}")
        return conn

    def create_db_engine(self):
        quoted = urllib.parse.quote_plus(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=master;"
            "UID=sa;"
            "PWD=yourStrong(!)Password"
        )
        engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(quoted))
        return engine

    def create_ts_conn(self):
        dbname = "postgres"
        user = "postgres"
        password = "1234"
        host = "localhost"
        port = 5432
        connection = (
            f"dbname ={dbname} user={user} password={password} host={host} port={port}"
        )
        conn = psycopg2.connect(connection)
        return conn

    def create_ts_engine():
        # TODO implement ts engine
        pass

    def create_trader(self):
        conn = self.db_conn
        sql = f"SELECT tradername FROM traders WHERE tradername = '{self.name}'"
        try:
            cur = conn.cursor()
            cur.execute(sql)
            names = cur.fetchall()
            if len(names) == 0:
                self.insert_trader()
                logger.info(f"New Trader Created | tradername: {self.name}")
            else:
                logger.info(f"Trader already exists | tradername: {self.name}")

        except Exception as e:
            logger.info(f"Exception | create trader | {e}")

    def insert_trader(self):
        conn = self.db_conn
        sql = f"INSERT INTO traders (tradername, creation_date, initial_capital, max_risk) VALUES ('{self.name}','{todayms}',{self.capital},{self.max_risk});"
        logger.info(sql)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            logger.info(f"Trader inserted | traderid: {self.traderid}")
        except Exception as e:
            logger.info(f"Exception | insert_trader | {e}")

    def insert_trade(
        self,
        tradeid,
        date_entry,
        symbol,
        shares,
        entry_price,
        entry_stop_loss,
        entry_take_profit,
        out_price,
    ):
        conn = self.db_conn
        sql = f"INSERT INTO trades (tradeid, traderid, date_entry, symbol, shares, entry_price, entry_stop_loss, entry_take_profit, out_price) VALUES ({tradeid},{self.traderid},{date_entry},{symbol},{shares},{entry_price},{entry_stop_loss},{entry_take_profit},{out_price});"
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            logger.info(f"Exception | insert_trade | {e}")

    def insert_symbols(self, df):

        df["min_price"] = (
            (df["ask"] * df["contractSize"]) / ((1 / df["leverage"]) * 100)
        ) * df["lotStep"]
        try:
            df.to_sql("symbols", schema="dbo", con=self.db_engine, if_exists="replace")
        except Exception as e:
            logger.info(f"Exception | insert symbol | {e}")

    def insert_trans(self, conn, tradeid, trans_date, trans_type, balance):
        sql = f"INSERT INTO balance (traderid, tradeid, trans_date, trans_type, balance) VALUES ({self.traderid},{tradeid},{trans_date},{trans_type},{balance});"
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            logger.info(f"Exception | insert_trans | {e}")

    def insert_params(self, day, symbol, score, short_ma, long_ma, out, min_angle):
        sql = f"INSERT INTO trading_params(symbol_name, date, score, short_ma, long_ma, out, min_angle)VALUES('{symbol}', '{day}', {score}, {short_ma}, {long_ma}, {out}, {min_angle});"
        try:
            cur = self.db_conn.cursor()
            cur.execute(sql)
            self.db_conn.commit()
        except Exception as e:
            logger.info(f"Exception | insert_trading_params | {e}")

    def get_profit_losses(self):
        # TODO implement query to get profits and losses
        pass

    def ping_client(self):
        while True:
            commandResponse = self.client.commandExecute(
                "ping",
                return_df=False,
            )
            logger.info(f"Ping sent | {commandResponse['status']}")
            time.sleep(60 * 5)

    def start_streaming(self):
        self.stream_client = APIStreamClient(
            ssId=self.ssid,
            tickFun=self.tick_processor,
            candleFun=self.candle_processor,
            tradeFun=self.trade_processor,
            profitFun=self.profit_processor,
            tradeStatusFun=self.trade_status_processor,
            keepAliveFun=self.keepAlive,
        )

    def keepAlive(self):
        # logger.debug("Keep Alive")
        pass

    def start_trading_session(self, params):
        df_symbol = pd.read_sql_query(
            f"SELECT * FROM symbols where symbol = '{params['symbol_name']}'",
            con=self.db_engine,
        )
        session = TradingSession(
            symbol=params["symbol_name"],
            short_ma=params["short_ma"],
            long_ma=params["long_ma"],
            out=params["out"],
            min_angle=params["min_angle"],
            capital=self.capital * self.max_risk,
            short_enabled=df_symbol["shortSelling"][0],
            leverage=df_symbol["leverage"][0],
            contractSize=df_symbol["contractSize"][0],
            lotStep=df_symbol["lotStep"][0],
            lotMin=df_symbol["lotMin"][0],
            apiClient=self.client,
            ts_conn=self.ts_conn,
            db_conn=self.db_conn,
            db_engine=self.db_engine,
            trader_name=self.name,
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

    def tick_processor(self, msg):
        tick_df = return_as_df([msg["data"]])
        tick_df["timestamp"] = xtb_time_to_date(
            int(tick_df["timestamp"].values[0]), local_tz=True
        )
        cursor = self.ts_conn.cursor()
        try:
            cursor.execute(
                f"INSERT INTO ticks (timestamp, symbol, ask, bid, high, low, askVolume, bidVolume, tick_level, quoteId, spreadTable, spreadRaw) VALUES ('{tick_df['timestamp'].values[0]}', '{tick_df['symbol'].values[0]}', {tick_df['ask'].values[0]}, {tick_df['bid'].values[0]}, {tick_df['high'].values[0]}, {tick_df['low'].values[0]}, {tick_df['askVolume'].values[0]}, {tick_df['bidVolume'].values[0]}, {tick_df['level'].values[0]}, {tick_df['quoteId'].values[0]}, {tick_df['spreadTable'].values[0]}, {tick_df['spreadRaw'].values[0]});"
            )
        except (Exception, psycopg2.Error) as error:
            logger.info(error.pgerror)
        self.ts_conn.commit()
        if self.sessions[tick_df["symbol"][0]].start_time is None:
            self.sessions[tick_df["symbol"][0]].start_time = tick_df["timestamp"][0]

    def candle_processor(self, msg):
        logger.info("Candle")
        cursor = self.ts_conn.cursor()
        candle_df = return_as_df([msg["data"]])
        candle_df["ctmString"] = pd.to_datetime(
            candle_df["ctmString"], format="%b %d, %Y, %I:%M:%S %p"
        ).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        try:
            cursor.execute(
                f"INSERT INTO candles (symbol, ctm, ctmString, low, high, open, close, vol, quoteId) VALUES ('{candle_df['symbol'].values[0]}', {candle_df['ctm'].values[0]}, '{candle_df['ctmString'].values[0]}', {candle_df['low'].values[0]}, {candle_df['high'].values[0]}, {candle_df['open'].values[0]}, {candle_df['close'].values[0]}, {candle_df['vol'].values[0]}, {candle_df['quoteId'].values[0]});"
            )
        except (Exception, psycopg2.Error) as error:
            logger.info(f"Error inserting candles | {error}")
        self.ts_conn.commit()
        if self.sessions[candle_df["symbol"][0]].start_time is None:
            self.sessions[candle_df["symbol"][0]].start_time = candle_df["ctmString"][0]

        candleFun = threading.Thread(
            target=self.sessions[candle_df["symbol"][0]].step()
        )
        candleFun.start()

    def trade_processor(self, msg):
        logger.info("TRADE: ", msg)
        trades = return_as_df([msg["data"]])

    def balance_processor(self, msg):
        logger.info("BALANCE: ", msg)
        logger.info("\n")

    def trade_status_processor(self, msg):
        logger.info("TRADE STATUS: ", msg)
        logger.info("\n")

    def profit_processor(self, msg):
        logger.info("PROFIT: ", msg)
        logger.info("\n")

    def news_processor(self, msg):
        logger.info("NEWS: ", msg)
        logger.info("\n")

    def look_for_suitable_symbols_v1(self, df):
        # TODO look for suitable symbols
        # Look for symbols with:
        #   - Tight spread
        #   - Trader_type products
        df = df[df["categoryName"] == self.trader_type]
        #   - ask comparable to our max_risk
        # ((ask * contract_size) / leverage) * lotStep
        df = df[df["min_price"] <= (self.capital * self.max_risk)]
        df["spread_percentage"] = (df["ask"] - df["bid"]) / df["ask"]
        df = df.sort_values(by=["spread_percentage"])
        return df

    def update_stocks(self, df, period, days, save=False):
        start_date = datetime.now() - timedelta(days=days)
        for symbol in list(df["symbol"]):
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
                    try:
                        candles["symbol_name"] = symbol
                        candles["period"] = period
                        candles.to_sql(
                            "stocks",
                            con=self.db_engine,
                            if_exists="append",
                            index=False,
                        )
                    except Exception as e:
                        logger.info(f"Exception | insert symbol | {e}")
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
        cur = self.db_conn.cursor()
        cur.execute("SELECT DISTINCT(symbol_name) FROM stocks")
        symbols = cur.fetchall()
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

    def get_trading_params(self):
        params = pd.read_sql(
            "SELECT * FROM trading_params ORDER BY score DESC", self.db_engine
        )
        return params

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
        symbol,
        short_ma,
        long_ma,
        out,
        min_angle,
        capital,
        short_enabled,
        leverage,
        contractSize,
        lotStep,
        lotMin,
        apiClient,
        ts_conn,
        db_conn,
        db_engine,
        trader_name,
    ) -> None:
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.out = out
        self.min_angle = min_angle
        self.capital = capital
        self.short_enabled = short_enabled
        self.leverage = leverage
        self.contractSize = contractSize
        self.lotStep = lotStep
        self.lotMin = lotMin
        self.apiClient = apiClient
        self.ts_conn = ts_conn
        self.db_engine = db_engine
        self.db_conn = db_conn
        self.trader_name = trader_name
        self.is_bought = False
        self.is_short = False
        self.in_up_crossover = False
        self.in_down_crossover = False
        self.profits = []
        self.potential_profits = []
        self.buy_price = None
        self.buy_volume = None
        self.buy_multiplier = (
            self.contractSize / ((1 / self.leverage) * 100)
        ) * self.lotStep
        self.prev_candle_count = 0
        self.start_time = None
        self.step_count = 0
        self.insert_trade_session()

    def step(self):
        logger.debug("Start step")
        logger.info(f"Session step: {self.step_count}")
        self.step_count += 1
        if self.step_count > self.long_ma:

            candles = self.get_candles()
            c_tick = self.get_ticks(int(candles["ctm"].values[0]))
            c_candle = candles.iloc[0, :]
            prev_candle = candles.iloc[-1, :]

            logger.info(
                f"Step count ({self.step_count}) > long_ma ({self.long_ma})\n   Is bought: {self.is_bought}\n   Is short: {self.is_short}\n   Previous: \n  Time: {prev_candle['ctmstring']} \n  Short_ma: {prev_candle['short_ma']}\n       Long_ma: {prev_candle['long_ma']}\n   Current:  \n  Time: {c_candle['ctmstring']} \n   Short_ma: {c_candle['short_ma']}\n  Long_ma: {c_candle['long_ma']}"
            )
            if self.is_bought:

                diff = c_candle["low"] - prev_candle["low"]
                profit = c_candle["low"] - self.buy_price
                prev_profit = prev_candle["low"] - self.buy_price

                logger.info(f"Is Bought")
                logger.info(f"Buy price {self.buy_price}")
                logger.info(f"Profit {profit}")
                logger.info(f"Diff {diff}")
                logger.info(f"Potential profits {self.potential_profits}")

                if profit < -(self.buy_price * self.out):
                    self.sell_position()

                elif profit > 0:
                    self.potential_profits.append(diff)
                    if (prev_profit) > ((profit) * 1 + self.out):
                        self.sell_position()

                else:
                    self.potential_profits.append(diff)

            if self.is_short:
                diff = prev_candle["low"] - c_candle["low"]
                profit = self.buy_price - c_candle["low"]
                prev_profit = self.buy_price - prev_candle["low"]

                logger.info(f"Is Short")
                logger.info(f"Buy price {self.buy_price}")
                logger.info(f"Profit {profit}")
                logger.info(f"Diff {diff}")
                logger.info(f"Potential profits {self.potential_profits}")

                if profit < -(self.buy_price * self.out):
                    self.sell_position()

                elif profit > 0:
                    self.potential_profits.append(diff)
                    if (prev_profit) > ((profit) * 1 + self.out):
                        self.sell_position()
                else:
                    self.potential_profits.append(diff)

            if self.crossover(prev_candle, c_candle):

                ma_short_slope, ma_long_slope = self.get_slope(prev_candle, c_candle)
                crossover_angle = self.get_angle_between_slopes(
                    ma_short_slope, ma_long_slope
                )
                if self.up_crossover(prev_candle, c_candle):
                    logger.info(f"Up Crossover")
                    if self.is_bought:
                        self.sell_position()

                    if np.abs(crossover_angle) > self.min_angle:
                        if self.enough_money(c_tick):
                            self.buy_position()
                        else:
                            logger.info(f"Not enough money")
                            pass
                            # TODO: NOT ENOUGH MONEY (END SESSION?)
                    else:
                        self.in_up_crossover = True
                        logger.info(f"Crossover angle too small")
                        logger.info(f"  Angle: {crossover_angle}")
                        logger.info(f"  Min angle: {self.min_angle}")
                        # TODO: CROSSOVER ANGLE TOO SMALL (?)

                if self.down_crossover(prev_candle, c_candle):
                    logger.info(f"Down Crossover")
                    if self.is_bought:
                        self.sell_position()

                    if self.short_enabled:
                        if np.abs(crossover_angle) > self.min_angle:
                            if self.enough_money(c_tick):
                                self.buy_position(c_tick)
                            else:
                                logger.info(f"Not enough money")
                                pass
                                # TODO: NOT ENOUGH MONEY (END SESSION?)
                        else:
                            self.in_down_crossover = True
                            logger.info(f"Crossover angle too small")
                            logger.info(f"  Angle: {crossover_angle}")
                            logger.info(f"  Min angle: {self.min_angle}")
                            # TODO: CROSSOVER ANGLE TOO SMALL (?)
                            pass

        logger.debug("End step")

    def get_trade(self, opened=False):
        commandResponse = self.apiClient.commandExecute(
            "tradeTransactionStatus",
            arguments={"order": self.order_n},
            return_df=False,
        )
        trans_status = commandResponse["returnData"]
        commandResponse = self.apiClient.commandExecute(
            "getTrades",
            arguments={"openedOnly": opened},
            return_df=False,
        )
        trade_records = commandResponse["returnData"]
        for trade_record in trade_records:
            if trade_record["order2"] == trans_status["order"]:
                trade = trade_record

        return trade

    def sell_position(self, profit, c_tick):
        logger.info("Enters self.sell position")
        self.is_bought = False
        self.is_short = False
        self.profits.append(profit)
        self.potential_profits = [0]
        self.buy_volume = None
        self.buy_price = None
        position = self.get_trade(opened=True)
        commandResponse = self.apiClient.commandExecute(
            "tradeTransaction",
            arguments={
                "tradeTransInfo": {
                    "cmd": TransactionSide.SELL,
                    "customComment": self.trader_name,
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
        logger.info(commandResponse)
        self.update_trade()

    def buy_position(self, c_tick):
        logger.info("Enters self.buy position")
        self.is_bought = True
        self.buy_price = c_tick["ask"][0]
        commandResponse = self.apiClient.commandExecute(
            "tradeTransaction",
            arguments={
                "tradeTransInfo": {
                    "cmd": TransactionSide.BUY,
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
        logger.info(commandResponse)
        self.order_n = commandResponse["returnData"]["order"]

    def enough_money(self, c_tick):
        if np.sum(self.profits) + self.capital > (
            self.calculate_position(
                price=c_tick["ask"][0], vol=self.get_max_volume(price=c_tick["ask"][0])
            )
        ):
            return True
        else:
            return False

    def crossover(self, prev_canlde, c_candle):
        if (
            (
                (prev_canlde["short_ma"] < prev_canlde["long_ma"])
                and (c_candle["short_ma"] > c_candle["long_ma"])
                or (prev_canlde["short_ma"] > prev_canlde["long_ma"])
                and (c_candle["short_ma"] < c_candle["long_ma"])
            )
            or (self.in_up_crossover and not self.is_bought)
            or (self.in_down_crossover and not self.is_short)
        ):
            return True
        else:
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
        p1 = [prev_candle["ctm"], prev_candle["short_ma"]]
        p2 = [c_candle["ctm"], c_candle["short_ma"]]
        x1, y1 = p1
        x2, y2 = p2
        ma_short_slope = (y2 - y1) / (x2 - x1)

        p1 = [prev_candle["ctm"], prev_candle["long_ma"]]
        p2 = [c_candle["ctm"], c_candle["long_ma"]]
        x1, y1 = p1
        x2, y2 = p2
        ma_long_slope = (y2 - y1) / (x2 - x1)
        return ma_short_slope, ma_long_slope

    def get_angle_between_slopes(self, m1, m2):
        return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))

    def get_max_volume(self, price):
        n = 0
        buy_price = 0
        while buy_price < self.capital:
            buy_price = (
                (self.contractSize / ((1 / self.leverage) * 100))
                * (self.lotMin + n * self.lotStep)
                * price
            )
            n += 1
        if n > 1:
            volume = self.lotMin + (n - 2) * self.lotStep
        else:
            volume = 0
        self.volume = volume
        return volume

    def calculate_position(self, price, vol):
        position = (self.contractSize / ((1 / self.leverage) * 100)) * (vol) * price
        return position

    def get_trade(self):
        commandResponse = self.apiClient.commandExecute(
            "getTrades",
            arguments={"openedOnly": False},
            return_df=False,
        )
        trade_records = commandResponse["returnData"]
        for record in trade_records:
            if record["order2"] == self.order_n:
                return record

    def get_candle_count(self):
        pass
        sql = f"select count(*) from candles WHERE symbol = '{self.symbol}' AND ctmstring > '{self.start_time}';"
        cursor = self.ts_conn.cursor()
        cursor.execute(sql)
        count = cursor.fetchall()[0][0]
        return count

    def get_candles(self):
        sql = f"SELECT ctm, ctmstring, symbol, low, high, open, close, vol, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {self.short_ma} PRECEDING AND CURRENT ROW)AS short_ma, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {self.long_ma} PRECEDING AND CURRENT ROW)AS long_ma  FROM candles WHERE symbol = '{self.symbol}' ORDER BY ctmstring DESC Limit 2;"
        cursor = self.ts_conn.cursor()
        cursor.execute(sql)
        candles = cursor.fetchall()
        candles_df = pd.DataFrame(
            columns=[
                "ctm",
                "ctmstring",
                "symbol",
                "low",
                "high",
                "open",
                "close",
                "vol",
                "short_ma",
                "long_ma",
            ],
            data=candles,
        )
        return candles_df

    def get_ticks(self, timestamp):
        tick_df = self.apiClient.commandExecute(
            commandName="getTickPrices",
            arguments={"level": 0, "symbols": [self.symbol], "timestamp": timestamp},
            return_df=False,
        )
        tick_df = return_as_df(tick_df["returnData"]["quotations"])
        # sql = f"SELECT * FROM ticks WHERE symbol = '{self.symbol}' ORDER BY timestamp limit 1;"
        # cursor = self.ts_conn.cursor()
        # cursor.execute(sql)
        # ticks = cursor.fetchall()
        # ticks_df = pd.DataFrame(
        #     columns=[
        #         "timestamp",
        #         "symbol",
        #         "ask",
        #         "bid",
        #         "high",
        #         "low",
        #         "askvolume",
        #         "bidvolume",
        #         "tick_level",
        #         "quoteid",
        #         "spreadtable",
        #         "spreadraw",
        #     ],
        #     data=ticks,
        # )
        return tick_df

    def insert_trade(self):
        trade = self.get_trade()
        entry_position = self.calculate_position(trade["open_price"], trade["volume"])
        out_position = self.calculate_position(trade["close_price"], trade["volume"])
        sql = f"INSERT INTO trades (tradername, symbol, time_entry, time_close, vol, entry_price, out_price, entry_position, out_position)VALUES('{self.trader_name}', '{self.symbol}', '{trade['open_time']}', '{trade['close_time']}', {trade['volume']}, {trade['open_price']}, {trade['close_price']}, {entry_position}, {out_position});"
        cursor = self.db_conn.cursor()
        cursor.execute(sql)
        self.db_conn.commit()

    def insert_trade_session(self):
        if self.is_bought:
            state = "BUY"
        elif self.is_short:
            state = "SHORT"
        else:
            state = "OUT"
        sql = f"INSERT INTO trade_session (ctmstring, symbol, state)VALUES('{datetime.now()}', '{self.symbol}', '{state}');"
        cursor = self.ts_conn.cursor()
        cursor.execute(sql)
        self.ts_conn.commit()
