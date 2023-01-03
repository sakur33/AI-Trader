import os
import pandas as pd
from trader_utils import *
import sqlite3
from xAPIConnector import *
from creds import creds
import psycopg2
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

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
    def __init__(self, name, capital, max_risk, trader_type) -> None:
        self.traderid = today_int
        self.user = creds["trader1"]["user"]
        self.passw = creds["trader1"]["passw"]
        self.name = name
        self.capital = capital
        self.max_risk = max_risk
        self.db_conn = self.create_db_conn("ai_trader.db")
        self.ts_conn = self.create_ts_conn()
        self.insert_trader()
        self.client = APIClient()
        self.ssid = None
        self.stream_client = None
        self.trader_type = trader_type

    def make_trade(self, symbol_info, current_price, stop_loss=None, take_profit=None):
        # TODO implement make trade:
        pass

    def create_db_conn(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(f"{database_path}{db_file}")
        except Exception as e:
            print(f"Exception | create_db_conn | {e}")
        return conn

    def create_ts_conn(self):
        dbname = "postgres"
        user = "postgres"
        password = "1234"
        host = "localhost"
        port = 5432
        connection = (
            f"dbname ={dbname} user={user} password={password} host={host} port={port}"
        )
        return psycopg2.connect(connection)

    def insert_trader(self):
        conn = self.db_conn
        sql = f"INSERT INTO traders (traderid, tradername, creation_date, initial_capital, max_risk) VALUES ({int(self.traderid)},'{self.name}','{todayms}',{self.capital},{self.max_risk});"
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            print(f"Trader inserted | traderid: {self.traderid}")
        except Exception as e:
            print(f"Exception | insert_trader | {e}")

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
            print(f"Exception | insert_trade | {e}")

    def insert_symbols(self, df):
        try:
            df["min_price"] = (
                (df["ask"] * df["contractSize"]) / ((1 / df["leverage"]) * 100)
            ) * df["lotStep"]
            df.to_sql("symbols", self.db_conn, if_exists="replace", index=False)
        except Exception as e:
            print(f"Exception | insert symbol | {e}")

    def insert_trans(self, conn, tradeid, trans_date, trans_type, balance):
        sql = f"INSERT INTO balance (traderid, tradeid, trans_date, trans_type, balance) VALUES ({self.traderid},{tradeid},{trans_date},{trans_type},{balance});"
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            print(f"Exception | insert_trans | {e}")

    def insert_params(self, day, symbol, score, short_ma, long_ma):
        sql = f"INSERT INTO trading_params(symbol_name, date, score, short_ma, long_ma)VALUES('{symbol}', '{day}', {score}, {short_ma}, {long_ma});"
        try:
            cur = self.db_conn.cursor()
            cur.execute(sql)
            self.db_conn.commit()
        except Exception as e:
            print(f"Exception | insert_trade | {e}")

    def insert_params(self, day, symbol, score, short_ma, long_ma):
        sql = f"INSERT INTO trading_params(symbol_name, date, score, short_ma, long_ma)VALUES('{symbol}', '{day}', {score}, {short_ma}, {long_ma});"
        try:
            cur = self.db_conn.cursor()
            cur.execute(sql)
            self.db_conn.commit()
        except Exception as e:
            print(f"Exception | insert_trade | {e}")

    def get_profit_losses(self):
        # TODO implement query to get profits and losses
        pass

    def start_streaming(self):
        self.stream_client = APIStreamClient(
            ssId=self.ssid,
            tickFun=self.tick_processor,
            tradeFun=self.trade_processor,
            profitFun=self.profit_processor,
            tradeStatusFun=self.trade_status_processor,
        )

    def tick_processor(self, msg):
        tick_df = return_as_df([msg["data"]])
        tick_df["timestamp"] = xtb_time_to_date(int(tick_df["timestamp"].values[0]))
        cursor = self.ts_conn.cursor()
        try:
            cursor.execute(
                f"INSERT INTO ticks (timestamp, symbol, ask, bid, high, low, askVolume, bidVolume, tick_level, quoteId, spreadTable, spreadRaw) VALUES ('{tick_df['timestamp'].values[0]}', '{tick_df['symbol'].values[0]}', {tick_df['ask'].values[0]}, {tick_df['bid'].values[0]}, {tick_df['high'].values[0]}, {tick_df['low'].values[0]}, {tick_df['askVolume'].values[0]}, {tick_df['bidVolume'].values[0]}, {tick_df['level'].values[0]}, {tick_df['quoteId'].values[0]}, {tick_df['spreadTable'].values[0]}, {tick_df['spreadRaw'].values[0]});"
            )
        except (Exception, psycopg2.Error) as error:
            print(error.pgerror)
        self.ts_conn.commit()

    def trade_processor(msg):
        print("TRADE: ", msg)
        print("\n")

    def balance_processor(msg):
        print("BALANCE: ", msg)
        print("\n")

    def trade_status_processor(msg):
        print("TRADE STATUS: ", msg)
        print("\n")

    def profit_processor(msg):
        print("PROFIT: ", msg)
        print("\n")

    def news_processor(msg):
        print("NEWS: ", msg)
        print("\n")

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
                "getChartLastRequest", arguments={"info": CHART_RANGE_INFO_RECORD}
            )
            if commandResponse["status"] == False:
                error_code = commandResponse["errorCode"]
                print(f"Login failed. Error code: {error_code}")
            else:
                returnData = commandResponse["returnData"]
                digits = returnData["digits"]
                candles = return_as_df(returnData["rateInfos"])
                if not candles is None:
                    candles = cast_candles_to_types(candles, digits)
                    candles = adapt_data(candles)
                    try:

                        candles["symbol_name"] = symbol
                        candles["period"] = period
                        candles.to_sql(
                            "stocks", self.db_conn, if_exists="append", index=False
                        )
                    except Exception as e:
                        print(f"Exception | insert symbol | {e}")
                else:
                    print(f"Symbol {symbol} did not return candles")
            # get ticks
            # commandResponse = self.client.commandExecute(
            #     "getTickPrices",
            #     arguments={
            #         "level": -1,
            #         "symbols": [symbol],
            #         "timestamp": date_to_xtb_time(start_date),
            #     },
            # )
            # if commandResponse["status"] == False:
            #     error_code = commandResponse["errorCode"]
            #     print(f"Login failed. Error code: {error_code}")
            # else:
            #     returnData = commandResponse["returnData"]
            #     ticks = return_as_df(returnData["quotations"])
            #     if not candles is None:
            #         ticks = cast_ticks_to_types(ticks)
            #         ticks["tick_level"] = ticks["level"]
            #         for column in list(ticks.columns):
            #             if column in [
            #                 "symbol",
            #                 "timestamp",
            #                 "ask",
            #                 "askVolume",
            #                 "bid",
            #                 "bidVolume",
            #                 "high",
            #                 "low",
            #                 "tick_level",
            #                 "spreadTable",
            #                 "spreadRaw",
            #             ]:
            #                 pass
            #             else:
            #                 ticks = ticks.drop(columns=column)
            #         try:
            #             ticks.to_sql(
            #                 "ticks", self.db_conn, if_exists="append", index=False
            #             )
            #         except Exception as e:
            #             print(f"Exception | insert symbol | {e}")
            #     else:
            #         print(f"Symbol {symbol} did not return candles")

    def evaluate_stocks(self):
        cur = self.db_conn.cursor()
        cur.execute("SELECT DISTINCT(symbol_name) FROM stocks")
        symbols = cur.fetchall()
        period = 24 * 10
        parameters = {"short_ma": list(range(1, 20)), "long_ma": list(range(21, 200))}
        profits = []
        period = 24 * 10
        parameters = {"short_ma": list(range(1, 20)), "long_ma": list(range(21, 200))}
        profits = []
        for symbol in symbols:
            candles = pd.read_sql(
                f"SELECT ctm, ctmString, low, high, open, close, vol FROM stocks where symbol_name = '{symbol[0]}'",
                self.db_conn,
            )
            candles = cast_candles_to_types(candles, digits=None)
            cur.execute(
                f"SELECT shortSelling FROM symbols WHERE symbol = '{symbol[0]}' ORDER BY time LIMIT 1"
            )
            short_enabled = cur.fetchall()[0][0]
            trading_estimator = TradingEstimator(
                period=period,
                capital=self.capital * self.max_risk,
                symbol=symbol[0],
                short_enabled=short_enabled,
            )
            if period < (candles.shape[0] - 200):
                trend = test_trend(candles, period=period)
                if (trend >= 0) or (trend < 0 and short_enabled):
                    estimator = trading_estimator.fit(candles)
                    clf = RandomizedSearchCV(
                        estimator, parameters, n_iter=50, verbose=0, cv=2
                    )
                    clf.fit(candles)
                    print(" Results from Grid Search ")
                    print(
                        f"    The best estimator across ALL searched params: {clf.best_estimator_}"
                    )
                    print(
                        f"    The best score across ALL searched params: {clf.best_score_}"
                    )
                    print(
                        f"    The best parameters across ALL searched params: {clf.best_params_}"
                    )
                    if clf.best_score_ > 0:
                        self.insert_params(
                            day=todayms,
                            symbol=symbol[0],
                            score=clf.best_score_,
                            short_ma=clf.best_params_["short_ma"],
                            long_ma=clf.best_params_["long_ma"],
                        )
                    # fig, profit = back_test(
                    #     candles,
                    #     period=period,
                    #     capital=self.capital * self.max_risk,
                    #     symbol=symbol[0],
                    #     short_ma=4,
                    #     long_ma=112,
                    #     trend=trend,
                    #     short_enabled=short_enabled,
                    #     fig=fig,
                    #     show=False,
                    # )
                    # if profit > 0:
                    #     potential_symbols.append[symbol]

                else:
                    print(f"Downward trend and short not enabled")
            trading_estimator = TradingEstimator(
                period=period,
                capital=self.capital * self.max_risk,
                symbol=symbol[0],
                short_enabled=short_enabled,
            )
            if period < (candles.shape[0] - 200):
                trend = test_trend(candles, period=period)
                if (trend >= 0) or (trend < 0 and short_enabled):
                    estimator = trading_estimator.fit(candles)
                    clf = RandomizedSearchCV(
                        estimator, parameters, n_iter=50, verbose=0, cv=2
                    )
                    clf.fit(candles)
                    print(" Results from Grid Search ")
                    print(
                        f"    The best estimator across ALL searched params: {clf.best_estimator_}"
                    )
                    print(
                        f"    The best score across ALL searched params: {clf.best_score_}"
                    )
                    print(
                        f"    The best parameters across ALL searched params: {clf.best_params_}"
                    )
                    if clf.best_score_ > 0:
                        self.insert_params(
                            day=todayms,
                            symbol=symbol[0],
                            score=clf.best_score_,
                            short_ma=clf.best_params_["short_ma"],
                            long_ma=clf.best_params_["long_ma"],
                        )
                    # fig, profit = back_test(
                    #     candles,
                    #     period=period,
                    #     capital=self.capital * self.max_risk,
                    #     symbol=symbol[0],
                    #     short_ma=4,
                    #     long_ma=112,
                    #     trend=trend,
                    #     short_enabled=short_enabled,
                    #     fig=fig,
                    #     show=False,
                    # )
                    # if profit > 0:
                    #     potential_symbols.append[symbol]

                else:
                    print(f"Downward trend and short not enabled")
            else:
                print(f"Not enough values in candles {candles.shape[0]}")
        print(f"Profits: {np.nansum(profits)}")

    def trader_logic():
        #   if bought
        #       current_profit = c_price - buy_price
        #       if current_profit <= 0:
        #           SELL
        #       if current_profit > 0:
        #           potential_profit.append(current_profit)
        #           if last_price > c_price * 1.05:
        #               SELL
        # UPWARD CROSSOVER
        #   if ask < (capital_at_risk + profits)
        #       BUY
        # DOWNWARD CROSSOVER
        #   if short enabled
        #       if ask < (capital_at_risk + profits)
        #           SHORT
        #   else
        #       NOTHING
        #

        pass
