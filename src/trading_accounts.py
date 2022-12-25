import os
import pandas as pd
from trader_utils import *
import sqlite3
from xAPIConnector import *
from creds import creds

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


class Trader:
    def __init__(self, name, capital, max_risk, trade_type) -> None:
        self.traderid = today_int
        self.user = creds["trader1"]["user"]
        self.passw = creds["trader1"]["passw"]
        self.name = name
        self.capital = capital
        self.max_risk = max_risk
        self.db_conn = self.create_db_conn("ai_trader.db")
        self.insert_trader()
        self.client = APIClient()
        self.ssid = None
        self.stream_client = None

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

    def get_profit_losses(self):
        # TODO implement query to get profits and losses
        pass

    def start_streaming(self):
        self.stream_client = APIStreamClient(
            ssId=self.ssid,
            tickFun=procTickExample,
            tradeFun=procTradeExample,
            profitFun=procProfitExample,
            tradeStatusFun=procTradeStatusExample,
        )

    def look_for_suitable_symbols_v1(self, df):
        # TODO look for suitable symbols
        # Look for symbols with:
        #   - Tight spread
        #   - STC type
        #   - ask comparable to our max_risk
        df = df[df["categoryName"] == "STC"]
        df = df[df["ask"] <= (self.capital * self.max_risk)]
        df["spread_percentage"] = (df["ask"] - df["bid"]) / df["ask"]
        df = df.sort_values(by=["spread_percentage"])
        return df

    def filter_symbols_by_liquidity(self, df):
        symbol_volumes = []
        start_date = datetime.now() - timedelta(days=10)
        for symbol in list(df["symbol"]):
            CHART_RANGE_INFO_RECORD = {
                "period": 60,
                "start": date_to_xtb_time(start_date),
                "symbol": symbol,
            }
            commandResponse = self.client.commandExecute(
                "getChartLastRequest", arguments={"info": CHART_RANGE_INFO_RECORD}
            )
            if commandResponse["status"] == False:
                error_code = commandResponse["errorCode"]
                print(f"Login failed. Error code: {error_code}")
                symbols_df = None
            else:
                candles = return_as_df(commandResponse["returnData"])
                symbol_volumes.append(candles["vol"].max())
        df["daily_max_volume"] = symbol_volumes
        df = df[df["daily_max_volume"] > df["daily_max_volume"].quantile(0.90)]
        return df

    def update_stocks(self, df, period, save=False):
        start_date = datetime.now() - timedelta(days=365)
        for symbol in list(df["symbol"]):
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
                            "stocks", self.db_conn, if_exists="replace", index=False
                        )
                    except Exception as e:
                        print(f"Exception | insert symbol | {e}")
                else:
                    print(f"Symbol {symbol} did not return candles")

    def evaluate_stocks(self):
        cur = self.db_conn.cursor()
        cur.execute("SELECT DISTINCT(symbol_name) FROM stocks")
        symbols = cur.fetchall()
        for symbol in symbols:
            candles = pd.read_sql(
                f"SELECT ctm, ctmString, low, high, open, close, vol FROM stocks where symbol_name = {symbol}",
                self.db_conn,
            )
            trend = test_trend(candles, period=24)
            if trend < 1:
                # DOWNWARD
                # TODO test short is enabled
                # TODO update backtest with short enabled
                back_test(
                    candles,
                    period=24 * 10,
                    capital=self.capital * self.max_risk,
                    symbol=symbol,
                    short=4,
                    long=112,
                )
            elif trend >= 1:
                # UPWARD
                back_test(
                    candles,
                    period=24 * 10,
                    capital=250 * 0.05,
                    symbol=symbol,
                    short=4,
                    long=112,
                )

        pass

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
