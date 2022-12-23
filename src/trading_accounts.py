import os
import pandas as pd
from utils import *
import sqlite3
from xAPIConnector import *
import creds

today = get_today()
todayms = get_today_ms()
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
        self.traderid = int(todayms)
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

    def insert_trader(self, conn):
        sql = f"INSERT OR IGNORE INTO traders (traderid, tradername, creation_date, initial_capital, max_risk) VALUES ({int(self.traderid)},{self.name},'{todayms}',{self.capital},{self.max_risk},0, 0);"
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            print(f"Exception | insert_trader | {e}")

    def insert_trade(
        self,
        conn,
        tradeid,
        date_entry,
        symbol,
        shares,
        entry_price,
        entry_stop_loss,
        entry_take_profit,
        out_price,
    ):
        sql = f"INSERT INTO trades (tradeid, traderid, date_entry, symbol, shares, entry_price, entry_stop_loss, entry_take_profit, out_price) VALUES ({tradeid},{self.traderid},{date_entry},{symbol},{shares},{entry_price},{entry_stop_loss},{entry_take_profit},{out_price});"
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            print(f"Exception | insert_trade | {e}")

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
