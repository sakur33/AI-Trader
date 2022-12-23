import os
import pandas as pd
from utils import *
import sqlite3
from xAPIConnector import *

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


class trader:
    def __init__(self, name, capital, max_risk, trade_type) -> None:
        self.traderid = int(todayms)
        self.name = name
        self.capital = capital
        self.max_risk = max_risk
        db_conn = self.client("ai_trader.db")
        self.stream_client = APIStreamClient()
        self.client = APIClient()

    def make_trade(self, symbol_info, current_price, stop_loss=None, take_profit=None):
        # TODO implement make trade:
        pass

    def create_connection(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(f"{database_path}{db_file}")
        except Exception as e:
            print(e)
        return conn

    def insert_trader(self, conn):
        sql = f"INSERT INTO traders (traderid, tradername, creation_date, initial_capital, max_risk) VALUES ({int(self.traderid)},{self.name},'{todayms}',{self.capital},{self.max_risk},0, 0);"
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

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
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    def insert_trans(self, conn, tradeid, trans_date, trans_type, balance):
        sql = f"INSERT INTO balance (traderid, tradeid, trans_date, trans_type, balance) VALUES ({self.traderid},{tradeid},{trans_date},{trans_type},{balance});"
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    def get_profit_losses(self):
        # TODO implement query to get profits and losses
        pass
