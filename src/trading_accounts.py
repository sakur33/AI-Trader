import os
import pandas as pd
from utils import *
import sqlite3
from xAPIConnector import *

today = get_today()
today_str = get_today_ms
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
        self.name = name
        self.capital = capital
        self.max_risk = max_risk
        db_conn = self.client("ai_trader.db")
        self.stream_client = APIStreamClient()
        self.client = APIClient()

    def make_trade(self, symbol_info, current_price, stop_loss=None, take_profit=None):
        # TODO: 

    def create_connection(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(f"{database_path}{db_file}")
        except Exception as e:
            print(e)
        return conn

    def insert_trader(self, conn):
        sql = f"INSERT INTO traders (tradername, creation_date, capital, max_risk, profits, losses) VALUES ({self.name},'{today}',{self.capital},{self.max_risk},0, 0);"
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
