import os
import pandas as pd
from utils import *

today = get_today()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../data/"
symbol_path = curr_path + "../../symbols/"
cluster_path = curr_path + "../../clusters/"
model_path = curr_path + "../../model/"
result_path = curr_path + "../../result/"
docs_path = curr_path + "../../docs/"

class trader:
    def __init__(self, capital, max_risk, trade_type) -> None:
        self.capital = capital
        self.max_risk = max_risk
        self.trade_type = trade_type
        self.invested_capital = 0
        self.active_trades = pd.DataFrame(data=[], columns=["Date", "symbol", "amount", "buy_price", "stop_loss", "take_profit"])
        self.historic_trades = pd.DataFrame(data=[], columns=["Type","Date_entry", "Date_close", "symbol", "amount", "buy_price", "initial_stop_loss", "initial_take_profit", "stop_loss_trail", "take_profit_trail"])
    
    def make_trade(self, symbol, current_price, stop_loss=None, take_profit=None):
        symbol_df = pd.read_pickle(f"{symbol_path}{self.trade_type}symbols_{today}.pickle")
        symbol_info = symbol_df[symbol_df["symbol"] == symbol]
        
        


        print()