import os
import pandas as pd
from trader_utils import *
import sqlite3
from xAPIConnector import *
from creds import creds
import pywt
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


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


def create_db_conn(db_file):
    conn = None
    try:
        conn = sqlite3.connect(f"{database_path}{db_file}")
    except Exception as e:
        print(f"Exception | create_db_conn | {e}")
    return conn


def add_rolling_means(df, short, long):
    df["MA_short"] = df["close"].rolling(window=short).mean()
    df["MA_long"] = df["close"].rolling(window=long).mean()
    return df


db_conn = create_db_conn("ai_trader.db")
cur = db_conn.cursor()
cur.execute("SELECT DISTINCT(symbol_name) FROM stocks")
symbols = cur.fetchall()


candles = pd.read_sql(
    f"SELECT ctm, ctmString, low, high, open, close, vol FROM stocks where symbol_name = '{symbols[0][0]}'",
    db_conn,
)
candles = cast_candles_to_types(candles, digits=None, dates=False)
candles = add_rolling_means(candles, 500, 1000)


y = np.array(candles.close)
y_short = np.array(candles.MA_short)
y_long = np.array(candles.MA_long)

x = np.array(candles.ctm)

date_array = pd.to_datetime(candles.ctmString)

plt.plot(range(len(y)), y)
plt.plot(range(len(y_short)), y_short)
plt.plot(range(len(y_long)), y_long)

y_detrend = signal.detrend(y)
x_fft = np.fft.fft(y_detrend)
N = int(len(x_fft))
n = np.arange(N)
T = N / len(y_detrend)
freq = n / T

plt.stem(freq, np.abs(x_fft), "b", markerfmt=" ", basefmt="-b")

n_oneside = N // 2
f_oneside = freq[:n_oneside]

plt.plot(f_oneside, np.abs(x_fft[:n_oneside]), "b")

t_h = 1 / f_oneside / len(y_detrend)
