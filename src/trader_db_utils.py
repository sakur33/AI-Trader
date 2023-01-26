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

if os.path.exists(f"{logs_path}{__name__}.log"):
    os.remove(f"{logs_path}{__name__}.log")
logger = logging.getLogger(__name__)


try:
    DB_CONN = pyodbc.connect(
        driver="{ODBC Driver 17 for SQL Server}",
        server="localhost",
        port=1433,
        database="master",
        user="sa",
        password="yourStrong(!)Password",
    )
except Exception as e:
    logger.info(f"Exception | create db_conn | {e}")

try:
    quoted = urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=master;"
        "UID=sa;"
        "PWD=yourStrong(!)Password"
    )
    DB_ENGINE = create_engine("mssql+pyodbc:///?odbc_connect={}".format(quoted))
except Exception as e:
    logger.info(f"Exception | crate db_engine | {e}")

try:
    dbname = "postgres"
    user = "postgres"
    password = "1234"
    host = "localhost"
    port = 5432
    connection = (
        f"dbname ={dbname} user={user} password={password} host={host} port={port}"
    )
    TS_CONN = psycopg2.connect(connection)
except Exception as e:
    logger.info(f"Exception | crate ts_con | {e}")


def get_trader_id(name):
    sql = f"SELECT traderid FROM traders WHERE tradername = '{name}'"
    try:
        cur = DB_CONN.cursor()
        cur.execute(sql)
        traderid = cur.fetchone()
        if traderid is not None:
            traderid = traderid[0]
    except Exception as e:
        logger.info(f"Exception | get trader id | {e}")
    return traderid


def insert_trader(name, capital, max_risk):
    todayms, today_int = get_today_ms()
    sql = f"INSERT INTO traders (tradername, creation_date, initial_capital, max_risk) VALUES ('{name}','{todayms}',{capital},{max_risk});"
    logger.info(sql)
    try:
        cur = DB_CONN.cursor()
        cur.execute(sql)
        DB_CONN.commit()
        traderid = get_trader_id(name)
    except Exception as e:
        logger.info(f"Exception | insert_trader | {e}")
    return traderid


def insert_symbols(symbols_df):
    try:
        symbols_df.to_sql("symbols", schema="dbo", con=DB_ENGINE, if_exists="replace")
    except Exception as e:
        logger.info(f"Exception | insert symbol | {e}")


def insert_params(
    day, symbol, score, short_ma, long_ma, profit_exit, loss_exit, min_angle
):
    sql = f"INSERT INTO trading_params(symbol_name, date, score, short_ma, long_ma, profit_exit, loss_exit, min_angle)VALUES('{symbol}', '{day}', {score}, {short_ma}, {long_ma}, {profit_exit}, {loss_exit}, {min_angle});"
    try:
        cur = DB_CONN.cursor()
        cur.execute(sql)
        DB_CONN.commit()
    except Exception as e:
        logger.info(f"Exception | insert_trading_params | {e}")


def insert_tick(tick):
    try:
        TS_CONN.cursor.execute(
            f"INSERT INTO ticks (timestamp, symbol, ask, bid, high, low, askVolume, bidVolume, tick_level, quoteId, spreadTable, spreadRaw) VALUES ('{tick['timestamp']}', '{tick['symbol']}', {tick['ask']}, {tick['bid']}, {tick['high']}, {tick['low']}, {tick['askVolume']}, {tick['bidVolume']}, {tick['level']}, {tick['quoteId']}, {tick['spreadTable']}, {tick['spreadRaw']});"
        )
        TS_CONN.commit()
    except (Exception, psycopg2.Error) as error:
        logger.info(error.pgerror)


def insert_candle(candle):
    try:
        cur = TS_CONN.cursor()
        cur.execute(
            f"INSERT INTO candles (symbol, ctm, ctmString, low, high, open, close, vol, quoteId) VALUES ('{candle['symbol']}', {candle['ctm']}, '{candle['ctmString']}', {candle['low']}, {candle['high']}, {candle['open']}, {candle['close']}, {candle['vol']}, -1);"
        )
        TS_CONN.commit()
    except (Exception, psycopg2.Error) as error:
        logger.info(f"Error inserting candles | {error}")


def get_last_two_candles(symbol, short_ma, long_ma):
    sql = f"SELECT ctm, ctmstring, symbol, low, high, open, close, vol, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {short_ma} PRECEDING AND CURRENT ROW)AS short_ma, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {long_ma} PRECEDING AND CURRENT ROW)AS long_ma  FROM candles WHERE symbol = '{symbol}' ORDER BY ctmstring DESC Limit 2;"
    try:
        cursor = TS_CONN.cursor()
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
    except Exception as e:
        logger.info(f"Exception | get_last_two_candles | {e}")
        candles_df = None
    return candles_df


def get_symbol_info(symbol):
    sql = f"SELECT TOP(1) * FROM symbols WHERE symbol = '{symbol}' order by time DESC"
    try:
        symbol_info = pd.read_sql_query(sql, con=DB_ENGINE)
        symbol_info = symbol_info.dict()
    except Exception as e:
        logger.info(f"Exception | get_symbol_info | {e}")
        symbol_info = None
    return symbol_info


def insert_trade_session(symbol, is_bought, is_short):
    if is_bought:
        state = "BUY"
    elif is_short:
        state = "SHORT"
    else:
        state = "OUT"
    sql = f"INSERT INTO trade_session (ctmstring, symbol, state)VALUES('{datetime.now()}', '{symbol}', '{state}');"
    try:
        cursor = TS_CONN.cursor()
        cursor.execute(sql)
        TS_CONN.commit()
    except Exception as e:
        logger.info(f"Exception | insert_trade_session | {e}")


def insert_trade(
    name,
    symbol,
    trade_type,
    open_time,
    close_time,
    volume,
    open_price,
    entry_position,
    close_price,
    out_position,
    entry_slipage,
    out_slipage,
    profit,
):

    sql = f"INSERT INTO trades (trader_name, symbol, trade_type, time_entry, time_close, volume, entry_price, entry_position ,close_price, close_position, entry_slipage, close_slipage, profit)VALUES('{name}', '{symbol}','{trade_type}', '{open_time}', '{close_time}', {volume}, {open_price}, {entry_position}, {close_price}, {out_position}, {entry_slipage}, {out_slipage}, {profit});"
    try:
        cursor = DB_CONN.cursor()
        cursor.execute(sql)
        DB_CONN.commit()
    except Exception as e:
        logger.info(f"Exception | insert_trade | {e}")


def get_trading_params():
    params = pd.read_sql("SELECT * FROM trading_params ORDER BY score DESC", DB_ENGINE)
    return params


def get_distict_symbols():
    sql = "SELECT DISTINCT(symbol_name) FROM stocks"
    cur = DB_CONN.cursor()
    cur.execute(sql)
    symbols = cur.fetchall()
    return symbols


def get_symbol_stats(symbol, start_date):
    sql = f"select avg(close) as avg_close, max(close) as max_close, min(close) as min_close, stddev(close) as std_close from candles where symbol = '{symbol}' and ctmstring > '{start_date}'"
    cur = DB_CONN.cursor()
    cur.execute(sql)
    symbols = cur.fetchall()
    return symbols
