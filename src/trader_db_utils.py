import os
import pandas as pd
from trader_utils import *
from sqlalchemy import create_engine
import urllib
from xAPIConnector import *
from creds import creds
import pyodbc
from pyodbc import IntegrityError
import psycopg2
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import threading
import re
from logger_settings import setup_logging
import logging

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
logger = setup_logging(logger, logs_path, __name__)

TYPES = {
    "SQL": {
        "trades": {
            "tradeid": "int",
            "trader_name": "String",
            "close_price": "double",
            "close_time": "Time",
            "closed": "BIT",
            "cmd": "int",
            "comment": "String",
            "commission": "double",
            "customComment": "String",
            "digits": "int",
            "expiration": "Time",
            "margin_rate": "double",
            "offset": "int",
            "open_price": "double",
            "open_time": "Time",
            "order": "int",
            "order2": "int",
            "position": "int",
            "profit": "double",
            "sl": "double",
            "state": "String",
            "storage": "double",
            "symbol": "String",
            "tp": "double",
            "type": "int",
            "volume": "double",
            "nominalValue": "double",
        }
    },
    "TS": {},
}

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


def test_query(sql, conn):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        res = cur.fetchone()
        cur.close()
        if res is None:
            return False
        else:
            return True
    except Exception as e:
        return False


def execut_query(sql, conn, mode="dict"):
    cur = conn.cursor()
    cur.execute(sql)
    fields = [i[0] for i in cur.description]
    if mode == "dict" or mode == 1:
        values = list(cur.fetchall()[0])
        return to_dict(fields, values)
    elif mode == "df" or mode == 2:
        values = np.array(cur.fetchall())
        return to_df(fields, values)
    elif mode == "list" or mode == 3:
        values = list(cur.fetchall())
        return values
    elif mode == "insert" or mode == 4:
        conn.commit()


def execute_insert(db, table, obj):
    if db == "SQL":
        conn = DB_CONN
    elif db == "TS":
        conn = TS_CONN
    sql = generate_insert_stmt(db, table, obj)
    sql = format_db_naming(db, sql)

    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
    except IntegrityError as e:
        logger.info(f"Integrity Error | trade already exists")
        sql = generate_update_stmt(db, table, obj)
        sql = format_db_naming(db, sql, mode="update")
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
    except Exception as e:
        logger.info(f"Exception at insert | {e}")
        print()


def format_db_naming(db, sql, mode="insert"):
    if mode == "insert":
        if db == "SQL":
            sql = sql.replace("order2,", "[order2],")
            sql = sql.replace("order,", "[order],")
            sql = sql.replace("position,", "[position],")
            sql = sql.replace("type,", "[type],")
    elif mode == "update":
        if db == "SQL":
            sql = sql.replace("order2=", "[order2]=")
            sql = sql.replace("order=", "[order]=")
            sql = sql.replace("position=", "[position]=")
            sql = sql.replace("type=", "[type]=")

    return sql


def generate_insert_stmt(db, table, obj):
    if db == "SQL":
        insert_stmt = f"INSERT INTO master.dbo.{table}"
    elif db == "TS":
        insert_stmt = f"INSERT INTO public.{table}"
    key_list = list(obj.keys())
    for key in key_list:
        if obj[key] is None:
            obj.pop(key)
        elif key not in list(TYPES[db][table].keys()):
            obj.pop(key)

    sql = insert_stmt + "(" + ",".join(obj.keys()) + ")VALUES("

    for key in obj.keys():
        try:
            if TYPES[db][table][key] == "String":
                sql = sql + f"'{obj[key]}',"

            elif TYPES[db][table][key] == "double" or TYPES[db][table][key] == "int":
                sql = sql + f"{obj[key]},"

            elif TYPES[db][table][key] == "BIT":
                if obj[key]:
                    sql = sql + f"1,"
                else:
                    sql = sql + f"0,"

            elif TYPES[db][table][key] == "Time":
                val = xtb_time_to_date(obj[key])
                sql = sql + f"'{val}',"
        except KeyError as e:
            continue

    sql = sql[:-1] + ");"

    return sql


def generate_update_stmt(db, table, obj):
    if db == "SQL":
        update_stmt = f"UPDATE master.dbo.{table} SET "
    elif db == "TS":
        update_stmt = f"UPDATE public.{table} SET "
    key_list = list(obj.keys())
    for key in key_list:
        if obj[key] is None:
            obj.pop(key)
    sql = update_stmt
    for key in obj.keys():
        if TYPES[db][table][key] == "String":
            sql = sql + f"{key}='{obj[key]}',"

        elif TYPES[db][table][key] == "double" or TYPES[db][table][key] == "int":
            sql = sql + f"{key}={obj[key]},"

        elif TYPES[db][table][key] == "BIT":
            if obj[key]:
                sql = sql + f"{key}=1,"
            else:
                sql = sql + f"{key}=0,"

        elif TYPES[db][table][key] == "Time":
            val = xtb_time_to_date(obj[key])
            sql = sql + f"{key}='{val}',"

    sql = sql[:-1] + " WHERE "
    for key in prim_keys.keys():
        if TYPES[db][table][key] == "String":
            sql = sql + f"{key}='{prim_keys[key]}'"

        elif TYPES[db][table][key] == "double" or TYPES[db][table][key] == "int":
            sql = sql + f"{key}={prim_keys[key]}"

        elif TYPES[db][table][key] == "BIT":
            if prim_keys[key]:
                sql = sql + f"{key}=1"
            else:
                sql = sql + f"{key}=0"

        elif TYPES[db][table][key] == "Time":
            val = xtb_time_to_date(prim_keys[key])
            sql = sql + f"{key}='{val}'"

        sql = sql + " and "
    sql = sql[:-5] + ";"
    return sql


def to_df(columns, data):
    df = pd.DataFrame(
        columns=columns,
        data=data,
    )
    return df


def to_dict(keys, values):
    dic = {}
    for key, value in zip(keys, values):
        dic[key] = value
    return dic


def get_trader_id(name):
    sql = f"SELECT traderid FROM traders WHERE tradername = '{name}'"
    try:
        cur = DB_CONN.cursor()
        cur.execute(sql)
        traderid = cur.fetchone()
        if traderid is not None:
            traderid = traderid[0]
        cur.close()
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
        cur.close()
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
    sql = f"INSERT INTO trading_params(symbol, date, score, short_ma, long_ma, profit_exit, loss_exit, min_angle)VALUES('{symbol}', '{day}', {score}, {short_ma}, {long_ma}, {profit_exit}, {loss_exit}, {min_angle});"
    try:
        cur = DB_CONN.cursor()
        cur.execute(sql)
        DB_CONN.commit()
        cur.close()
    except Exception as e:
        logger.info(f"Exception | insert_trading_params | {e}")


def insert_tick(tick):
    try:
        tick["timestamp"] = xtb_time_to_date(int(tick["timestamp"]), local_tz=True)
        # logger.debug(f"TICK {tick['timestamp']} | {datetime.now()}")
        cur = TS_CONN.cursor()
        cur.execute(
            f"INSERT INTO mili_ticks (timestamp, symbol, ask, bid, high, low, askVolume, bidVolume, tick_level, quoteId, spreadTable, spreadRaw) VALUES ('{tick['timestamp']}', '{tick['symbol']}', {tick['ask']}, {tick['bid']}, {tick['high']}, {tick['low']}, {tick['askVolume']}, {tick['bidVolume']}, {tick['level']}, {tick['quoteId']}, {tick['spreadTable']}, {tick['spreadRaw']});"
        )
        TS_CONN.commit()
        cur.close()
    except (Exception, psycopg2.Error) as error:
        logger.debug(error.pgerror)


def insert_candle(candle):
    try:
        candle["ctmString"] = pd.to_datetime(
            candle["ctmString"], format="%b %d, %Y, %I:%M:%S %p"
        ).strftime("%Y-%m-%d %H:%M:%S.%f")
        # logger.debug(f"CANDLE {candle['ctmString']} | {datetime.now()}")
        cur = TS_CONN.cursor()
        cur.execute(
            f"INSERT INTO candles (symbol, ctm, ctmString, low, high, open, close, vol, quoteId) VALUES ('{candle['symbol']}', {candle['ctm']}, '{candle['ctmString']}', {candle['low']}, {candle['high']}, {candle['open']}, {candle['close']}, {candle['vol']}, -1);"
        )
        TS_CONN.commit()
        cur.close()
    except (Exception, psycopg2.Error) as error:
        logger.debug(f"Error inserting candles | {error}")


def insert_trade(trade):
    try:
        logger.debug(f"TRADE {trade['open_time']} | {datetime.now()}")
        if trade["customComment"] is None:
            trade["trader_name"] = "WebPage"
        else:
            trade["trader_name"] = trade["customComment"]
        execute_insert(db="SQL", table="trades", obj=trade)
    except Exception as error:
        logger.debug(f"Error inserting candles | {error}")


def get_last_two_candles(symbol, short_ma, long_ma):
    sql = f"SELECT ctm, ctmstring, symbol, low, high, open, close, vol, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {short_ma} PRECEDING AND CURRENT ROW)AS short_ma, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {long_ma} PRECEDING AND CURRENT ROW)AS long_ma  FROM candles WHERE symbol = '{symbol}' ORDER BY ctmstring DESC Limit 2;"
    try:
        cursor = TS_CONN.cursor()
        cursor.execute(sql)
        candles = cursor.fetchall()
        cursor.close()

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
        symbol_info = execut_query(sql, DB_CONN)
    except Exception as e:
        logger.info(f"Exception | get_symbol_info | {e}")
        symbol_info = None
    return symbol_info


def insert_trade_session(trader_name, symbol, is_bought, is_short):
    if is_bought:
        state = "BUY"
    elif is_short:
        state = "SHORT"
    else:
        state = "OUT"
    sql = f"INSERT INTO trade_session (trader_name, ctmstring, symbol, state)VALUES('{trader_name}', '{datetime.now()}', '{symbol}', '{state}');"
    try:
        cursor = TS_CONN.cursor()
        cursor.execute(sql)
        TS_CONN.commit()
    except Exception as e:
        logger.info(f"Exception | insert_trade_session | {e}")


def get_trading_params(today=False):
    if today:
        sql = f"SELECT * FROM trading_params where date = '{datetime.today().strftime('%Y-%m-%d')}' ORDER BY score DESC"
    else:
        sql = f"SELECT * FROM trading_params ORDER BY score DESC"
    params = execut_query(sql, DB_CONN, mode="df")
    return params


def get_distict_symbols():
    sql = "SELECT DISTINCT(symbol_name) FROM stocks"
    distinct_symbols = execut_query(sql, DB_CONN, mode="list")
    return distinct_symbols


def get_symbol_stats(symbol, start_date):
    sql = f"select avg(close) as avg_close, max(close) as max_close, min(close) as min_close, stddev(close) as std_close from candles where symbol = '{symbol}' and ctmstring > '{start_date}'"
    symbol_stats = execut_query(sql, TS_CONN)
    return symbol_stats


def get_symbol_today():
    timeString = get_today_timeString()
    sql = f"SELECT * FROM symbols WHERE timeString >= '{timeString}'"
    if test_query(sql, DB_CONN):
        symbol_today = execut_query(sql, DB_CONN, mode=2)
    else:
        symbol_today = False
    return symbol_today


def get_candles_today():
    sql = f"SELECT * FROM candles WHERE ctmstring >= '{today}'"
    if test_query(sql, TS_CONN):
        return False
    else:
        return True


def find_closed_trade(order_n):
    sql = f"SELECT [order] FROM trades WHERE [order] = {order_n}"
    if test_query(sql=sql, conn=DB_CONN):
        return True
    else:
        return False
