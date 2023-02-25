import os
import pandas as pd
from trader_utils import *
from sqlalchemy import create_engine
import urllib
from creds import creds
import pyodbc
from pyodbc import IntegrityError
import psycopg2
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import threading
import re
from logger_settings import *
import logging
from custom_exceptions import *
from connection_clock import Clock

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


class TraderDB:
    def __init__(
        self,
        sql_driver="{ODBC Driver 17 for SQL Server}",
        sql_server="localhost",
        sql_port=1433,
        sql_database="master",
        sql_user="sa",
        sql_password="yourStrong(!)Password",
        dbname="postgres",
        user="postgres",
        password="1234",
        host="localhost",
        port=5432,
        logger=logging.getLogger(__name__),
    ) -> None:
        self.logger = logger

        try:
            self.DB_CONN = pyodbc.connect(
                driver="{ODBC Driver 17 for SQL Server}",
                server="localhost",
                port=1433,
                database="master",
                user="sa",
                password="yourStrong(!)Password",
            )
        except Exception as e:
            self.logger.info(f"Exception | create db_conn | {e}")
            raise DbError

        try:
            quoted = urllib.parse.quote_plus(
                "DRIVER={ODBC Driver 17 for SQL Server};"
                "SERVER=localhost;"
                "DATABASE=master;"
                "UID=sa;"
                "PWD=yourStrong(!)Password"
            )
            self.DB_ENGINE = create_engine(
                "mssql+pyodbc:///?odbc_connect={}".format(quoted)
            )
        except Exception as e:
            self.logger.info(f"Exception | crate db_engine | {e}")
            raise DbError

        try:
            dbname = "postgres"
            user = "postgres"
            password = "1234"
            host = "localhost"
            port = 5432
            connection = f"dbname ={dbname} user={user} password={password} host={host} port={port}"
            self.TS_CONN = psycopg2.connect(connection)
        except Exception as e:
            self.logger.info(f"Exception | crate ts_con | {e}")
            raise DbError
        self.CLOCK = Clock()

    def test_query(self, sql, conn):
        self.CLOCK.wait_clock()
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

    def execute_query(self, sql, conn, mode="dict"):
        self.CLOCK.wait_clock()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            fields = [i[0] for i in cur.description]
            if mode == "dict" or mode == 1:
                values = list(cur.fetchall()[0])
                return self.to_dict(fields, values)
            elif mode == "df" or mode == 2:
                values = np.array(cur.fetchall())
                return self.to_df(fields, values)
            elif mode == "list" or mode == 3:
                values = list(cur.fetchall())
                return values
            elif mode == "insert" or mode == 4:
                conn.commit()
        except Exception as e:
            raise DbError

    def execute_insert(self, db, table, obj):
        self.CLOCK.wait_clock()
        if db == "SQL":
            conn = self.DB_CONN
        elif db == "TS":
            conn = self.TS_CONN
        sql = self.generate_insert_stmt(db, table, obj)
        sql = self.format_db_naming(db, sql)

        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            cur.close()
        except IntegrityError as e:
            self.logger.info(f"Integrity Error | trade already exists")
            sql = self.generate_update_stmt(db, table, obj)
            sql = self.format_db_naming(db, sql, mode="update")
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            cur.close()
        except Exception as e:
            self.logger.info(f"Exception at insert | {e}")
            self.CLOCK.wait_clock()
            self.execute_insert(db, table, obj)
            raise DbError

    def format_db_naming(self, db, sql, mode="insert"):
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

    def generate_insert_stmt(self, db, table, obj):
        try:
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

                    elif (
                        TYPES[db][table][key] == "double"
                        or TYPES[db][table][key] == "int"
                    ):
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
        except Exception as e:
            raise DbError

    def generate_update_stmt(self, db, table, obj):
        try:
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

                elif (
                    TYPES[db][table][key] == "double" or TYPES[db][table][key] == "int"
                ):
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
            sql = sql[:-5] + ";"
            return sql
        except Exception as e:
            raise DbError

    def to_df(self, columns, data):
        try:
            df = pd.DataFrame(
                columns=columns,
                data=data,
            )
            return df
        except Exception as e:
            raise DbError

    def to_dict(self, keys, values):
        try:
            dic = {}
            for key, value in zip(keys, values):
                dic[key] = value
            return dic
        except Exception as e:
            raise DbError

    def get_trader_id(self, name):
        try:
            sql = f"SELECT traderid FROM traders WHERE tradername = '{name}'"
            try:
                cur = self.DB_CONN.cursor()
                cur.execute(sql)
                traderid = cur.fetchone()
                if traderid is not None:
                    traderid = traderid[0]
                cur.close()
            except Exception as e:
                self.logger.info(f"Exception | get trader id | {e}")
                traderid = None
            return traderid
        except Exception as e:
            raise DbError

    def insert_trader(self, name, capital, max_risk):
        self.CLOCK.wait_clock()
        try:
            todayms, today_int = get_today_ms()
            sql = f"INSERT INTO traders (tradername, creation_date, initial_capital, max_risk) VALUES ('{name}','{todayms}',{capital},{max_risk});"
            self.logger.info(sql)
            try:
                cur = self.DB_CONN.cursor()
                cur.execute(sql)
                self.DB_CONN.commit()
                cur.close()
                traderid = self.get_trader_id(name)
            except Exception as e:
                self.logger.info(f"Exception | insert_trader | {e}")
                traderid = None
            return traderid
        except Exception as e:
            raise DbError

    def insert_symbols(self, symbols_df):
        self.CLOCK.wait_clock()
        try:
            symbols_df.to_sql(
                "symbols", schema="dbo", con=self.DB_ENGINE, if_exists="replace"
            )
        except Exception as e:
            self.logger.info(f"Exception | insert symbol | {e}")
            raise DbError

    def insert_params(
        self, day, symbol, score, short_ma, long_ma, profit_exit, loss_exit, min_angle
    ):
        self.CLOCK.wait_clock()
        sql = f"INSERT INTO trading_params(symbol, date, score, short_ma, long_ma, profit_exit, loss_exit, min_angle)VALUES('{symbol}', '{day}', {score}, {short_ma}, {long_ma}, {profit_exit}, {loss_exit}, {min_angle});"
        try:
            cur = self.DB_CONN.cursor()
            cur.execute(sql)
            self.DB_CONN.commit()
            cur.close()
        except Exception as e:
            self.logger.info(f"Exception | insert_trading_params | {e}")
            raise DbError

    def insert_tick(self, tick):
        self.CLOCK.wait_clock()
        try:
            tick["timestamp"] = xtb_time_to_date(int(tick["timestamp"]), local_tz=True)
            # self.logger.debug(f"TICK {tick['timestamp']} | {datetime.now()}")
            cur = self.TS_CONN.cursor()
            cur.execute(
                f"INSERT INTO mili_ticks (timestamp, symbol, ask, bid, high, low, askVolume, bidVolume, tick_level, quoteId, spreadTable, spreadRaw) VALUES ('{tick['timestamp']}', '{tick['symbol']}', {tick['ask']}, {tick['bid']}, {tick['high']}, {tick['low']}, {tick['askVolume']}, {tick['bidVolume']}, {tick['level']}, {tick['quoteId']}, {tick['spreadTable']}, {tick['spreadRaw']});"
            )
            self.TS_CONN.commit()
            cur.close()
        except (Exception, psycopg2.Error) as error:
            self.logger.debug(error.pgerror)
            raise DbError

    def insert_candle(self, candle):
        self.CLOCK.wait_clock()
        try:
            candle["ctmString"] = pd.to_datetime(
                candle["ctmString"], format="%b %d, %Y, %I:%M:%S %p"
            ).strftime("%Y-%m-%d %H:%M:%S.%f")
            # self.logger.debug(f"CANDLE {candle['ctmString']} | {datetime.now()}")
            cur = self.TS_CONN.cursor()
            sql = f"INSERT INTO candles (symbol, ctm, ctmString, low, high, open, close, vol, quoteId) VALUES ('{candle['symbol']}', {candle['ctm']}, '{candle['ctmString']}', {candle['low']}, {candle['high']}, {candle['open']}, {candle['close']}, {candle['vol']}, -1);"
            cur.execute(sql)
            self.TS_CONN.commit()
            cur.close()
        except (Exception, psycopg2.Error) as error:
            self.logger.debug(f"Error inserting candles | {error}")
            self.CLOCK.wait_clock()
            self.insert_candle(candle)

    def insert_candle_batch(self, candles):
        try:
            cur = self.TS_CONN.cursor()
            base_sql = f"INSERT INTO candles (symbol, ctm, ctmString, low, high, open, close, vol, quoteId) VALUES "
            sql = ""
            end_sql = ";"
            cont = 0
            for index, row in candles.iterrows():
                sql = (
                    sql
                    + f"('{row['symbol']}', {row['ctm']}, '{row['ctmString']}', {row['low']}, {row['high']}, {row['open']}, {row['close']}, {row['vol']}, -1),"
                )
                cont += 1
                if cont >= 1000:
                    cur.execute(base_sql + sql[:-1] + end_sql)
                    self.TS_CONN.commit()
                    sql = ""
                    cont = 0
            if sql != "":
                cur.execute(base_sql + sql[:-1] + end_sql)
                self.TS_CONN.commit()
            cur.close()
        except Exception as e:
            self.logger.debug(f"Error inserting candle batch| {e}")
            self.CLOCK.wait_clock()
            self.insert_candle_batch(candles)

    def insert_trade(self, trade):
        self.CLOCK.wait_clock()
        try:
            self.logger.debug(f"TRADE {trade['open_time']} | {datetime.now()}")
            if trade["customComment"] is None:
                trade["trader_name"] = "WebPage"
            else:
                trade["trader_name"] = trade["customComment"]
            self.execute_insert(db="SQL", table="trades", obj=trade)
        except Exception as error:
            self.logger.debug(f"Error inserting candles | {error}")
            self.CLOCK.wait_clock()
            self.insert_trade(trade)

    def get_last_tick(self, symbol):
        self.CLOCK.wait_clock()
        try:
            sql = f"SELECT * FROM mili_ticks WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT 1;"
            tick = self.execute_query(sql, self.TS_CONN, mode="dict")
            return tick
        except Exception as e:
            self.logger.info(f"Exception | get_last_two_candles | {e}")
            tick = None
            return tick
        except Exception as e:
            raise DbError

    def get_last_candle(self, symbol):
        self.CLOCK.wait_clock()
        try:
            sql = f"SELECT * FROM candles WHERE symbol = '{symbol}' ORDER BY ctm DESC LIMIT 1;"
            candle = self.execute_query(sql, self.TS_CONN, mode="dict")
            return candle
        except Exception as e:
            self.logger.info(f"Exception | get_last_two_candles | {e}")
            candle = None
            return candle
        except Exception as e:
            raise DbError

    def get_last_two_candles(self, symbol, short_ma, long_ma):
        self.CLOCK.wait_clock()
        try:
            sql = f"SELECT ctm, ctmstring, symbol, low, high, open, close, vol, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {short_ma} PRECEDING AND CURRENT ROW)AS short_ma, AVG(close) OVER(ORDER BY ctmstring ROWS BETWEEN {long_ma} PRECEDING AND CURRENT ROW)AS long_ma  FROM candles WHERE symbol = '{symbol}' ORDER BY ctmstring DESC Limit 2;"
            try:
                cursor = self.TS_CONN.cursor()
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
                self.logger.info(f"Exception | get_last_two_candles | {e}")
                candles_df = None
            return candles_df
        except Exception as e:
            raise DbError

    def get_symbol_info(self, symbol):
        self.CLOCK.wait_clock()
        try:
            sql = f"SELECT TOP(1) * FROM symbols WHERE symbol = '{symbol}' order by time DESC"
            try:
                symbol_info = self.execute_query(sql, self.DB_CONN, mode="dict")
            except Exception as e:
                self.logger.info(f"Exception | get_symbol_info | {e}")
                symbol_info = None
            return symbol_info
        except Exception as e:
            raise DbError

    def get_symbol_infos(self, type="FX"):
        self.CLOCK.wait_clock()
        try:
            sql = f"SELECT * FROM symbols WHERE categoryName = '{type}' order by time DESC"
            try:
                symbol_info = self.execute_query(sql, self.DB_CONN, mode="dict")
            except Exception as e:
                self.logger.info(f"Exception | get_symbol_info | {e}")
                symbol_info = None
            return symbol_info
        except Exception as e:
            raise DbError

    def insert_trade_session(self, trader_name, symbol, is_bought, is_short):
        self.CLOCK.wait_clock()
        try:
            if is_bought:
                state = "BUY"
            elif is_short:
                state = "SHORT"
            else:
                state = "OUT"
            sql = f"INSERT INTO trade_session (trader_name, ctmstring, symbol, state)VALUES('{trader_name}', '{datetime.now()}', '{symbol}', '{state}');"
            try:
                cursor = self.TS_CONN.cursor()
                cursor.execute(sql)
                self.TS_CONN.commit()
            except Exception as e:
                self.logger.info(f"Exception | insert_trade_session | {e}")
        except Exception as e:
            raise DbError

    def get_trading_params(self, today=False):
        self.CLOCK.wait_clock()
        try:
            if today:
                sql = f"SELECT * FROM trading_params where date = '{datetime.today().strftime('%Y-%m-%d')}' ORDER BY score DESC"
            else:
                sql = f"SELECT * FROM trading_params ORDER BY score DESC"
            params = self.execute_query(sql, self.DB_CONN, mode="df")
            return params
        except Exception as e:
            raise DbError

    def get_distict_symbols(self):
        self.CLOCK.wait_clock()
        try:
            sql = "SELECT DISTINCT(symbol_name) FROM stocks"
            distinct_symbols = self.execute_query(sql, self.DB_CONN, mode="list")
            return distinct_symbols
        except Exception as e:
            raise DbError

    def get_symbol_stats(self, symbol, start_date):
        self.CLOCK.wait_clock()
        try:
            sql = f"select avg(close) as avg_close, max(close) as max_close, min(close) as min_close, stddev(close) as std_close from candles where symbol = '{symbol}' and ctmstring > '{start_date}'"
            symbol_stats = self.execute_query(sql, self.TS_CONN)
            return symbol_stats
        except Exception as e:
            raise DbError

    def get_symbol_today(self):
        self.CLOCK.wait_clock()
        try:
            timeString = get_today_timeString()
            sql = f"SELECT * FROM symbols WHERE timeString >= '{timeString}'"
            if self.test_query(sql, self.DB_CONN):
                symbol_today = self.execute_query(sql, self.DB_CONN, mode=2)
            else:
                symbol_today = False
            return symbol_today
        except Exception as e:
            raise DbError

    def get_candles_today(self):
        try:
            sql = f"SELECT * FROM candles WHERE ctmstring >= '{get_today()}'"
            if self.test_query(sql, self.TS_CONN):
                return False
            else:
                return True
        except Exception as e:
            raise DbError

    def get_candles_range(self, start_date, end_date):
        self.CLOCK.wait_clock()
        try:
            sql = f"SELECT * FROM candles WHERE ctmString BETWEEN '{start_date}' and '{end_date}' ORDER BY ctm ASC;"
            candle = self.execute_query(sql, self.TS_CONN, mode="df")
            return candle
        except Exception as e:
            self.logger.info(f"Exception | get candles range | {e}")
            candle = None
            return candle
        except Exception as e:
            raise DbError

    def find_closed_trade(self, order_n):
        try:
            sql = f"SELECT [order] FROM trades WHERE [order] = {order_n}"
            if self.test_query(sql=sql, conn=self.DB_CONN):
                return True
            else:
                return False
        except Exception as e:
            raise DbError
