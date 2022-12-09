import websockets
import json
import asyncio
import time
from datetime import datetime, timedelta
import pytz
from utils import xtb_time_to_date, date_to_xtb_time
from creds import user, passw
import pandas as pd
import pathlib


class XTBclient:

    def __init__(self) -> None:
        self.uri = "wss://ws.xtb.com/demo"
        self.login = {
            "command": "login",
            "arguments": {
                "userId": user,
                "password": passw,
                "appId": "test",
                "appName": "test",
            },
        }
        self.sessionid = None
        self.timezone = pytz.timezone('GMT')
        self.initial_time = datetime(1970, 1, 1, 00, 00, 00, 000000, tzinfo=self.timezone)
        self.logout = {"command": "logout"}
        self.ping = {"command": "ping"}

    async def connect(self):
        connection = await websockets.connect(self.uri, max_size=1_000_000_000)
        await connection.send(json.dumps(self.login))
        response = await connection.recv()
        print(f"{datetime.now()} | Response: {response}")
        self.sessionid = json.loads(response)["streamSessionId"]
        return connection

    async def sendMessage(self, connection, command):
        """
        Sending message to webSocket server
        """
        try:
            await connection.send(json.dumps(command))
            response = await connection.recv()
            return json.loads(response)
        except websockets.exceptions.ConnectionClosed:
            print(f"{datetime.now()} | Message not sent.")
            print(f"{datetime.now()} | Connection with server closed.")
            return None
        except Exception as e:
                print(f"{datetime.now()} | Other exception")
                print(f"{datetime.now()} | {e}")
                return None

    async def disconnect(self):
        connection = await websockets.connect("wss://ws.xtb.com/demoStream")
        await connection.send(json.dumps(self.logout))
        response = await connection.recv()
        print(f"{datetime.now()} | Response: {response}")
    
    def return_as_df(self, response):
        columns = response[0].keys()
        df_dict = {}
        for col in columns:
            df_dict[col] = []
        for r_dict in response:
            for col in columns:
                df_dict[col].append(r_dict[col])
        df = pd.DataFrame.from_dict(df_dict)
        return df    

    async def get_AllSymbols(self, connection):
        allsymbols = {"command": "getAllSymbols"}
        response = await self.sendMessage(connection, allsymbols)
        status = response["status"]
        print(f"{datetime.now()} | Response: {status}")
        response = response["returnData"]
        return response

    async def get_STC_Symbols(self, connection):
        allsymbols = self.get_AllSymbols(connection)
        STC = []
        for symbol in allsymbols["returnData"]:
            if symbol["categoryName"] == "STC":
                STC.append(symbol)
        return STC

    async def get_candles_range(self, connection, symbol, start, period, save=False):
        CHART_RANGE_INFO_RECORD = {
            "period": period,
            "start": date_to_xtb_time(start),
            "symbol": symbol
        }
        get_candles = {
            "command": "getChartLastRequest",
            "arguments": {"info": CHART_RANGE_INFO_RECORD},
        }
        response = await self.sendMessage(connection, get_candles)
        status = response["status"]
        response = response["returnData"]
        rate_infos = response["rateInfos"]
        # digits = response["digits"]
        # exemode = response["exemode"]
        print(f"{datetime.now()} | Response: {status}")
        df = self.return_as_df(rate_infos)

        df['ctm'] = pd.to_numeric(df['ctm'])
        df['ctmString'] = pd.to_datetime(df['ctmString'])
        df['open'] = pd.to_numeric(df['open'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['vol'] = pd.to_numeric(df['vol'])
        if save:
            df.to_pickle(f'../data/{symbol}_{start.strftime("%m-%d-%Y")}_{period}.pickle')

        return df