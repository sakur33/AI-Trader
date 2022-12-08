import websockets
import json
import asyncio
import time
from datetime import datetime, timedelta
import pytz
from creds import user, passw


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
    
    def date_to_xtb_time(self, target):
        initial = self.initial_time
        target = target.astimezone(self.timezone)
        diff = 1000 * (target - initial).seconds
        return diff

    def xtb_time_to_date(self, time):
        initial = self.initial_time
        date = initial + timedelta(milliseconds=time)
        return date

    def return_as_dict(self, response):
        columns = response[0].keys()
        df_dict = {}
        for col in columns:
            df_dict[col] = []

        for r_dict in response:
            for col in columns:
                df_dict[col] = r_dict[col]
        return df_dict     

    async def get_candles_range(self, connection, symbol, start, period):
        CHART_RANGE_INFO_RECORD = {
            "period": period,
            "start": self.date_to_xtb_time(start),
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
        rate_info_dict = self.return_as_dict(rate_infos)
        return rate_infos