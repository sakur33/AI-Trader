from XTB_WS_CLIENT import XTBclient
import threading
import asyncio, nest_asyncio
import time
from datetime import datetime
import warnings
import json
import concurrent.futures
from datetime import timedelta, datetime
import pandas as pd
import pickle as pkl
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=DeprecationWarning)

async def main(client):
    global loop
    loop = asyncio.get_event_loop()
    connection = await client.connect()
    print(f"{datetime.now()} | Connected")


    ## FIND SYMBOLS THAT ARE STOCKS

    df = await client.get_STC_Symbols(connection)

    ## GET CANDLES FOR A GIVEN SYMBOL FROM DATE UNTIL NOW AND PLOT
    # symbol = 'EURUSD'
    # start_date = (datetime.now() - timedelta(days=90))
    # period = 1440
    # df = await client.get_candles_range(
    #     connection, symbol=symbol,
    #     start=start_date, period=period
    # )


    # fig = go.Figure(data=[go.Candlestick(x=df['ctmString'],
    #             open=df['open'],
    #             high=df['open'] + df['high'],
    #             low=df['open'] + df['low'],
    #             close=df['open'] + df['close'])])
    # fig.show()

if __name__ == "__main__":
    client = XTBclient()
    asyncio.run(main(client))
