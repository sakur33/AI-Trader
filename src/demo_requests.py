from XTB_WS_CLIENT import XTBclient
import threading
import asyncio, nest_asyncio
import time
from datetime import datetime
import warnings
import json
import concurrent.futures
from datetime import timedelta
import pandas as pd
import pickle as pkl

warnings.filterwarnings("ignore", category=DeprecationWarning)

async def main(client):
    global loop
    loop = asyncio.get_event_loop()
    connection = await client.connect()
    print(f"{datetime.now()} | Connected")


    symbol = 'EURUSD'
    start_date = (datetime.now() - timedelta(days=90))
    period = 1440
    candles = await client.get_candles_range(
        connection, symbol=symbol,
        start=start_date, period=period
    )

    print(candles.head(20))

if __name__ == "__main__":
    client = XTBclient()
    asyncio.run(main(client))
