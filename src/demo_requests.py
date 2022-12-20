from XTB_WS_CLIENT import XTBclient
import asyncio
from datetime import datetime
import warnings
from datetime import timedelta, datetime
import os
import glob
from utils import get_today
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

today = get_today()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../data/"
symbol_path = curr_path + "../../symbols/"
cluster_path = curr_path + "../../clusters/"
model_path = curr_path + "../../model/"
result_path = curr_path + "../../result/"
docs_path = curr_path + "../../docs/"

paths = [data_path, symbol_path, cluster_path, model_path, docs_path]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)


async def main(client):
    if os.path.exists(data_path):
        picks = glob.glob(f"{data_path}*.pickle")
        if len(picks) > 0:
            print(f"Removing old data [{len(picks)}]...")
            for pick in tqdm(picks):
                os.remove(pick)
            print(f"... removed.")

    global loop
    loop = asyncio.get_event_loop()
    connection = await client.connect()
    print(f"{datetime.now()} | Connected")

    ## FIND SYMBOLS THAT ARE STOCKS
    # Categories = ['STC' 'CRT' 'ETF' 'IND' 'FX' 'CMD']
    df = await client.get_X_Symbols(connection, "STC", save=True)
    df.sort_values(by=["instantMaxVolume"], inplace=True)
    picks = list(df["symbol"][:50].values)

    for pick in picks:
        start_date = datetime.now() - timedelta(days=365)
        period = 1440
        df = await client.get_candles_range(
            connection, symbol=pick, start=start_date, period=period, save=True
        )

    # fig = go.Figure(data=[go.Candlestick(x=df['ctmString'],
    #             open=df['open'],
    #             high=df['open'] + df['high'],
    #             low=df['open'] + df['low'],
    #             close=df['open'] + df['close'])])
    # fig.show()


if __name__ == "__main__":
    client = XTBclient()
    asyncio.run(main(client))
