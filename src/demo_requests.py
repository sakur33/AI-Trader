from XTB_WS_CLIENT import XTBclient
import threading
import asyncio, nest_asyncio
import time
from datetime import datetime
import warnings
import json
import concurrent.futures
warnings.filterwarnings("ignore", category=DeprecationWarning)


async def main(client):
    global loop
    loop = asyncio.get_event_loop()
    connection = await client.connect()
    print(f"{datetime.now()}| Connected")

    await connection.send(json.dumps({"command": "getAllSymbols"}))
    response = await connection.recv()
    print(f"{datetime.now()} | Response: {response}")

    
    # with concurrent.futures.ThreadPoolExecutor() as pool:
    #     pers_tasks = [
    #         asyncio.ensure_future(client.heartbeat(connection)),
    #         asyncio.ensure_future(client.receiveMessage(connection)),
    #     ]
    #     result = await loop.run_in_executor(
    #         pool, exec, pers_tasks)

    # with concurrent.futures.ThreadPoolExecutor() as persistent_thread:
        
    # mssg_tasks = [
    #         asyncio.ensure_future(client.sendMessage(connection, "all_symbols")),
    #         asyncio.ensure_future(client.receiveMessage(connection)),
    #     ]

def exec(tasks):
    global loop
    loop.run_until_complete(asyncio.wait(tasks))

if __name__ == "__main__":
    client = XTBclient()
    asyncio.run(main(client))


