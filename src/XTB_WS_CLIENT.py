import websockets
import json
import asyncio
import time
from datetime import datetime
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
        self.logout = {"command": "logout"}
        self.ping = {"command": "ping"}
        self.commands = {"all_symbols": {"command": "getAllSymbols"}}

    async def connect(self):
        connection = await websockets.connect(self.uri, max_size=1_000_000_000)
        await connection.send(json.dumps(self.login))
        response = await connection.recv()
        print(f"{datetime.now()} | Response: {response}")
        return connection

    async def sendMessage(self, connection, command):
        """
        Sending message to webSocket server
        """
        try:
            await connection.send(json.dumps(self.commands[command]))
            response = await connection.recv()
            return response
        except websockets.exceptions.ConnectionClosed:
            print(f"{datetime.now()} | Message not sent.")
            print(f"{datetime.now()} | Connection with server closed.")
            return None
        except Exception as e:
                print(f"{datetime.now()} | No heartbeat")
                print(f"{datetime.now()} | {e}")


    async def receiveMessage(self, connection):
        """
        Receiving all server messages and handling them
        """
        while (True):
            try:
                response = await connection.recv()
                print(f"{datetime.now()} | Response: {response}")
            except websockets.exceptions.ConnectionClosed:
                print(f"{datetime.now()} | Message not received")
                print(f"{datetime.now()} | Connection with server closed")
                break
            except Exception as e:
                print(f"{datetime.now()} | No heartbeat")
                print(f"{datetime.now()} | {e}")
                break

    async def heartbeat(self, connection):
        """
        Sending heartbeat to server every 5 seconds
        Ping - pong messages to verify connection is alive
        """
        while (True):
            try:
                await connection.send(json.dumps(self.ping))
                await asyncio.sleep(0.2)
            except websockets.exceptions.ConnectionClosed:
                print(f"{datetime.now()} | No heartbeat")
                print(f"{datetime.now()} | Connection with server closed")
                break
            except Exception as e:
                print(f"{datetime.now()} | No heartbeat")
                print(f"{datetime.now()} | {e}")
                break

    async def disconnect(self):
        connection = await websockets.connect("wss://ws.xtb.com/demoStream")
        await connection.send(json.dumps(self.logout))
        response = await connection.recv()
        print(f"{datetime.now()} | Response: {response}")