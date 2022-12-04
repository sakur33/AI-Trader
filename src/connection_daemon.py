import websocket
from threading import Thread
import _thread as thread
import time
import rel
import json
from creds import user, passw


class PersistentConnector:
    def __init__(self) -> None:
        self.uri = "wss://ws.xtb.com/demo"
        self.user = user
        self.passw = passw
        self.persistent_conn_th = None
        self.sender_th = None
        self.login = {
            "command": "login",
            "arguments": {
                "userId": self.user,
                "password": self.passw,
                "appId": "test",
                "appName": "test",
            },
        }
        self.ping = {"command": "ping"}
        
    def on_message(self, message):
        print(f"MESSAGE: {message}")

    def on_error(self, error_code, error_msg):
        print(f"{error_code}:{error_msg}")

    def on_close(self, close_status_code, close_msg):
        print(f"{close_status_code}:{close_msg}")

    def on_open(self, ws):
        self.send_message(message=self.login)
        print("Opened connection")
    
    def run(self):
        self.send_message(self.login)
        result = self.ws.recv()
        while(True):
            time.sleep(0.1)
            self.send_message(self.ping)
            result = self.ws.recv()

    def connect(self):
        self.ws = websocket.create_connection(self.uri)
        self.conn_th = Thread(target=self.run)
        self.conn_th.start()
        self.conn_th.join()

    def send(self, message):
        thread2 = Thread(target=self.send_message({"command": "ping"}))
        thread2.start()
        thread2.join()

    def send_message(self, message):
        self.ws.send(json.dumps(message))
