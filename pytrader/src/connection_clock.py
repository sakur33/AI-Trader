import time


class Clock:
    def __init__(self, interval=0.2) -> None:
        self.interval = interval
        self.last_message_time = time.time()

    def wait_clock(self):
        while time.time() - self.last_message_time < self.interval:
            pass
        self.last_message_time = time.time()
