import logging, logging.handlers
from trader_utils import get_today, get_today_ms
import os


class CustomLogger:
    def __init__(self, test_name) -> None:
        self.today = get_today()
        self.todayms, self.today_int = get_today_ms()
        curr_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = curr_path + "../../../data/"
        self.symbol_path = curr_path + "../../../symbols/"
        self.cluster_path = curr_path + "../../../clusters/"
        self.model_path = curr_path + "../../../model/"
        self.result_path = curr_path + "../../../result/"
        self.docs_path = curr_path + "../../../docs/"
        self.database_path = curr_path + "../../../database/"
        self.logs_path = curr_path + "../../../logs/" + test_name + "/"
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

    def setup_logging(self, logger, name, console_debug=False):
        handler = logging.FileHandler(self.logs_path + name + ".log")
        logger.setLevel(logging.INFO)
        fileformatter = logging.Formatter(
            "| %(threadName)s | %(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(fileformatter)
        logger.addHandler(handler)

        if console_debug:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "| %(threadName)s | %(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
            console.setFormatter(formatter)
            logger.addHandler(console)
        self.logger = logger
        return logger

    def clean_old_files(self):
        files = os.listdir(self.logs_path)
        if len(files) > 0:
            for file in files:
                os.remove(self.logs_path + file)

    def get_logger(self):
        return self.logger
