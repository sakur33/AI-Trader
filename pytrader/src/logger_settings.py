import logging, logging.handlers
from trader_utils import get_today, get_today_ms
import os

today = get_today()
todayms, today_int = get_today_ms()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../../data/"
symbol_path = curr_path + "../../../symbols/"
cluster_path = curr_path + "../../../clusters/"
model_path = curr_path + "../../../model/"
result_path = curr_path + "../../../result/"
docs_path = curr_path + "../../../docs/"
database_path = curr_path + "../../../database/"
logs_path = curr_path + "../../../logs/"


def setup_logging(logger, path, name, console_debug=False):
    handler = logging.FileHandler(path + name + ".log")
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
    return logger
