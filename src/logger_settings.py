import os
import logging

curr_path = os.path.dirname(os.path.realpath(__file__))
logs_path = curr_path + "../../logs/"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s : %(levelname)s : %(threadName)s : %(name)s %(message)s",
    filename=f"{logs_path}main_logger.log",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(
    "| %(threadName)s | %(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
