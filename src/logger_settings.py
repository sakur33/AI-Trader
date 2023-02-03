import logging, logging.handlers


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
