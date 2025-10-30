import logging

grey = "\033[90m"
yellow = "\033[33m"
red = "\033[31m"
bold_red = "\033[31m;1m"
reset = "\033[0m"
format = "%(asctime)s - %(message)s"

class CustomFormatter(logging.Formatter):


    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)