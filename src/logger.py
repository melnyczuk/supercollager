import logging
from flask import request
from dataclasses import dataclass


@dataclass(frozen=True)
class Logger:
    logger: logging.Logger = logging.getLogger()
    handler: logging.StreamHandler = logging.StreamHandler()

    def __init__(self: "Logger") -> None:
        self.logger.setLevel(logging.INFO)
        self.handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(message)s", datefmt="%Y/%m/%d-%H:%M:%S"
            )
        )
        self.logger.addHandler(self.handler)

    def request(self: "Logger", status: int = 200) -> None:
        self.log(f"{request.path} {status}")

    def log(self: "Logger", msg: str) -> None:
        self.logger.info(msg)
        self.handler.close()


logger = Logger()
