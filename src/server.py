import os

from flask import Flask
from flask_cors import CORS  # type: ignore

from .celery import celery
from .controller import routes
from .logger import logger

server = Flask(__name__)
celery.conf.update(server.config)
CORS(server, origins=str(os.getenv("ORIGINS", "")).split(","))
server.register_blueprint(routes)

logger.log("✂️")
