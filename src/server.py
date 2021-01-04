import os

from flask import Flask
from flask_cors import CORS  # type: ignore

from src.celery import celery
from src.controller import routes
from src.logger import logger

server = Flask(__name__)
celery.conf.update(server.config)
CORS(server, origins=str(os.getenv("ORIGINS", "")).split(","))
server.register_blueprint(routes)

logger.log("✂️")
