from typing import Any, Optional, Tuple

from celery.app.base import Celery
from celery.result import AsyncResult
from flask import Blueprint, Response, request
from flask.json import jsonify

from src.logger import logger
from src.server.celery import celery
from src.server.tasks import Tasks

routes = Blueprint("controller", __name__)

StandardResponse = Tuple[Response, int]


@routes.route("/segment", methods=["POST"])
def set_segment() -> StandardResponse:
    return set_task(Tasks.segment, request.json)


@routes.route("/segment/<id>", methods=["GET"])
def get_segment(id: str) -> StandardResponse:
    return get_task(id)


@routes.route("/ping", methods=["GET"])
def ping() -> StandardResponse:
    return make_response(data="pong")


@routes.errorhandler(400)
def four_hundred(e: object) -> StandardResponse:
    return make_response(error=str(e), status=400)


@routes.errorhandler(403)
def four_oh_three(e: object) -> StandardResponse:
    return make_response(error=str(e), status=403)


@routes.errorhandler(404)
def four_oh_four(e: object) -> StandardResponse:
    return make_response(error=str(e), status=404)


@routes.errorhandler(500)
def five_hundred(e: object) -> StandardResponse:
    return make_response(error=str(e), status=500)


def make_response(
    data: Optional[Any] = None,
    error: Optional[str] = None,
    status: int = 200,
) -> StandardResponse:
    logger.request(status)
    if data:
        return (jsonify({"data": data}), status)
    if error:
        return (jsonify({"error": error}), status)
    else:
        return (jsonify(), 204)


def set_task(task: Celery.task, data: Any) -> StandardResponse:
    try:
        logger.log(f"{data=}")
        result = task.delay(data)
        return make_response(data=result.id, status=202)
    except Exception as e:
        return make_response(error=str(e), status=500)


def get_task(id: str) -> StandardResponse:
    task = AsyncResult(id, app=celery)

    if not task.ready():
        return make_response(error=task.state, status=202)

    if task.failed():
        return make_response(error=task.state, status=500)

    return make_response(data=task.result)
