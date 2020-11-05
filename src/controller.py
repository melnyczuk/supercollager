from flask import Blueprint, request
from flask.json import jsonify

from celery.app.base import Celery  # type: ignore
from celery.result import AsyncResult  # type: ignore

from typing import Any, Tuple, Union

from .celery import celery
from .logger import logger
from .tasks import segment

routes = Blueprint("controller", __name__)


@routes.route("/segment", methods=["POST"])
def set_segment() -> Tuple[str, int]:
    return set_task(segment, request.json)


@routes.route("/segment/<id>", methods=["GET"])
def get_segment(id: str) -> Tuple[str, int]:
    return get_task(id)


@routes.route("/ping", methods=["GET"])
def ping() -> Tuple[str, int]:
    return make_response(data="pong")


@routes.errorhandler(400)
def four_hundred(e: object) -> Tuple[str, int]:
    return make_response(error=str(e), status=400)


@routes.errorhandler(403)
def four_oh_three(e: object) -> Tuple[str, int]:
    return make_response(error=str(e), status=403)


@routes.errorhandler(404)
def four_oh_four(e: object) -> Tuple[str, int]:
    return make_response(error=str(e), status=404)


@routes.errorhandler(500)
def five_hundred(e: object) -> Tuple[str, int]:
    return make_response(error=str(e), status=500)


def make_response(
    data: Union[Any, None] = None,
    error: Union[str, None] = None,
    status: int = 200,
) -> Tuple[str, int]:
    logger.request(status)
    if data:
        return (jsonify({"data": data}), status)
    if error:
        return (jsonify({"error": error}), status)
    else:
        return (jsonify(), 204)


def set_task(task: Celery.task, data: Any) -> Tuple[str, int]:
    try:
        logger.log(f"{data=}")
        result = task.delay(data)
        return make_response(data=result.id, status=202)
    except Exception as e:
        return make_response(error=str(e), status=500)


def get_task(id: str) -> Tuple[str, int]:
    task = AsyncResult(id, app=celery)

    if not task.ready():
        return make_response(error=task.state, status=202)

    if task.failed():
        return make_response(error=task.state, status=500)

    return make_response(data=task.result)
