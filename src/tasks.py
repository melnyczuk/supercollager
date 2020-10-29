import src.app as app
from .celery import celery


@celery.task  # type: ignore
def a_task() -> int:
    return 2 * 2
