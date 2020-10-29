from typing import Dict

import src.app as app
from .celery import celery


@celery.task  # type: ignore
def a_task(data: Dict[str, any]) -> int:
    return 2 * 2
