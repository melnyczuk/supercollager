from typing import Any, Dict, List

from .celery import celery
from .pipelines import LabelImage, collage, segment


def check(fn):
    @staticmethod
    def f(data: Dict[str, Any]) -> List[LabelImage]:
        if "uris" not in data.keys():
            raise ValueError("uris missing")
        try:
            return fn(**data)
        except Exception as e:
            raise e

    return f


class Tasks:
    collage = celery.task(check(collage))
    segment = celery.task(check(segment))
