from typing import Any, Dict, List

from src import pipelines
from .celery import celery


@celery.task  # type: ignore
def segment(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "url" not in data.keys():
        raise ValueError("url missing")
    try:
        return pipelines.segment(data["urls"])
    except Exception as e:
        raise e
