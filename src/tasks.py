from typing import Any, Dict, List

from .app import Segmentation
from .celery import celery


@celery.task  # type: ignore
def segment(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "url" not in data.keys():
        raise ValueError("url missing")
    try:
        return Segmentation.from_url(data["url"])
    except Exception as e:
        raise e
