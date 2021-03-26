from typing import Any, Dict

from src.app import App
from src.server.celery import celery


class Tasks:
    @celery.task
    @staticmethod
    def collage(data: Dict[str, Any]) -> Dict[str, Any]:
        if "uris" not in data.keys():
            raise ValueError("uris missing")
        uris = data["uris"]
        rotate = data.get("rotate", None)
        output = App.collage(uris, rotate=rotate)
        return {"result": [output]}

    @celery.task
    @staticmethod
    def segment(data: Dict[str, Any]) -> Dict[str, Any]:
        if "uris" not in data.keys():
            raise ValueError("uris missing")
        uris = data["uris"]
        rotate = data.get("rotate", None)
        output = App.segment(uris, rotate=rotate)
        return {"result": list(output)}
