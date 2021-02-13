import json
from dataclasses import is_dataclass
from os import environ
from typing import Any

import numpy as np
from celery import Celery  # type: ignore
from kombu.serialization import register  # type: ignore


class DataclassJsonEncoder(json.JSONEncoder):
    def default(self: "DataclassJsonEncoder", obj: Any) -> Any:
        if is_dataclass(obj):
            return obj.__dict__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


register(
    "dataclassjson",
    lambda obj: json.dumps(obj, cls=DataclassJsonEncoder),
    json.loads,
    content_type="application/json",
    content_encoding="utf-8",
)

celery: Celery = Celery(
    __name__,
    backend=str(environ["CELERY_BACKEND"]),
    broker=str(environ["CELERY_BROKER"]),
    include=["src.tasks"],
)

celery.conf.update({"CELERY_RESULT_SERIALIZER": "dataclassjson"})
