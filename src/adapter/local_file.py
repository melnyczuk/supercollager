import os
from io import BytesIO
from typing import Optional

from src.constants import VALID_EXTS


class LocalFileAdapter:
    @staticmethod
    def load(path: str) -> Optional[BytesIO]:
        if not os.path.isfile(path):
            raise ValueError("input is not local")
        if path.lower().split(".")[-1] not in VALID_EXTS:
            return None
        with open(path, "rb") as f:
            return BytesIO(f.read())
