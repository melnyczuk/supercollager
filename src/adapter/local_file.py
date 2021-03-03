import os
from io import BytesIO
from typing import Optional

from src.constants import VALID_EXTS


class LocalFileAdapter:
    @staticmethod
    def load(input: str) -> Optional[BytesIO]:
        if not os.path.isfile(input):
            raise ValueError("input is not local")
        if input.lower().split(".")[-1] not in VALID_EXTS:
            return None
        with open(input, "rb") as f:
            return BytesIO(f.read())
