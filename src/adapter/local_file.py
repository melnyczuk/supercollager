import os
from io import BytesIO
from typing import Optional

from src.constants import VALID_EXTS


class LocalFileAdapter:
    @staticmethod
    def load(inp: str) -> Optional[BytesIO]:
        if not os.path.isfile(inp):
            raise ValueError("input is not local")
        if inp.lower().split(".")[-1] not in VALID_EXTS:
            return None
        with open(inp, "rb") as f:
            return BytesIO(f.read())
