import os
from io import BytesIO
from typing import Iterable

from src.adapter.local_file import LocalFileAdapter


class DirectoryAdapter:
    @staticmethod
    def load(inp: str) -> Iterable[BytesIO]:
        if not os.path.isdir(inp):
            raise ValueError("input is not directory")
        for file in os.listdir(inp):
            if not os.path.isdir(uri := os.path.join(inp, file)):
                if data := LocalFileAdapter.load(uri):
                    yield data
            else:
                for data in DirectoryAdapter.load(uri):
                    if data:
                        yield data
