import os
from io import BytesIO
from typing import Iterable

from src.adapter.local_file import LocalFileAdapter


class DirectoryAdapter:
    @staticmethod
    def load(input: str) -> Iterable[BytesIO]:
        if not os.path.isdir(input):
            raise ValueError("input is not directory")
        for file in os.listdir(input):
            if not os.path.isdir(uri := os.path.join(input, file)):
                if data := LocalFileAdapter.load(uri):
                    yield data
            else:
                for data in DirectoryAdapter.load(uri):
                    if data:
                        yield data
