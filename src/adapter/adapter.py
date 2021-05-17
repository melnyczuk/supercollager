import os
from io import BytesIO
from typing import Any, Iterable

import numpy as np
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
from PIL import Image  # type: ignore

from src.adapter.directory import DirectoryAdapter
from src.adapter.local_file import LocalFileAdapter
from src.logger import logger


class Adapter:
    @staticmethod
    def load(*inp: str, img_mode: str = "RGB") -> Iterable[np.ndarray]:
        files = (file for i in inp for file in Adapter.__match(i))
        imgs = (Image.open(file).convert(img_mode) for file in files)
        return (np.array(img, dtype=np.uint8) for img in imgs)

    @staticmethod
    def video(*inp: str) -> Iterable[VideoFileClip]:
        for i in inp:
            try:
                yield VideoFileClip(i)
            except OSError as e:
                logger.error(str(e))

    @staticmethod
    def __match(inp: Any) -> Iterable[BytesIO]:
        if os.path.isfile(inp):
            if data := LocalFileAdapter.load(inp):
                yield data
        if os.path.isdir(inp):
            for data in DirectoryAdapter.load(inp):
                if data:
                    yield data
        if inp.startswith("http"):
            yield BytesIO(requests.get(inp).content)
