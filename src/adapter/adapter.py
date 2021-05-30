import os
from io import BytesIO
from typing import Any, Iterable

import numpy as np
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

from src.adapter.directory import DirectoryAdapter
from src.adapter.local_file import LocalFileAdapter
from src.logger import logger


class Adapter:
    @staticmethod
    def load(*inputs: str, img_mode: str = "RGB") -> Iterable[np.ndarray]:
        files = (file for inp in inputs for file in Adapter.__match(inp))
        imgs = (Image.open(file).convert(img_mode) for file in files)
        return (np.array(img, dtype=np.uint8) for img in imgs)

    @staticmethod
    def video(*inputs: str) -> Iterable[VideoFileClip]:
        for inp in inputs:
            try:
                yield VideoFileClip(inp)
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
