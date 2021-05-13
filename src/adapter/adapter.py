import os
from io import BytesIO
from typing import Any, Iterable

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
from PIL import Image  # type: ignore

from src.adapter.directory import DirectoryAdapter
from src.adapter.local_file import LocalFileAdapter
from src.adapter.url import UrlAdapter


class Adapter:
    @staticmethod
    def load_one(inp: str, img_mode: str = "RGB") -> np.ndarray:
        file = Adapter.__match(inp)
        img = Image.open(file).convert(img_mode)
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def load(*inputs: str, img_mode: str = "RGB") -> Iterable[np.ndarray]:
        return (Adapter.load_one(inp, img_mode=img_mode) for inp in inputs)

    @staticmethod
    def video(input: str) -> VideoFileClip:
        return VideoFileClip(input)

    @staticmethod
    def __match(
        input: Any,
    ) -> Iterable[BytesIO]:
        if os.path.isfile(input):
            if data := LocalFileAdapter.load(input):
                yield data
            pass
        if os.path.isdir(input):
            for data in DirectoryAdapter.load(input):
                if data:
                    yield data
            pass
        if input.startswith("http"):
            for data in UrlAdapter.scrape_site(input):
                yield data
            pass
        pass
